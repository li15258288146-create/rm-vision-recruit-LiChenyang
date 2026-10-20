"""灯条几何检测、平面 PnP 和匀速卡尔曼滤波基础实现。"""
import math
import cv2
import numpy as np

OBJECT_POINTS=np.array([[-.0675,-.0275,0],[-.0675,.0275,0],[.0675,.0275,0],[.0675,-.0275,0]],np.float64)


def detect(image, color='red', brightness=140, difference=60):
    b,g,r=cv2.split(image)
    channel,other=(r,b) if color=='red' else (b,r)
    mask=((channel.astype(np.int16)-other.astype(np.int16)>difference)&(channel>brightness)).astype(np.uint8)*255
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    lights=[]
    for contour in contours:
        if cv2.contourArea(contour)<4 or len(contour)<8:continue
        xy=contour[:,0].astype(float);center=xy.mean(0)
        _,_,vectors=np.linalg.svd(xy-center,full_matrices=False)
        direction=vectors[0]
        if direction[1]<0:direction=-direction
        angle=math.degrees(math.atan2(direction[0],direction[1]))
        projection=(xy-center)@direction
        width=np.ptp((xy-center)@vectors[1]);length=np.ptp(projection)
        if length<8 or length/max(width,1)<2.2 or abs(angle)>35:continue
        lights.append(dict(center=center,top=center+direction*projection.min(),bottom=center+direction*projection.max(),length=length,angle=angle))
    lights.sort(key=lambda a:a['center'][0]);pairs=[]
    for i,left in enumerate(lights):
        for j in range(i+1,len(lights)):
            right=lights[j];height=(left['length']+right['length'])/2
            dx=(right['center'][0]-left['center'][0])/height
            dy=abs(right['center'][1]-left['center'][1])/height
            ratio=min(left['length'],right['length'])/max(left['length'],right['length'])
            angle=abs(left['angle']-right['angle'])
            if not (.55<dx<3.6 and dy<.55 and ratio>.55 and angle<22):continue
            if any(abs(lights[k]['center'][1]-(left['center'][1]+right['center'][1])/2)<height*.6 for k in range(i+1,j)):continue
            score=(1-ratio)+dy+.02*angle+.08*abs(dx-2.45)
            pairs.append((score,i,j,np.array([left['top'],left['bottom'],right['bottom'],right['top']])))
    used=set();detections=[]
    for score,i,j,points in sorted(pairs,key=lambda a:a[0]):
        if i in used or j in used:continue
        used.update([i,j]);detections.append(dict(points=points,score=score))
    return detections,mask


def pose(points,K,dist):
    ok,rvecs,tvecs,_=cv2.solvePnPGeneric(OBJECT_POINTS,np.ascontiguousarray(points,np.float64),K,dist,flags=cv2.SOLVEPNP_IPPE)
    if not ok:return None
    choices=[]
    for r,t in zip(rvecs,tvecs):
        R,_=cv2.Rodrigues(r)
        if np.any((R@OBJECT_POINTS.T+t)[2]<=0):continue
        projected,_=cv2.projectPoints(OBJECT_POINTS,r,t,K,dist)
        rms=float(np.sqrt(np.mean(np.sum((projected.reshape(4,2)-points)**2,axis=1))))
        # 对象 +Z 法向在相机 x-z 平面中的方向；正对相机时约为零。
        yaw=float(math.atan2(R[0,2],R[2,2]))
        choices.append(dict(position=t.ravel(),yaw=yaw,rms=rms))
    return min(choices,key=lambda x:x['rms']) if choices else None


def filter_positions(observations,dt=1/150,horizon=.25,q=300.,r=.001):
    n=len(observations);filtered=np.full((n,3),np.nan);predicted=filtered.copy();accepted=np.zeros(n,bool)
    F=np.eye(6);F[:3,3:]=np.eye(3)*dt;H=np.zeros((3,6));H[:,:3]=np.eye(3)
    G=np.vstack([np.eye(3)*dt**2/2,np.eye(3)*dt]);Q=G@G.T*q;R=np.eye(3)*r
    state=None;P=None;gap=0
    for i,z in enumerate(observations):
        valid=np.isfinite(z).all()
        if state is None:
            if not valid:continue
            state=np.r_[z,[0.,0.,0.]];P=np.diag([r]*3+[4.]*3);accepted[i]=True;gap=0
        else:
            state=F@state;P=F@P@F.T+Q;gap+=1
            if valid:
                residual=z-H@state;S=H@P@H.T+R
                if residual@np.linalg.solve(S,residual)<25:
                    gain=np.linalg.solve(S,(P@H.T).T).T;state+=gain@residual
                    M=np.eye(6)-gain@H;P=M@P@M.T+gain@R@gain.T;accepted[i]=True;gap=0
            if gap>75:state=None;continue
        filtered[i]=state[:3];predicted[i]=state[:3]+state[3:]*horizon
    return filtered,predicted,accepted
