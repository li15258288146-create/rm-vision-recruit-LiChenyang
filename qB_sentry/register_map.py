"""Metric SE(2) registration; never copies reference occupancy into generated terrain."""
import json
from pathlib import Path
import cv2
import numpy as np
import yaml
from scipy.optimize import differential_evolution
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'qB_sentry/results/advanced'


def transform(points,theta,translation):
    c,s=np.cos(theta),np.sin(theta);R=np.array([[c,-s],[s,c]])
    return np.asarray(points)@R.T+translation


def register_points(a,b,theta,initial):
    tree=cKDTree(b)
    def score(v):
        moved=transform(a,v[0],v[1:]);d=tree.query(moved)[0]
        back=cKDTree(moved).query(b[::3])[0]
        return float(np.mean(np.minimum(d,.6))+np.mean(np.minimum(back,.6)))
    return differential_evolution(score,[(theta-.06,theta+.06),(initial[0]-1.2,initial[0]+1.2),(initial[1]-1.2,initial[1]+1.2)],seed=12,popsize=10,maxiter=60,tol=1e-5,polish=True)


def feature_points(image,cfg,reference=False):
    if reference:
        mask=(image>240).astype(np.uint8)
        mask=mask-cv2.erode(mask,np.ones((3,3),np.uint8))
    else:
        mask=(image<80).astype(np.uint8)
        n,label,stats,_=cv2.connectedComponentsWithStats(mask,8)
        mask=(stats[label,cv2.CC_STAT_AREA]>=12)&(label>0)
    row,col=np.where(mask)
    return np.c_[col+.5,image.shape[0]-row-.5]*cfg['resolution']+np.array(cfg['origin'][:2])


def main():
    src_cfg=yaml.safe_load((OUT/'terrain_map.yaml').read_text())
    ref_cfg=yaml.safe_load((ROOT/'data/map/rmuc_2025.yaml').read_text())
    src=cv2.imread(str(OUT/src_cfg['image']),0);ref=cv2.imread(str(ROOT/'data/map'/ref_cfg['image']),0)
    a=feature_points(src,src_cfg);b=feature_points(ref,ref_cfg,True)
    rng=np.random.default_rng(42);a=a[rng.choice(len(a),min(7000,len(a)),replace=False)]
    tree=cKDTree(b)
    ca=(a.min(0)+a.max(0))/2;cb=(b.min(0)+b.max(0))/2
    candidates=[]
    for degree in [-90,90]:
        theta=np.radians(degree);initial=cb-transform(ca[None],theta,[0,0])[0]
        result=register_points(a,b,theta,initial)
        moved=transform(a,result.x[0],result.x[1:]);dist=tree.query(moved)[0]
        candidates.append(dict(angle_deg=float(np.degrees(result.x[0])),theta=float(result.x[0]),translation=result.x[1:].tolist(),objective=float(result.fun),median_residual_m=float(np.median(dist)),fraction_within_15cm=float(np.mean(dist<.15))))
    best=min(candidates,key=lambda x:x['objective'])
    theta=best['theta'];translation=np.array(best['translation']);c,s=np.cos(theta),np.sin(theta)
    T=np.eye(3);T[:2,:2]=[[c,-s],[s,c]];T[:2,2]=translation
    # Resample generated map into reference world extent (nearest neighbor, unknown fill).
    rr,cc=np.indices(ref.shape);world=np.c_[cc.ravel()+.5,ref.shape[0]-rr.ravel()-.5]*ref_cfg['resolution']+np.array(ref_cfg['origin'][:2])
    local=(world-translation)@T[:2,:2]
    cells=np.floor((local-np.array(src_cfg['origin'][:2]))/src_cfg['resolution']).astype(int)
    row=src.shape[0]-1-cells[:,1];col=cells[:,0];inside=(row>=0)&(row<src.shape[0])&(col>=0)&(col<src.shape[1])
    registered=np.full(world.shape[0],128,np.uint8);registered[inside]=src[row[inside],col[inside]];registered=registered.reshape(ref.shape)
    cv2.imwrite(str(OUT/'registered_map.pgm'),registered)
    cfg=ref_cfg|dict(image='registered_map.pgm');(OUT/'registered_map.yaml').write_text(yaml.safe_dump(cfg),encoding='utf-8')
    known=registered!=128;rf=ref>240;sf=registered>240
    report=dict(best=best,candidates=candidates,transform_aligned_to_reference=T.tolist(),free_iou=float(np.sum(sf&rf)/np.sum(sf|rf)),agreement_on_known_cells=float(np.mean(sf[known]==rf[known])),known_coverage=float(known.mean()),warning='Near 180-degree field symmetry: rank both candidates; residual is map agreement, not survey accuracy.')
    (OUT/'registration.json').write_text(json.dumps(report,indent=2))
    overlay=np.ones((*ref.shape,3),float)*.85;overlay[rf]=[1,1,1];overlay[~rf]=[.2,.2,.2];overlay[sf&~rf]=[1,.35,.2];overlay[rf&~sf]=[.25,.65,1]
    fig,axes=plt.subplots(3,1,figsize=(12,15),constrained_layout=True)
    for ax,im,title in zip(axes,[ref,registered,overlay],['Provided map','Registered independent terrain','Red: self-only free; blue: reference-only free']):ax.imshow(im,cmap='gray',vmin=0,vmax=255 if im.ndim==2 else 1);ax.set_title(title)
    fig.savefig(OUT/'registration_overlay.png',dpi=140);plt.close(fig)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
