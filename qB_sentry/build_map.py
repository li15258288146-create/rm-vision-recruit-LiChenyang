"""点云预处理、候选地面选择、场地对正和保守的 2.5D 通行栅格。"""
import json
from pathlib import Path
import cv2
import numpy as np
import open3d as o3d
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]


def main():
    out=ROOT/'qB_sentry/results/pointcloud';out.mkdir(parents=True,exist_ok=True)
    o3d.utility.random.seed(7)
    raw=o3d.io.read_point_cloud(str(ROOT/'data/rmuc_2025_map.pcd'))
    raw_count=len(raw.points)
    down=raw.voxel_down_sample(.05)
    cloud,kept=down.remove_statistical_outlier(nb_neighbors=20,std_ratio=2.)
    points=np.asarray(cloud.points)
    remaining=cloud;planes=[]
    for i in range(8):
        coeff,indices=remaining.segment_plane(distance_threshold=.035,ransac_n=3,num_iterations=1000)
        coeff=np.asarray(coeff)
        if coeff[2]<0:coeff=-coeff
        plane_points=np.asarray(remaining.points)[indices]
        planes.append(dict(index=i,coefficients=coeff.tolist(),count=len(indices),centroid=plane_points.mean(0).tolist()))
        remaining=remaining.select_by_index(indices,invert=True)
    # 约 20° 倾斜安装、地面应在雷达下方约 0.4 m；最大面不直接当地面。
    candidates=[p for p in planes if p['coefficients'][2]>.85 and .30<p['coefficients'][3]<.55]
    if not candidates:raise RuntimeError('No ground candidate satisfies installation prior; inspect planes')
    ground=max(candidates,key=lambda p:p['count']);coeff=np.array(ground['coefficients']);normal=coeff[:3]
    # 用最大墙面法向投影作为水平轴，场地边界对正。
    wall=np.array(planes[0]['coefficients'][:3]);axis_x=wall-normal*np.dot(wall,normal);axis_x/=np.linalg.norm(axis_x)
    axis_y=np.cross(normal,axis_x)
    rotation=np.vstack([axis_x,axis_y,normal]);translation=np.array([0,0,coeff[3]])
    aligned=points@rotation.T+translation
    transform=np.eye(4);transform[:3,:3]=rotation;transform[:3,3]=translation
    leveled=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(aligned))
    o3d.io.write_point_cloud(str(out/'leveled.ply'),leveled)
    ground_mask=np.abs(aligned[:,2])<.035
    o3d.io.write_point_cloud(str(out/'ground.ply'),leveled.select_by_index(np.flatnonzero(ground_mask)))
    res=.05;origin=np.floor(aligned[:,:2].min(0)/res)*res-res
    cell=np.floor((aligned[:,:2]-origin)/res).astype(int)
    width,height=cell.max(0)+2;shape=(int(height),int(width))
    flat=cell[:,1]*width+cell[:,0];size=int(width*height)
    zmin=np.full(size,np.inf);zmax=np.full(size,-np.inf);count=np.zeros(size,int)
    np.minimum.at(zmin,flat,aligned[:,2]);np.maximum.at(zmax,flat,aligned[:,2]);np.add.at(count,flat,1)
    zmin=zmin.reshape(shape);zmax=zmax.reshape(shape);count=count.reshape(shape)
    observed=count>0
    # 小范围邻域只用于地面高度估计，不把无观测格直接补成可行驶。
    low=cv2.erode(zmin.astype(np.float32),np.ones((3,3),np.uint8))
    support=np.isfinite(low)&(low>=-.12)&(low<=1.2)
    occupied=np.zeros(shape,bool);local_base=low[cell[:,1],cell[:,0]]
    obstacle_points=(aligned[:,2]-local_base>.12)&(aligned[:,2]-local_base<.65)
    occupied.ravel()[np.unique(flat[obstacle_points])]=True
    # 邻接表面突变大于 12 cm 视为台阶边缘；坡道依靠连续小高度差。
    edge=np.zeros(shape,bool)
    for dr,dc in [(0,1),(1,0)]:
        a=(slice(0,shape[0]-dr),slice(0,shape[1]-dc))
        b=(slice(dr,shape[0]),slice(dc,shape[1]))
        valid=np.isfinite(zmin[a])&np.isfinite(zmin[b])
        delta=np.zeros_like(zmin[a]);np.subtract(zmin[a],zmin[b],out=delta,where=valid)
        jump=valid&(np.abs(delta)>.12)
        edge[a]|=jump;edge[b]|=jump
    # 每格邻近 5x5 区域的局部平面拟合，避免对 NaN 求梯度。
    surface=observed&support&(~occupied)
    slopes=np.full(shape,np.nan)
    valid=surface.astype(np.float64);z=np.where(surface,zmin,0.)
    coordinates=np.arange(-2,3)*res;xx,yy=np.meshgrid(coordinates,coordinates)
    kernels=[np.ones((5,5)),xx,yy,xx*xx,xx*yy,yy*yy]
    sums=[cv2.filter2D(valid,-1,k,borderType=cv2.BORDER_CONSTANT) for k in kernels]
    sz,sxz,syz=[cv2.filter2D(z,-1,k,borderType=cv2.BORDER_CONSTANT) for k in kernels[:3]]
    n,sx,sy,sxx,sxy,syy=sums
    eligible=surface&(n>=6)
    rr,cc=np.where(eligible)
    matrices=np.stack([sxx[eligible],sxy[eligible],sx[eligible],sxy[eligible],syy[eligible],sy[eligible],sx[eligible],sy[eligible],n[eligible]],axis=-1).reshape(-1,3,3)
    rhs=np.stack([sxz[eligible],syz[eligible],sz[eligible]],axis=-1)
    nondeg=np.abs(np.linalg.det(matrices))>1e-10
    coefficients=np.linalg.solve(matrices[nondeg],rhs[nondeg,:,None])[...,0]
    slopes[rr[nondeg],cc[nondeg]]=np.degrees(np.arctan(np.hypot(coefficients[:,0],coefficients[:,1])))
    free=surface&(~edge)&(slopes<=30)
    # 30° 为工程假设，题目没有给出底盘最大爬坡角。
    raster=np.full(shape,128,np.uint8)
    raster[observed&(~free)]=0;raster[free]=255
    image=np.flipud(raster)
    cv2.imwrite(str(out/'generated_map.pgm'),image)
    config=dict(image='generated_map.pgm',resolution=res,origin=[float(origin[0]),float(origin[1]),0.],negate=0,occupied_thresh=.65,free_thresh=.25)
    (out/'generated_map.yaml').write_text(yaml.safe_dump(config,sort_keys=False),encoding='utf-8')
    np.savez_compressed(out/'terrain.npz',min_height=zmin,max_height=zmax,slope_degrees=slopes,count=count,step_edges=edge,free=free)
    report=dict(raw_points=raw_count,downsampled_points=len(down.points),denoised_points=len(cloud.points),planes=planes,ground_candidate=ground,transform_lidar_to_aligned=transform.tolist(),ground_residual_rms_m=float(np.sqrt(np.mean(aligned[ground_mask,2]**2))),resolution_m=res,grid_shape=list(shape),free_cells=int(free.sum()),occupied_cells=int((raster==0).sum()),unknown_cells=int((raster==128).sum()),assumptions=dict(step_limit_m=.12,body_clearance_m=.6,slope_limit_degrees=30,max_support_height_m=1.2),limitations=['No sensor ray history: unseen cells remain unknown.','Ground uses installation prior and plane geometry; maximum plane is a wall.','Single surface per cell: overhangs/multiple levels may be conservatively blocked.','Aligned pointcloud frame is not yet registered to the supplied PGM world frame.','No arbitrary deletion of dense objects; isolated points removed statistically.'])
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    sample=aligned[::max(1,len(aligned)//80000)]
    fig=plt.figure(figsize=(14,5));a=fig.add_subplot(121,projection='3d');a.scatter(sample[:,0],sample[:,1],sample[:,2],c=sample[:,2],s=.2,cmap='viridis');a.set(xlabel='X (m)',ylabel='Y (m)',zlabel='Z (m)',title='Leveled point cloud');a.set_box_aspect(np.ptp(sample,axis=0))
    b=fig.add_subplot(122);b.hist(aligned[:,2],bins=150);b.set(xlabel='Height above fitted ground (m)',ylabel='Point count',title='Height distribution');fig.tight_layout();fig.savefig(out/'pointcloud_overview.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,5),constrained_layout=True)
    axes[0].imshow(np.flipud(np.where(observed,zmin,np.nan)),vmin=-.1,vmax=1.2,cmap='terrain');axes[0].set_title('Minimum observed height (-0.1 to 1.2 m)')
    axes[1].imshow(np.flipud(slopes),vmin=0,vmax=45,cmap='viridis');axes[1].set_title('Local surface slope (degrees)')
    axes[2].imshow(image,cmap='gray',vmin=0,vmax=255);axes[2].set_title('Free / blocked / unknown')
    fig.savefig(out/'terrain_maps.png',dpi=170);plt.close(fig)
    print(json.dumps({k:v for k,v in report.items() if k not in ['planes','limitations']},indent=2))


if __name__=='__main__':main()
