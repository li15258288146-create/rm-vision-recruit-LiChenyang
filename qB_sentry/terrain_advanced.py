"""Local support surface, bounded hole interpolation, and terrain evidence."""
import json
from pathlib import Path
import cv2
import numpy as np
import open3d as o3d
import yaml
from scipy import ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'qB_sentry/results/advanced'


def fill_small_holes(z,resolution,max_radius=.15):
    valid=np.isfinite(z)
    distance,indices=ndimage.distance_transform_edt(~valid,return_indices=True)
    nearest=z[tuple(indices)]
    # Locally bridge scan sampling gaps; never infer large unobserved interiors.
    enclosed=ndimage.binary_closing(valid,structure=np.ones((7,7)))
    count=ndimage.convolve(valid.astype(float),np.ones((7,7)),mode='constant')
    small=enclosed&(distance*resolution<=max_radius)&~valid&(count>=12)
    hi=ndimage.maximum_filter(np.where(valid,z,-np.inf),size=7)
    lo=ndimage.minimum_filter(np.where(valid,z,np.inf),size=7)
    small&=(hi-lo<=.10)
    filled=z.copy();filled[small]=nearest[small]
    return filled,small


def classify_surface(z,obstacles,resolution,slope_limit=30):
    valid=np.isfinite(z);edge=np.zeros(z.shape,bool)
    for axis in [0,1]:
        a=[slice(None)]*2;b=a.copy();a[axis]=slice(None,-1);b[axis]=slice(1,None)
        a,b=tuple(a),tuple(b)
        delta=np.zeros_like(z[a]);v=valid[a]&valid[b]
        np.subtract(z[a],z[b],out=delta,where=v)
        jump=v&(np.abs(delta)>.12);edge[a]|=jump;edge[b]|=jump
    v=valid.astype(float);height=np.where(valid,z,0)
    xx,yy=np.meshgrid(np.arange(-2,3)*resolution,np.arange(-2,3)*resolution)
    kernels=[np.ones((5,5)),xx,yy,xx*xx,xx*yy,yy*yy]
    n,sx,sy,sxx,sxy,syy=[cv2.filter2D(v,-1,k,borderType=cv2.BORDER_CONSTANT) for k in kernels]
    sz,sxz,syz=[cv2.filter2D(height,-1,k,borderType=cv2.BORDER_CONSTANT) for k in kernels[:3]]
    A=np.stack([sxx,sxy,sx,sxy,syy,sy,sx,sy,n],-1).reshape(*z.shape,3,3)
    b=np.stack([sxz,syz,sz],-1)
    good=valid&(n>=12)&(np.abs(np.linalg.det(A))>1e-10)
    coeff=np.linalg.solve(A[good],b[good,:,None])[...,0]
    slopes=np.full(z.shape,np.nan);slopes[good]=np.degrees(np.arctan(np.hypot(coeff[:,0],coeff[:,1])))
    return good&~obstacles&~edge&(slopes<=slope_limit),slopes,edge


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    p=np.asarray(o3d.io.read_point_cloud(str(ROOT/'qB_sentry/results/pointcloud/leveled.ply')).points)
    cfg=yaml.safe_load((ROOT/'qB_sentry/results/pointcloud/generated_map.yaml').read_text())
    origin=np.array(cfg['origin'][:2]);res=cfg['resolution']
    cell=np.floor((p[:,:2]-origin)/res).astype(int)
    shape=tuple(json.loads((ROOT/'qB_sentry/results/pointcloud/report.json').read_text())['grid_shape'])
    flat=cell[:,1]*shape[1]+cell[:,0]
    z=np.full(np.prod(shape),np.inf);np.minimum.at(z,flat,p[:,2]);z=z.reshape(shape)
    z[(z<-.12)|(z>1.2)]=np.nan
    support,interpolated=fill_small_holes(z,res)
    obstacles=np.zeros(shape,bool)
    # Body collision above local support. Keep tall walls as obstacles, irrespective of ceiling crop.
    base=ndimage.minimum_filter(np.where(np.isfinite(support),support,np.inf),size=3)
    rel=p[:,2]-base[cell[:,1],cell[:,0]]
    hit=(rel>.12)&(rel<.65)
    obstacles.ravel()[np.unique(flat[hit])]=True
    free,slope,edge=classify_surface(support,obstacles,res)
    raster=np.full(shape,128,np.uint8);raster[np.isfinite(support)&~free]=0;raster[obstacles]=0;raster[free]=255
    cv2.imwrite(str(OUT/'terrain_map.pgm'),np.flipud(raster))
    cfg['image']='terrain_map.pgm';(OUT/'terrain_map.yaml').write_text(yaml.safe_dump(cfg),encoding='utf-8')
    np.savez_compressed(OUT/'terrain.npz',height=support,slope=slope,step=edge,free=free,interpolated=interpolated,obstacles=obstacles)
    # Diagnose nonphysical above-field shell separately; preserve source and wall footprint.
    high=p[:,2]>3.
    edge_distance=np.minimum.reduce([p[:,0]-np.quantile(p[:,0],.005),np.quantile(p[:,0],.995)-p[:,0],p[:,1]-np.quantile(p[:,1],.005),np.quantile(p[:,1],.995)-p[:,1]])
    shell=high&(edge_distance<.25)
    cleaned=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p[~shell]))
    o3d.io.write_point_cloud(str(OUT/'navigation_cloud.ply'),cleaned)
    o3d.io.write_point_cloud(str(OUT/'excluded_upper_shell.ply'),o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p[shell])))
    report=dict(interpolated_cells=int(interpolated.sum()),free_cells=int(free.sum()),step_edge_cells=int(edge.sum()),ramp_cells=int((free&(slope>3)&(slope<=30)).sum()),upper_shell_points=int(shell.sum()),above_3m_points=int(high.sum()),above_3m_near_boundary_fraction=float(shell.sum()/high.sum()),max_height_m=float(p[:,2].max()),exclusion_rule='z>3 m and within 0.25 m of robust outer boundary; retain low wall footprint',cause='Likely simulation enclosing wall geometry; static merged cloud cannot establish provenance or prove dynamic objects.',source_untouched=True)
    (OUT/'terrain_report.json').write_text(json.dumps(report,indent=2))
    fig,axes=plt.subplots(1,4,figsize=(17,7),constrained_layout=True)
    for ax,a,title in zip(axes,[support,slope,np.where(edge,1,np.where(free&(slope>3),2,0)),raster],['Support height (m)','Slope (degrees)','Step=1 / ramp=2','Independent terrain map']):
        im=ax.imshow(a,origin='lower');ax.set_title(title);fig.colorbar(im,ax=ax,shrink=.6)
    fig.savefig(OUT/'terrain_validation.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    for ax,mask,title in zip(axes,[np.ones(len(p),bool),~shell],['Before: tall enclosure','Navigation cloud: upper shell excluded']):
        s=p[mask][::12];ax.scatter(s[:,1],s[:,2],s=.1);ax.set(xlabel='aligned Y (m)',ylabel='height (m)',title=title,ylim=(-.5,19))
    fig.savefig(OUT/'anomaly_before_after.png',dpi=160);plt.close(fig)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
