"""Evidence-based surface reconstruction from point geometry; no reference-map input."""
import argparse,json
from pathlib import Path
import numpy as np,cv2,yaml,open3d as o3d
from scipy.spatial import cKDTree,Delaunay
from scipy import ndimage
from scipy.signal import find_peaks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from terrain_advanced import classify_surface

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'qB_sentry/results/optimized'
UNKNOWN=128  # p=0.498 lies between free_thresh=.25 and occupied_thresh=.65.


def interpolate_surface(z,resolution,max_edge=1.2):
    valid=np.isfinite(z);rc=np.argwhere(valid);xy=rc[:,::-1]*resolution
    tri=Delaunay(xy);vertices=tri.simplices
    xyz=np.c_[xy,z[valid]];a,b,c=xyz[vertices[:,0]],xyz[vertices[:,1]],xyz[vertices[:,2]]
    longest=np.maximum.reduce([np.linalg.norm(a[:,:2]-b[:,:2],axis=1),np.linalg.norm(a[:,:2]-c[:,:2],axis=1),np.linalg.norm(b[:,:2]-c[:,:2],axis=1)])
    # Reconstruct height first; terrain_free distinguishes a short step from a long slope.
    allowed=(longest<=max_edge)&(np.ptp(xyz[vertices,2],axis=1)<=.12)
    missing=np.argwhere(~valid);points=missing[:,::-1]*resolution;simplex=tri.find_simplex(points)
    accepted=(simplex>=0)&allowed[np.maximum(simplex,0)]
    index=simplex[accepted];p=points[accepted]
    bary=np.einsum('nij,nj->ni',tri.transform[index,:2],p-tri.transform[index,2]);bary=np.c_[bary,1-bary.sum(1)]
    values=(xyz[vertices[index],2]*bary).sum(1)
    filled=z.copy();mask=np.zeros(z.shape,bool);r,c=missing[accepted].T;filled[r,c]=values;mask[r,c]=True
    return filled,mask


def terrain_free(z,obstacles,resolution):
    free,slope,edge=classify_surface(z,obstacles,resolution)
    # A discrete small step is not a continuous steep ramp: allow <=12 cm local jumps.
    valid=np.isfinite(z);small_step=np.zeros(z.shape,bool)
    for axis in [0,1]:
        diff=np.abs(np.diff(z,axis=axis));jump=np.isfinite(diff)&(diff>.04)&(diff<=.12)
        a=[slice(None)]*2;b=a.copy();a[axis]=slice(None,-1);b[axis]=slice(1,None)
        small_step[tuple(a)]|=jump;small_step[tuple(b)]|=jump
    small_step=ndimage.binary_dilation(small_step,iterations=2)
    hi=ndimage.maximum_filter(np.where(valid,z,-np.inf),size=5)
    lo=ndimage.minimum_filter(np.where(valid,z,np.inf),size=5)
    free|=valid&small_step&(hi-lo<=.120001)&~edge&~obstacles
    return free,slope,edge


def fill_planar_layers(observed,surface,resolution,max_edge=2.5,max_distance=.8):
    """Interpolate dominant horizontal layers separately; retain conflicting layers as unknown."""
    samples=observed[np.isfinite(observed)];edges=np.arange(-.065,1.216,.01)
    counts,_=np.histogram(samples,edges);smooth=ndimage.gaussian_filter1d(counts.astype(float),.7)
    peaks,_=find_peaks(smooth,distance=7,prominence=200)
    levels=[];predictions=[]
    for index in peaks:
        level=(edges[index]+edges[index+1])/2
        match=np.isfinite(observed)&(abs(observed-level)<.018)
        if match.sum()<500:continue  # Requires >=1.25 m² of direct 0.05 m cell evidence.
        layer=np.where(match,observed,np.nan)
        candidate,filled=interpolate_surface(layer,resolution,max_edge)
        distance=ndimage.distance_transform_edt(~match)*resolution
        candidate[(distance>max_distance)|~filled]=np.nan
        predictions.append(candidate);levels.append(dict(height_m=float(level),observed_cells=int(match.sum())))
    if not predictions:return surface.copy(),np.zeros(surface.shape,bool),levels
    stack=np.array(predictions);valid=np.isfinite(stack);n=valid.sum(0)
    low=np.min(np.where(valid,stack,np.inf),axis=0);high=np.max(np.where(valid,stack,-np.inf),axis=0)
    accepted=~np.isfinite(surface)&(n>0)&(high-low<.04)
    result=surface.copy();result[accepted]=low[accepted]
    return result,accepted,levels


def point_geometry(points):
    tree=cKDTree(points);eigenvalues=[];normalz=[];radii=[]
    for start in range(0,len(points),20000):
        d,idx=tree.query(points[start:start+20000],k=20,workers=-1)
        neighbors=points[idx];centered=neighbors-neighbors.mean(1,keepdims=True)
        cov=np.einsum('nki,nkj->nij',centered,centered)/20
        values,vectors=np.linalg.eigh(cov)
        eigenvalues.append(values);normalz.append(abs(vectors[:,2,0]));radii.append(d[:,-1])
    return np.concatenate(eigenvalues),np.concatenate(normalz),np.concatenate(radii)


def rasterize_surface(surface,p,eig,nz,radii,cells,res):
    reliable=(radii<.25)&(np.sqrt(np.maximum(eig[:,0],0))<.035)&(nz<.75)
    distance,index=ndimage.distance_transform_edt(~np.isfinite(surface),return_indices=True)
    base=surface[tuple(index)];base[distance*res>.25]=np.nan
    relative=p[:,2]-base[cells[:,1],cells[:,0]]
    hit=reliable&(relative>.12)&(relative<.65)
    obstacles=np.zeros(surface.shape,bool);obstacles[cells[hit,1],cells[hit,0]]=True
    free,slope,edge=terrain_free(surface,obstacles,res)
    raster=np.full(surface.shape,UNKNOWN,np.uint8);raster[np.isfinite(surface)&~free]=0;raster[obstacles]=0;raster[free]=255
    return raster,free,slope,edge,obstacles


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--max-edge',type=float,default=1.2);parser.add_argument('--layer-distance',type=float,default=.8);args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    cfg=yaml.safe_load((ROOT/'qB_sentry/results/pointcloud/generated_map.yaml').read_text())
    shape=tuple(json.loads((ROOT/'qB_sentry/results/pointcloud/report.json').read_text())['grid_shape']);res=cfg['resolution'];origin=np.array(cfg['origin'][:2])
    cache=OUT/'point_geometry.npz'
    if cache.exists():
        data=np.load(cache);p,eig,nz,radii=[data[k] for k in ['points','eigenvalues','normal_z','radius']]
    else:
        cloud=o3d.io.read_point_cloud(str(ROOT/'qB_sentry/results/pointcloud/leveled.ply'));p=np.asarray(cloud.points);p=p[(p[:,2]>-.15)&(p[:,2]<2.)]
        eig,nz,radii=point_geometry(p);np.savez_compressed(cache,points=p,eigenvalues=eig,normal_z=nz,radius=radii)
    planar=(np.sqrt(np.maximum(eig[:,0],0))<.018)&(radii<.45)&(eig[:,1]>.0001)
    support_points=planar&(nz>np.cos(np.radians(30)))&(p[:,2]>=-.06)&(p[:,2]<=1.2)
    cells=np.floor((p[:,:2]-origin)/res).astype(int);flat=cells[:,1]*shape[1]+cells[:,0]
    # Median rather than minimum: one below-surface return no longer creates a pit.
    ids=flat[support_points];heights=p[support_points,2];order=np.argsort(ids);ids,heights=ids[order],heights[order]
    unique,starts,counts=np.unique(ids,return_index=True,return_counts=True)
    z=np.full(np.prod(shape),np.nan)
    for idx,start,n in zip(unique,starts,counts):z[idx]=np.median(heights[start:start+n])
    z=z.reshape(shape);surface,interpolated=interpolate_surface(z,res,args.max_edge)
    surface,layer_filled,levels=fill_planar_layers(z,surface,res,max_distance=args.layer_distance)
    interpolated|=layer_filled
    # Detect dense non-horizontal surfaces independently from the support interpolation.
    raster,free,slope,edge,obstacles=rasterize_surface(surface,p,eig,nz,radii,cells,res)
    cv2.imwrite(str(OUT/'terrain_map.pgm'),np.flipud(raster));cfg['image']='terrain_map.pgm';(OUT/'terrain_map.yaml').write_text(yaml.safe_dump(cfg))
    np.savez_compressed(OUT/'terrain.npz',height=surface,observed_height=z,interpolated=interpolated,layer_filled=layer_filled,free=free,slope=slope,step=edge,obstacles=obstacles)
    # Registration uses reliable vertical structures, not all noisy occupancy cells.
    vertical=planar&(nz<.35)&(p[:,2]>.05)&(p[:,2]<1.5)
    np.save(OUT/'vertical_points.npy',p[vertical])
    report=dict(input_points=len(p),reliable_support_points=int(support_points.sum()),observed_cells=int(np.isfinite(z).sum()),interpolated_cells=int(interpolated.sum()),layer_filled_cells=int(layer_filled.sum()),dominant_layers=levels,free_cells=int(free.sum()),unknown_cells=int((raster==UNKNOWN).sum()),obstacle_cells=int(obstacles.sum()),step_cells=int(edge.sum()),max_triangle_edge_m=args.max_edge,layer_distance_m=args.layer_distance,layer_triangle_edge_m=2.5,unknown_gray=UNKNOWN,source='leveled point cloud only; no reference map or queries')
    (OUT/'reconstruction.json').write_text(json.dumps(report,indent=2))
    fig,axes=plt.subplots(1,4,figsize=(17,8),constrained_layout=True)
    for ax,im,title in zip(axes,[z,surface,np.where(obstacles,2,np.where(edge,1,0)),raster],['Reliable observed support','Constrained surface interpolation','Step=1 / obstacle=2','Reconstructed independent map']):
        artist=ax.imshow(im,origin='lower');ax.set_title(title);fig.colorbar(artist,ax=ax,shrink=.6)
    fig.savefig(OUT/'reconstruction.png',dpi=140);plt.close(fig);print(json.dumps(report,indent=2))

if __name__=='__main__':main()
