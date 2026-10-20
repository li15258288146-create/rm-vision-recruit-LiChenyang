"""Register independent terrain using stable vertical structures; queries are never read."""
import json
from pathlib import Path
import numpy as np,cv2,yaml
from scipy.spatial import cKDTree
from scipy.optimize import differential_evolution
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from register_map import transform,feature_points
from reconstruct_map import UNKNOWN

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'qB_sentry/results/optimized'


def resample(source,source_cfg,target_shape,target_cfg,theta,translation):
    rr,cc=np.indices(target_shape)
    world=np.c_[cc.ravel()+.5,target_shape[0]-rr.ravel()-.5]*target_cfg['resolution']+np.array(target_cfg['origin'][:2])
    local=transform(world-np.asarray(translation),-theta,[0,0])
    cell=np.floor((local-np.array(source_cfg['origin'][:2]))/source_cfg['resolution']).astype(int)
    row,col=source.shape[0]-1-cell[:,1],cell[:,0]
    inside=(row>=0)&(row<source.shape[0])&(col>=0)&(col<source.shape[1])
    result=np.full(len(world),UNKNOWN,np.uint8);result[inside]=source[row[inside],col[inside]]
    return result.reshape(target_shape)


def main():
    scfg=yaml.safe_load((OUT/'terrain_map.yaml').read_text());source=cv2.imread(str(OUT/'terrain_map.pgm'),0)
    rcfg=yaml.safe_load((ROOT/'data/map/rmuc_2025.yaml').read_text());ref=cv2.imread(str(ROOT/'data/map'/rcfg['image']),0)
    a=np.load(OUT/'vertical_points.npy')[:,:2]
    a=np.unique(np.floor(a/.05).astype(int),axis=0)*.05+.025
    # Enclosing rectangular shell does not represent playable field boundary in reference.
    low,high=np.quantile(a,.005,axis=0),np.quantile(a,.995,axis=0)
    a=a[np.min(np.c_[a-low,high-a],axis=1)>.3]
    b=feature_points(ref,rcfg,True)
    blo,bhi=b.min(0),b.max(0);b=b[np.min(np.c_[b-blo,bhi-b],axis=1)>.3]
    # Deterministic holdout points are not used to optimize the transform.
    rng=np.random.default_rng(2026);rng.shuffle(a);train=a[::2];test=a[1::2];tree=cKDTree(b)
    old=json.loads((ROOT/'qB_sentry/results/advanced/registration.json').read_text())['candidates']
    results=[]
    for initial in old:
        def objective(v):
            moved=transform(train,v[0],v[1:]);forward=tree.query(moved)[0];backward=cKDTree(moved).query(b[::2])[0]
            return float(np.mean(np.minimum(forward,.3))+np.mean(np.minimum(backward,.3)))
        theta=initial['theta'];tx,ty=initial['translation']
        fit=differential_evolution(objective,[(theta-.04,theta+.04),(tx-.6,tx+.6),(ty-.6,ty+.6)],seed=13,popsize=12,maxiter=75,tol=1e-6)
        d=tree.query(transform(test,fit.x[0],fit.x[1:]))[0]
        results.append(dict(theta=float(fit.x[0]),angle_deg=float(np.degrees(fit.x[0])),translation=fit.x[1:].tolist(),objective=float(fit.fun),holdout_median_m=float(np.median(d)),holdout_within_10cm=float(np.mean(d<.1))))
    # Subcell translation refinement by global support overlap, never by query feasibility.
    # Checkerboard spatial blocks split scoring from validation to expose alignment fragility.
    rr,cc=np.indices(ref.shape);train_pixels=((rr//16+cc//16)%2)==0;rf=ref>240
    for candidate in results:
        initial=np.array(candidate['translation']);best_overlap=None
        for dx in np.arange(-.1,.1001,.025):
            for dy in np.arange(-.1,.1001,.025):
                t=initial+[dx,dy];sf=resample(source,scfg,ref.shape,rcfg,candidate['theta'],t)>240
                score=np.sum(sf&rf&train_pixels)/np.sum((sf|rf)&train_pixels)
                if best_overlap is None or score>best_overlap[0]:best_overlap=(score,t,sf)
        score,t,sf=best_overlap
        candidate['structural_translation']=candidate['translation'];candidate['translation']=t.tolist()
        candidate['support_train_iou']=float(score)
        candidate['support_holdout_iou']=float(np.sum(sf&rf&~train_pixels)/np.sum((sf|rf)&~train_pixels))
        d=tree.query(transform(test,candidate['theta'],t))[0]
        candidate['final_holdout_median_m']=float(np.median(d))
    # Near-symmetric orientations cannot be distinguished by sub-percent raster overlap.
    # Within 1 percentage point use the independent structural score as tie-break.
    top=max(r['support_train_iou'] for r in results)
    contenders=[r for r in results if top-r['support_train_iou']<.01]
    best=min(contenders,key=lambda r:r['objective']);theta=best['theta'];translation=best['translation']
    result=resample(source,scfg,ref.shape,rcfg,theta,translation)
    cv2.imwrite(str(OUT/'registered_map.pgm'),result);(OUT/'registered_map.yaml').write_text(yaml.safe_dump(rcfg|dict(image='registered_map.pgm')))
    c,s=np.cos(theta),np.sin(theta);T=np.array([[c,-s,translation[0]],[s,c,translation[1]],[0,0,1.]])
    sf,rf=result>240,ref>240;known=result!=UNKNOWN
    report=dict(best=best,candidates=results,orientation_ambiguous=len(contenders)>1,selection_rule='Within 1 percentage point training support IoU, tie-break by structural objective; no endpoint scoring',transform_aligned_to_reference=T.tolist(),source_feature_count=len(a),train_count=len(train),holdout_count=len(test),free_iou=float(np.sum(sf&rf)/np.sum(sf|rf)),agreement_known=float(np.mean(sf[known]==rf[known])),known_coverage=float(known.mean()),source='stable vertical structures; half source features held out; no query input')
    (OUT/'registration.json').write_text(json.dumps(report,indent=2))
    fig,axes=plt.subplots(3,1,figsize=(12,13),constrained_layout=True)
    overlay=np.ones((*ref.shape,3))*.8;overlay[rf]=1;overlay[~rf]=.2;overlay[sf&~rf]=[1,.4,.2];overlay[rf&~sf]=[.25,.65,1]
    for ax,im,title in zip(axes,[ref,result,overlay],['Reference','Reconstructed and registered','Red: self-only free / blue: reference-only free']):ax.imshow(im,cmap='gray',vmin=0,vmax=255 if im.ndim==2 else 1);ax.set_title(title)
    fig.savefig(OUT/'registration_overlay.png',dpi=140);plt.close(fig)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
