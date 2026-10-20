"""Holdout surface check and global model/registration sensitivity, no endpoint tuning."""
import json
from pathlib import Path
import numpy as np,cv2,yaml
from reconstruct_map import interpolate_surface,fill_planar_layers,rasterize_surface
from refine_registration import resample
from planner import GridMap,search
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'qB_sentry/results/optimized'


def query_status(im,cfg,queries):
    grid=GridMap(im,cfg['resolution'],cfg['origin'],cfg['free_thresh']);rows=[]
    for i,q in enumerate(queries[:5],1):
        a,b=grid.world_to_cell(q['start']),grid.world_to_cell(q['goal'])
        why='; '.join(x for x in [grid.reason(a),grid.reason(b)] if x)
        if not why:why='reachable' if search(grid.free,a,b)['path'] else 'disconnected'
        rows.append(dict(query=i,status=why))
    return rows


def main():
    terrain=np.load(OUT/'terrain.npz');z=terrain['observed_height'];scfg=yaml.safe_load((OUT/'terrain_map.yaml').read_text());rcfg=yaml.safe_load((ROOT/'data/map/rmuc_2025.yaml').read_text())
    ref=cv2.imread(str(ROOT/'data/map'/rcfg['image']),0);source=cv2.imread(str(OUT/'terrain_map.pgm'),0)
    best=json.loads((OUT/'registration.json').read_text())['best'];q=json.loads((ROOT/'data/map/queries.json').read_text())['queries']
    # Remove complete cells, rather than checking values already supplied to interpolation.
    valid=np.flatnonzero(np.isfinite(z));held=np.random.default_rng(1234).choice(valid,3000,replace=False)
    hidden=z.copy();truth=hidden.ravel()[held].copy();hidden.ravel()[held]=np.nan
    predicted,_=interpolate_surface(hidden,.05);predicted,_,_=fill_planar_layers(hidden,predicted,.05)
    p=predicted.ravel()[held];ok=np.isfinite(p);error=p[ok]-truth[ok]
    holdout=dict(heldout_cells=len(held),reconstructed_cells=int(ok.sum()),rmse_m=float(np.sqrt(np.mean(error**2))),median_abs_error_m=float(np.median(abs(error))),p95_abs_error_m=float(np.quantile(abs(error),.95)),limitation='Held-out measured cells; not ground truth for originally unobserved holes.')
    pdata=np.load(OUT/'point_geometry.npz');points,eig,nz,radii=[pdata[k] for k in ['points','eigenvalues','normal_z','radius']];cells=np.floor((points[:,:2]-scfg['origin'][:2])/.05).astype(int)
    base,_=interpolate_surface(z,.05);sensitivity=[]
    for distance in [.5,.65,.8]:
        surface,filled,_=fill_planar_layers(z,base,.05,max_distance=distance)
        raster,*_=rasterize_surface(surface,points,eig,nz,radii,cells,.05)
        registered=resample(np.flipud(raster),scfg,ref.shape,rcfg,best['theta'],best['translation'])
        sensitivity.append(dict(layer_distance_m=distance,layer_filled_cells=int(filled.sum()),queries=query_status(registered,rcfg,q)))
    perturbations=[]
    for offset in [[-.05,0],[.05,0],[0,-.05],[0,.05]]:
        registered=resample(source,scfg,ref.shape,rcfg,best['theta'],np.array(best['translation'])+offset)
        perturbations.append(dict(translation_offset_m=offset,queries=query_status(registered,rcfg,q)))
    alternative=[]
    for candidate in json.loads((OUT/'registration.json').read_text())['candidates']:
        registered=resample(source,scfg,ref.shape,rcfg,candidate['theta'],candidate['translation'])
        alternative.append(dict(angle_deg=candidate['angle_deg'],queries=query_status(registered,rcfg,q)))
    report=dict(surface_holdout=holdout,layer_sensitivity=sensitivity,registration_sensitivity=perturbations,orientation_candidates=alternative)
    (OUT/'audit.json').write_text(json.dumps(report,indent=2))
    fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    axes[0].scatter(truth[ok],p[ok],s=3,alpha=.4);axes[0].plot([-.1,1.2],[-.1,1.2],'k--');axes[0].set(xlabel='Held-out measured height (m)',ylabel='Reconstructed height (m)')
    axes[1].hist(error,bins=60);axes[1].set(xlabel='Signed height error (m)',ylabel='Cell count');fig.savefig(OUT/'holdout_validation.png',dpi=150);plt.close(fig)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
