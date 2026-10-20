"""Dual-map boundary constraints and sampled SE(2) registration risk; no endpoint tuning."""
import hashlib, itertools, json, os, subprocess, sys
from pathlib import Path
import cv2, numpy as np, yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from planner import GridMap, search
from refine_registration import resample
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'qB_sentry/results/optimized'
OUT=ROOT/'qB_sentry/results/robust'


def clearance(raw, resolution):
    return cv2.distanceTransform(np.pad(raw.astype(np.uint8),1),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)[1:-1,1:-1]*resolution


def sample_path(points, step=.01):
    parts=[]
    for a,b in zip(points,points[1:]):
        n=max(1,int(np.ceil(np.linalg.norm(b-a)/step)))
        parts.append(a+(b-a)*((np.arange(n)+.5)/n)[:,None])
    return np.vstack(parts)


def metrics(points, grid, scenarios):
    mid=sample_path(points)
    cells=np.array([grid.world_to_cell(p) for p in mid])
    valid=scenarios[:,cells[:,0],cells[:,1]]
    ends=np.array([grid.world_to_cell(p) for p in [points[0],points[-1]]])
    return dict(all_scenarios_safe_sample_fraction=float(valid.all(axis=0).mean()),
                mean_scenario_safe_sample_fraction=float(valid.mean()),
                whole_path_passing_scenarios=int((valid.all(axis=1)&scenarios[:,ends[:,0],ends[:,1]].all(axis=1)).sum()),
                endpoint_passing_scenarios=scenarios[:,ends[:,0],ends[:,1]].sum(axis=0).tolist())


def main():
    assert hashlib.sha256((ROOT/'data/map/queries.json').read_bytes()).hexdigest() == '4804519b22049f084b1e7a5ed88d0ab5454cc13205ce6dc52f28ca3d408469eb'
    assert hashlib.sha256((ROOT/'tools/check_path.py').read_bytes()).hexdigest() == 'af95d5d5241d41ea9c76e3c36fac41e80bf022ea850e1980aee77da78f59bc6a'
    OUT.mkdir(parents=True,exist_ok=True)
    cfg=yaml.safe_load((ROOT/'data/map/rmuc_2025.yaml').read_text())
    ref=cv2.imread(str(ROOT/'data/map'/cfg['image']),0)
    own=cv2.imread(str(BASE/'registered_map.pgm'),0)
    def grid(im):return GridMap(im,cfg['resolution'],cfg['origin'],cfg['free_thresh'],radius=.30)
    rg,og=grid(ref),grid(own)
    # An obstacle/unknown in either map remains forbidden. This is fusion, not independent reconstruction.
    fused=np.minimum(ref,own);fg=grid(fused)
    cv2.imwrite(str(OUT/'fused_map.pgm'),fused)
    (OUT/'fused_map.yaml').write_text(yaml.safe_dump(cfg|{'image':'fused_map.pgm'}))
    fit=json.loads((BASE/'registration.json').read_text())['best']
    src=cv2.imread(str(BASE/'terrain_map.pgm'),0)
    scfg=yaml.safe_load((BASE/'terrain_map.yaml').read_text())
    hypotheses=[];masks=[]
    # Translation and rotation are coupled in SE(2), including lever-arm displacement.
    for dx,dy,deg in itertools.product([-.05,0,.05],[-.05,0,.05],[-.1,0,.1]):
        image=resample(src,scfg,ref.shape,cfg,fit['theta']+np.deg2rad(deg),np.array(fit['translation'])+[dx,dy])
        masks.append(grid(image).free & rg.free)
        hypotheses.append(dict(dx_m=dx,dy_m=dy,dtheta_deg=deg))
    masks=np.stack(masks)
    risk=1-masks.mean(axis=0)
    margin=clearance(fg.raw_free,cfg['resolution'])-.30
    cost=1+40*risk+2*np.exp(-np.maximum(margin,0)/.12)
    np.savez_compressed(OUT/'uncertainty.npz',risk=risk,robust_core=masks.all(axis=0),cost=cost)
    queries=json.loads((ROOT/'data/map/queries.json').read_text())['queries']
    records=[]
    for i,q in enumerate(queries,1):
        s,t=[fg.world_to_cell(q[k]) for k in ['start','goal']]
        reasons=[fg.reason(s),fg.reason(t)]
        if any(reasons):
            records.append(dict(query=i,status='invalid_endpoint',reasons=reasons));continue
        weighted=[]
        for algorithm in ['astar','dijkstra']:
            result=search(fg.free,s,t,algorithm=algorithm,traversal_cost=cost)
            if result['path'] is None:raise RuntimeError(f'Q{i} disconnected')
            points=np.array([q['start']]+[fg.cell_to_world(c) for c in result['path']]+[q['goal']])
            assert all(g.verify(points) for g in [fg,og,rg])
            assert np.array_equal(points[0],q['start']) and np.array_equal(points[-1],q['goal'])
            assert np.linalg.norm(np.diff(points,axis=0),axis=1).max()<=.15
            folder=OUT/algorithm;folder.mkdir(exist_ok=True)
            file=folder/f'path{i}.csv';np.savetxt(file,points,delimiter=',',header='x,y',comments='',fmt='%.10f')
            checks={}
            for label,path in [('self',BASE/'registered_map.yaml'),('reference',ROOT/'data/map/rmuc_2025.yaml'),('fused',OUT/'fused_map.yaml')]:
                check=subprocess.run([sys.executable,str(ROOT/'tools/check_path.py'),path.as_posix(),str(file)],capture_output=True,encoding='utf-8',env=os.environ|{'PYTHONIOENCODING':'utf-8'})
                log=check.stdout+check.stderr;(folder/f'{label}_check{i}.txt').write_text(log,encoding='utf-8')
                checks[label]=check.returncode==0 and log.strip().splitlines()[-1]=='通过'
            assert all(checks.values()), checks
            old=np.loadtxt(BASE/algorithm/f'path{i}.csv',delimiter=',',skiprows=1)
            records.append(dict(query=i,algorithm=algorithm,status='reachable',length_m=float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum()),weighted_cost=result['cost_cells']*.05,checks=checks,robustness=metrics(points,fg,masks),previous_robustness=metrics(old,fg,masks)))
            weighted.append(result['cost_cells'])
        assert np.isclose(*weighted,rtol=1e-10)
        print('Q',i,records[-2]['length_m'],records[-2]['robustness'],flush=True)
    report=dict(robot_radius_m=.30,max_segment_m=.15,transform=fit,hypotheses=hypotheses,
                limitation='27 discrete perturbations are sensitivity tests, not a continuous uncertainty guarantee. Fixed endpoints may be outside robust core. Reference map constraints are explicitly fused; independent map remains unchanged.',
                queries_sha256=hashlib.sha256((ROOT/'data/map/queries.json').read_bytes()).hexdigest(),
                checker_sha256=hashlib.sha256((ROOT/'tools/check_path.py').read_bytes()).hexdigest(),records=records)
    (OUT/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    fig,axes=plt.subplots(2,1,figsize=(14,12),constrained_layout=True)
    for ax,im,title in zip(axes,[ref,risk],['Dual-map paths on reference map','Registration sensitivity (27 hypotheses); lower is better']):
        ax.imshow(im,cmap='gray' if ax is axes[0] else 'magma',vmin=0,vmax=255 if ax is axes[0] else 1)
        for row in records:
            if row.get('algorithm')!='astar':continue
            pts=np.loadtxt(OUT/'astar'/f"path{row['query']}.csv",delimiter=',',skiprows=1)
            cells=np.array([fg.world_to_cell(p) for p in pts]);ax.plot(cells[:,1],cells[:,0],lw=1,label=f"Q{row['query']}")
        ax.set_title(title);ax.legend()
    fig.savefig(OUT/'paths_and_risk.png',dpi=150);plt.close(fig)

if __name__=='__main__':main()
