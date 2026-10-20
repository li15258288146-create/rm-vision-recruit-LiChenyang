"""Read-only verification of submitted paths; writes a separate evidence report."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
import cv2,numpy as np,yaml
from planner import GridMap
ROOT=Path(__file__).resolve().parents[1];B=ROOT/'qB_sentry'

def main():
    queries=json.loads((ROOT/'data/map/queries.json').read_text())
    assert queries['robot_radius_m']==.30
    expected={'data/map/queries.json':'4804519b22049f084b1e7a5ed88d0ab5454cc13205ce6dc52f28ca3d408469eb','tools/check_path.py':'af95d5d5241d41ea9c76e3c36fac41e80bf022ea850e1980aee77da78f59bc6a'}
    for file,sha in expected.items():assert hashlib.sha256((ROOT/file).read_bytes()).hexdigest()==sha,file
    reference=ROOT/'data/map/rmuc_2025.yaml';own=B/'results/optimized/registered_map.yaml';fused=B/'results/robust/fused_map.yaml'
    cases=[]
    for i in range(1,6):
        cases.append((B/f'path{i}.csv',reference,i))
        for alg in ['astar','dijkstra']:
            cases.append((B/f'results/{alg}/path{i}.csv',reference,i))
            cases.append((B/f'results/optimized/{alg}/path{i}.csv',own,i))
            for config in [reference,own,fused]:cases.append((B/f'results/robust/{alg}/path{i}.csv',config,i))
    grids={}
    for config in [reference,own,fused]:
        cfg=yaml.safe_load(config.read_text());im=cv2.imread(str(config.parent/cfg['image']),0)
        grids[config]=GridMap(im,cfg['resolution'],cfg['origin'],cfg['free_thresh'],radius=.30,negate=cfg.get('negate',0))
    records=[]
    for file,config,i in cases:
        p=np.loadtxt(file,delimiter=',',skiprows=1);q=queries['queries'][i-1]
        assert np.allclose(p[0],q['start'],rtol=0,atol=1e-12) and np.allclose(p[-1],q['goal'],rtol=0,atol=1e-12),file
        assert np.linalg.norm(np.diff(p,axis=0),axis=1).max()<=.15+1e-10,file
        assert grids[config].verify(p),(file,config)
        proc=subprocess.run([sys.executable,str(ROOT/'tools/check_path.py'),config.as_posix(),str(file)],capture_output=True,encoding='utf-8',env=os.environ|{'PYTHONIOENCODING':'utf-8'})
        log=proc.stdout+proc.stderr
        assert proc.returncode==0 and log.strip().splitlines()[-1]=='通过',(file,config,log)
        records.append(dict(path=file.relative_to(ROOT).as_posix(),map=config.relative_to(ROOT).as_posix(),official_passed=True,dense_segments_passed=True))
    reasons={}
    for i in [6,7]:
        q=queries['queries'][i-1];g=grids[reference]
        reasons[str(i)]=g.reason(g.world_to_cell(q['goal']))
    assert reasons=={'6':'insufficient_robot_clearance','7':'outside_map'}
    report=dict(check_count=len(records),checks=records,unreachable=reasons,hashes=expected)
    (B/'results/submission_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(f'{len(records)} official checks and dense segment checks passed; original endpoints/radius preserved; Q6/Q7 rejected correctly.')
if __name__=='__main__':main()
