"""Validate unchanged original queries and cross-check generated paths on reference map."""
import os,sys,json,hashlib,subprocess
from pathlib import Path
import numpy as np,cv2,yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from planner import GridMap
import compare_maps

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'qB_sentry/results/optimized'


def main():
    compare_maps.OUT=OUT
    compare_maps.main()
    query_path=ROOT/'data/map/queries.json';queries=json.loads(query_path.read_text())['queries']
    summary=json.loads((OUT/'comparison.json').read_text());checks=[]
    for row in summary:
        if row['status']!='reachable':continue
        i=row['query'];file=OUT/row['algorithm']/f'path{i}.csv';points=np.loadtxt(file,delimiter=',',skiprows=1)
        assert np.allclose(points[0],queries[i-1]['start'],atol=1e-12,rtol=0)
        assert np.allclose(points[-1],queries[i-1]['goal'],atol=1e-12,rtol=0)
        check=subprocess.run([sys.executable,str(ROOT/'tools/check_path.py'),(ROOT/'data/map/rmuc_2025.yaml').as_posix(),str(file)],capture_output=True,encoding='utf-8',env=os.environ|{'PYTHONIOENCODING':'utf-8'})
        log=check.stdout+check.stderr;(file.parent/f'cross_reference_check{i}.txt').write_text(log,encoding='utf-8')
        passed=check.returncode==0 and log.strip().endswith('通过') and not log.strip().endswith('没通过')
        checks.append(dict(query=i,algorithm=row['algorithm'],reference_checker_passed=passed,original_endpoints_preserved=True))
    report=dict(cross_checks=checks,queries_sha256=hashlib.sha256(query_path.read_bytes()).hexdigest(),checker_sha256=hashlib.sha256((ROOT/'tools/check_path.py').read_bytes()).hexdigest(),robot_radius_m=.30)
    (OUT/'validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    cfg=yaml.safe_load((ROOT/'data/map/rmuc_2025.yaml').read_text());ref=cv2.imread(str(ROOT/'data/map'/cfg['image']),0)
    refgrid=GridMap(ref,cfg['resolution'],cfg['origin'],cfg['free_thresh'])
    fig,axes=plt.subplots(2,1,figsize=(14,12),constrained_layout=True)
    for ax,im,title in zip(axes,[cv2.imread(str(OUT/'registered_map.pgm'),0),ref],['Paths on reconstructed map','Same paths on reference: red dots violate reference clearance']):
        ax.imshow(im,cmap='gray',vmin=0,vmax=255);ax.set_title(title)
        for row in summary:
            if row['algorithm']!='astar' or row['status']!='reachable':continue
            points=np.loadtxt(OUT/'astar'/f"path{row['query']}.csv",delimiter=',',skiprows=1)
            cells=np.array([refgrid.world_to_cell(p) for p in points]);ax.plot(cells[:,1],cells[:,0],label=f"Q{row['query']}",lw=1)
            if ax is axes[1]:
                bad=np.array([refgrid.reason(tuple(cell)) is not None for cell in cells]);ax.scatter(cells[bad,1],cells[bad,0],s=5,c='red')
        ax.legend()
    fig.savefig(OUT/'paths_cross_view.png',dpi=140);plt.close(fig)

if __name__=='__main__':main()
