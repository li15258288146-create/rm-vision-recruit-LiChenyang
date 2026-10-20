"""Run identical queries on the independent, registered map; do not move invalid endpoints."""
import csv,json,math,os,subprocess,sys
from pathlib import Path
import cv2,numpy as np,yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from planner import GridMap,search

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'qB_sentry/results/advanced'


def main():
    cfg_path=OUT/'registered_map.yaml';cfg=yaml.safe_load(cfg_path.read_text())
    im=cv2.imread(str(OUT/cfg['image']),0);grid=GridMap(im,cfg['resolution'],cfg['origin'],cfg['free_thresh'])
    queries=json.loads((ROOT/'data/map/queries.json').read_text())['queries']
    reference=json.loads((ROOT/'qB_sentry/results/summary.json').read_text())
    baseline={r['query']:r for r in reference if r['algorithm'] in ['astar','both']}
    reports=[];fig,axes=plt.subplots(4,2,figsize=(15,13),constrained_layout=True)
    for i,q in enumerate(queries,1):
        start,goal=grid.world_to_cell(q['start']),grid.world_to_cell(q['goal'])
        reason='; '.join(f'{label}: {value}' for label,value in [('start',grid.reason(start)),('goal',grid.reason(goal))] if value)
        ax=axes.flat[i-1];ax.imshow(im,cmap='gray',vmin=0,vmax=255);ax.scatter([start[1],goal[1]],[start[0],goal[0]],c=['green','red'],s=18)
        paths={}
        for algorithm in ['astar','dijkstra']:
            result=search(grid.free,start,goal,algorithm) if not reason else None
            status=reason or ('reachable' if result['path'] else 'disconnected')
            row=dict(query=i,algorithm=algorithm,status=status,length_m=None,expanded=0 if result is None else len(result['expanded']),seconds=0 if result is None else result['seconds'],reference_status=baseline[i]['status'],reference_length_m=baseline[i]['length_m'],checker='not_applicable')
            if result and result['path']:
                points=[q['start']]+[grid.cell_to_world(c) for c in result['path']]+[q['goal']]
                assert grid.verify(points)
                assert max(math.dist(a,b) for a,b in zip(points,points[1:]))<=.15
                directory=OUT/algorithm;directory.mkdir(exist_ok=True);p=directory/f'path{i}.csv'
                np.savetxt(p,points,delimiter=',',header='x,y',comments='')
                check=subprocess.run([sys.executable,str(ROOT/'tools/check_path.py'),cfg_path.as_posix(),str(p)],capture_output=True,encoding='utf-8',env=os.environ|{'PYTHONIOENCODING':'utf-8'})
                log=check.stdout+check.stderr;(directory/f'check_path{i}.txt').write_text(log,encoding='utf-8')
                assert check.returncode==0 and log.strip().endswith('通过') and not log.strip().endswith('没通过'),log
                row['length_m']=sum(math.dist(a,b) for a,b in zip(points,points[1:]));row['checker']='passed';paths[algorithm]=result['cost_cells']
                if algorithm=='astar':a=np.array(result['path']);ax.plot(a[:,1],a[:,0],c='tomato',lw=1)
            reports.append(row)
        if paths:assert abs(paths['astar']-paths['dijkstra'])<1e-7
        ax.set_title(f'Query {i}: {reports[-1]["status"]}',fontsize=9)
    axes.flat[-1].axis('off');fig.savefig(OUT/'self_map_queries.png',dpi=140);plt.close(fig)
    (OUT/'comparison.json').write_text(json.dumps(reports,indent=2))
    with (OUT/'comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=list(reports[0]));w.writeheader();w.writerows(reports)
    print(json.dumps(reports,indent=2))
    # Supplemental cases, explicitly separate from the seven original queries.
    refcfg=yaml.safe_load((ROOT/'data/map/rmuc_2025.yaml').read_text())
    refim=cv2.imread(str(ROOT/'data/map'/refcfg['image']),0)
    refgrid=GridMap(refim,refcfg['resolution'],refcfg['origin'],refcfg['free_thresh'])
    n,labels,stats,_=cv2.connectedComponentsWithStats(grid.free.astype(np.uint8),4)
    largest=1+np.argmax(stats[1:,cv2.CC_STAT_AREA])
    eligible=np.argwhere((labels==largest)&refgrid.free)
    supplemental=[];records=[];fig,axes=plt.subplots(3,2,figsize=(14,11),constrained_layout=True)
    for i,direction in enumerate([np.array([1,0]),np.array([0,1]),np.array([1,1])],1):
        order=eligible@direction;s=tuple(eligible[order.argmin()]);g=tuple(eligible[order.argmax()])
        start,goal=grid.cell_to_world(s),grid.cell_to_world(g)
        supplemental.append(dict(id=f'S{i}',start=start,goal=goal,selection='extreme common-free cells in largest self-map component; separate benchmark'))
        for col,(label,gm,configpath) in enumerate([('self',grid,cfg_path),('reference',refgrid,ROOT/'data/map/rmuc_2025.yaml')]):
            costs=[];ax=axes[i-1,col];ax.imshow(gm.image,cmap='gray',vmin=0,vmax=255)
            for alg in ['astar','dijkstra']:
                result=search(gm.free,gm.world_to_cell(start),gm.world_to_cell(goal),alg)
                assert result['path'] is not None
                points=[start]+[gm.cell_to_world(c) for c in result['path']]+[goal]
                assert gm.verify(points)
                directory=OUT/'supplemental'/label/alg;directory.mkdir(parents=True,exist_ok=True)
                file=directory/f'pathS{i}.csv';np.savetxt(file,points,delimiter=',',header='x,y',comments='')
                check=subprocess.run([sys.executable,str(ROOT/'tools/check_path.py'),configpath.as_posix(),str(file)],capture_output=True,encoding='utf-8',env=os.environ|{'PYTHONIOENCODING':'utf-8'})
                log=check.stdout+check.stderr;(directory/f'checkS{i}.txt').write_text(log,encoding='utf-8')
                assert check.returncode==0 and log.strip().endswith('通过') and not log.strip().endswith('没通过')
                length=sum(math.dist(a,b) for a,b in zip(points,points[1:]));costs.append(result['cost_cells'])
                records.append(dict(query=f'S{i}',map=label,algorithm=alg,length_m=length,expanded=len(result['expanded']),seconds=result['seconds'],checker='passed'))
                if alg=='astar':a=np.array(result['path']);ax.plot(a[:,1],a[:,0],c='tomato');ax.set_title(f'S{i} {label}: {length:.2f} m')
            assert abs(costs[0]-costs[1])<1e-7
    fig.savefig(OUT/'supplemental_paths.png',dpi=140);plt.close(fig)
    (OUT/'supplemental_queries.json').write_text(json.dumps(supplemental,indent=2))
    (OUT/'supplemental_results.json').write_text(json.dumps(records,indent=2))
    print(json.dumps(records,indent=2))

if __name__=='__main__':main()
