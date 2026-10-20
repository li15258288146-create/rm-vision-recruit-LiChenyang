"""运行给定地图全部查询，导出路径、检查日志、搜索图和对比表。"""
import csv
import json
import math
from pathlib import Path
import subprocess
import sys
import cv2
import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from planner import GridMap, search

ROOT = Path(__file__).resolve().parents[1]


def main():
    config_path = ROOT/'data/map/rmuc_2025.yaml'
    config = yaml.safe_load(config_path.read_text())
    image = cv2.imread(str(config_path.parent/config['image']),0)
    queries = json.loads((ROOT/'data/map/queries.json').read_text())
    grid = GridMap(image,config['resolution'],config['origin'],config['free_thresh'],queries['robot_radius_m'],config['negate'])
    out=ROOT/'qB_sentry/results';out.mkdir(parents=True,exist_ok=True)
    cv2.imwrite(str(out/'inflated_free.png'),grid.free.astype(np.uint8)*255)
    rows=[]
    fig,axes=plt.subplots(4,2,figsize=(14,16),constrained_layout=True)
    for i,q in enumerate(queries['queries'],1):
        start,goal=grid.world_to_cell(q['start']),grid.world_to_cell(q['goal'])
        why = [('start',grid.reason(start)),('goal',grid.reason(goal))]
        reason='; '.join(f'{label}: {value}' for label,value in why if value)
        ax=axes.flat[i-1]; ax.imshow(image,cmap='gray',vmin=0,vmax=255)
        ax.scatter([start[1],goal[1]],[start[0],goal[0]],c=['green','red'],s=20)
        results={}
        if reason:
            rows.append(dict(query=i,algorithm='both',status=reason,length_m='',expanded=0,seconds=0,checker='not_applicable'))
            ax.set_title(f'Query {i}: {reason}',fontsize=9);continue
        for algorithm,heuristic,weight in [('astar','octile',1),('dijkstra','octile',1),('astar','manhattan',1),('astar','octile',2)]:
            label=algorithm if heuristic=='octile' and weight==1 else f'{algorithm}_{heuristic}_w{weight}'
            result=search(grid.free,start,goal,algorithm,heuristic,weight)
            results[label]=result
            entry=dict(query=i,algorithm=label,status='reachable' if result['path'] else 'disconnected',length_m='',expanded=len(result['expanded']),seconds=result['seconds'],checker='not_applicable')
            if result['path']:
                path=[tuple(q['start'])]+[grid.cell_to_world(c) for c in result['path']]+[tuple(q['goal'])]
                assert grid.verify(path),f'Continuous validation failed: {i}, {label}'
                assert max(math.dist(a,b) for a,b in zip(path,path[1:]))<=.15
                directory=out/label;directory.mkdir(exist_ok=True)
                pathfile=directory/f'path{i}.csv'
                with pathfile.open('w',newline='') as f:
                    writer=csv.writer(f);writer.writerow(['x','y']);writer.writerows(path)
                length=sum(math.dist(a,b) for a,b in zip(path,path[1:]));entry['length_m']=length
                if label in ('astar','dijkstra'):
                    process=subprocess.run([sys.executable,str(ROOT/'tools/check_path.py'),config_path.as_posix(),str(pathfile)],capture_output=True,encoding='utf-8',env=__import__('os').environ|{'PYTHONIOENCODING':'utf-8'})
                    log=process.stdout+process.stderr
                    (directory/f'check_path{i}.txt').write_text(log,encoding='utf-8')
                    assert process.returncode==0 and log.strip().endswith('通过') and not log.strip().endswith('没通过'),log
                    entry['checker']='passed'
                if label=='astar':
                    xy=np.array(result['path']);ax.plot(xy[:,1],xy[:,0],color='tomato',lw=1.2)
                    with (ROOT/'qB_sentry'/f'path{i}.csv').open('w',newline='') as f:
                        writer=csv.writer(f);writer.writerow(['x','y']);writer.writerows(path)
            rows.append(entry)
        a,d=results['astar'],results['dijkstra']
        assert (a['path'] is None)==(d['path'] is None)
        if a['path']:assert abs(a['cost_cells']-d['cost_cells'])<1e-7
        ax.set_title(f'Query {i}: '+('reachable' if a['path'] else 'disconnected'))
        # Show expansion order, rather than only final path.
        fig2,ax2=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
        for panel,label in zip(ax2,['astar','dijkstra']):
            panel.imshow(image,cmap='gray');cells=np.array(results[label]['expanded'])
            if len(cells):panel.scatter(cells[:,1],cells[:,0],c=np.arange(len(cells)),s=1,cmap='viridis')
            panel.set_title(f'{label}: {len(cells)} expanded')
        fig2.savefig(out/f'search_query{i}.png',dpi=160);plt.close(fig2)
    axes.flat[-1].axis('off')
    fig.savefig(out/'all_queries.png',dpi=160);plt.close(fig)
    with (out/'summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (out/'summary.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
