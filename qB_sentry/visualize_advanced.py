"""Export FSM diagram, synthetic terrain checks and search expansion replay."""
import json
from pathlib import Path
import cv2,numpy as np,yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from planner import GridMap,search
from terrain_advanced import classify_surface

ROOT=Path(__file__).resolve().parents[1]


def main():
    advanced=ROOT/'qB_sentry/results/advanced';decision=ROOT/'qB_sentry/results/decision'
    # Priority selector shows complete transition guards, not only observed transitions.
    fig,ax=plt.subplots(figsize=(13,9));ax.set(xlim=(0,10),ylim=(0,10));ax.axis('off')
    conditions=['HP <= 0 or time <= 0','HP < 60; stay until HP >= 160','Ammo < 15; stay until ammo >= 80','Outpost lost OR time <= 30 s','Enemy visible AND distance <= 6 m','Otherwise']
    names=['STOP','RETREAT','RESUPPLY','DEFEND','ENGAGE','PATROL']
    for i,(condition,name) in enumerate(zip(conditions,names)):
        y=9-i*1.5
        ax.text(3.2,y,condition,ha='center',va='center',bbox=dict(boxstyle='round,pad=.6',fc='#e8eef8',ec='#5472a0'),fontsize=11)
        ax.text(8.5,y,name,ha='center',va='center',bbox=dict(boxstyle='round,pad=.6',fc='#e1f2e6',ec='#477753'),fontsize=12)
        ax.annotate('yes',xy=(7.7,y),xytext=(6,y),arrowprops=dict(arrowstyle='->'))
        if i<5:ax.annotate('no',xy=(3.2,y-1.15),xytext=(3.2,y-.45),ha='center',arrowprops=dict(arrowstyle='->'))
    ax.set_title('FSM transition guards: evaluate top to bottom from ANY state every 0.1 s\nRETREAT / RESUPPLY have exit hysteresis; any state can be interrupted by higher priority',fontsize=13)
    fig.savefig(decision/'state_transition.png',dpi=170,bbox_inches='tight');fig.savefig(decision/'state_transition.svg',bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
    y,x=np.indices((60,100));cases=[x*.05*.2,np.where(x<50,0.,.3),np.where(x<50,0.,.10)]
    titles=['11.31 degree ramp','30 cm step','10 cm step (below limit)']
    for col,(z,title) in enumerate(zip(cases,titles)):
        free,slope,edge=classify_surface(z,np.zeros_like(z,bool),.05)
        axes[0,col].plot(np.arange(100)*.05,z[30]);axes[0,col].set(title=title,xlabel='distance (m)',ylabel='height (m)')
        axes[1,col].imshow(np.where(edge,2,np.where(free,1,0)),vmin=0,vmax=2,cmap='viridis');axes[1,col].set_title('0 blocked / 1 free / 2 step edge')
    fig.savefig(advanced/'synthetic_terrain.png',dpi=160);plt.close(fig)
    cfg=yaml.safe_load((ROOT/'data/map/rmuc_2025.yaml').read_text());im=cv2.imread(str(ROOT/'data/map'/cfg['image']),0)
    grid=GridMap(im,cfg['resolution'],cfg['origin'],cfg['free_thresh']);q=json.loads((ROOT/'data/map/queries.json').read_text())['queries'][0]
    results=[search(grid.free,grid.world_to_cell(q['start']),grid.world_to_cell(q['goal']),alg) for alg in ['astar','dijkstra']]
    video=cv2.VideoWriter(str(advanced/'search_replay.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),15,(1166,380))
    if not video.isOpened():raise RuntimeError('Video writer unavailable')
    for frame in range(121):
        panels=[]
        budget=frame/100*max(len(r['expanded']) for r in results)
        for name,result in zip(['A*','Dijkstra'],results):
            panel=cv2.cvtColor(im,cv2.COLOR_GRAY2BGR);cells=np.array(result['expanded'][:min(int(budget),len(result['expanded']))])
            if len(cells):panel[cells[:,0],cells[:,1]]=[180,140,40]
            if budget>=len(result['expanded']):a=np.array(result['path']);cv2.polylines(panel,[a[:,::-1].astype(np.int32)],False,(0,0,255),2)
            panel=np.vstack([panel,np.full((56,583,3),245,np.uint8)]);cv2.putText(panel,f'{name}: {len(cells)} expanded',(12,355),0,.65,(20,20,20),2);panels.append(panel)
        video.write(np.hstack(panels))
    video.release()
    print('FSM, terrain validation and search replay exported')

if __name__=='__main__':main()
