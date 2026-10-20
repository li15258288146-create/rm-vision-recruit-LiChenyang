"""Reproducible 2-D sentry FSM with grid A*, resources, line of sight and event log."""
import csv,json,math
from pathlib import Path
import cv2
import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from planner import GridMap,search

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'qB_sentry/results/decision'
STATES=['PATROL','ENGAGE','RETREAT','RESUPPLY','DEFEND','STOP']


def decide(previous,hp,ammo,enemy_distance,remaining,outpost_alive):
    if hp<=0 or remaining<=0:return 'STOP'
    if hp<60 or (previous=='RETREAT' and hp<160):return 'RETREAT'
    if ammo<15 or (previous=='RESUPPLY' and ammo<80):return 'RESUPPLY'
    if not outpost_alive or remaining<=30:return 'DEFEND'
    if enemy_distance<=6:return 'ENGAGE'
    return 'PATROL'


def visible(grid,a,b):
    steps=max(1,int(math.dist(a,b)/(.5*grid.resolution))+1)
    for t in np.linspace(0,1,steps+1):
        r,c=grid.world_to_cell(np.asarray(a)*(1-t)+np.asarray(b)*t)
        if not (0<=r<grid.h and 0<=c<grid.w and grid.raw_free[r,c]):return False
    return True


def route(grid,start,goal):
    s,g=grid.world_to_cell(start),grid.world_to_cell(goal)
    if grid.reason(s) or grid.reason(g):return None
    result=search(grid.free,s,g)
    if result['path'] is None:return None
    points=np.array([start]+[grid.cell_to_world(c) for c in result['path']]+[goal])
    if not grid.verify(points):return None
    return points


def advance(position,points,index,budget):
    p=position.copy()
    while index<len(points) and budget>1e-10:
        delta=points[index]-p;distance=np.linalg.norm(delta)
        if distance<=budget:p=points[index].copy();budget-=distance;index+=1
        else:p+=delta/distance*budget;budget=0
    return p,index


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cfg=yaml.safe_load((ROOT/'data/map/rmuc_2025.yaml').read_text());im=cv2.imread(str(ROOT/'data/map'/cfg['image']),0)
    grid=GridMap(im,cfg['resolution'],cfg['origin'],cfg['free_thresh'])
    supply=np.array([-2.26,-1.36]);defense=np.array([.1,-4.21]);patrol=[np.array([11.,-5.82]),supply]
    pos=supply.copy();hp=200.;ammo=100.;outpost=True;enemy=None;enemyhp=0.
    previous='STOP';path=np.empty((0,2));index=0;lastgoal=None;patrol_index=0
    dt=.1;duration=240;speed=1.2;logs=[];events=[];transitions=[];failed_routes=0;distance=0.
    video=cv2.VideoWriter(str(OUT/'simulation.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),20,(1166,760))
    if not video.isOpened():raise RuntimeError('Video writer unavailable')
    def event(t,name,detail):events.append(dict(t=t,event=name,detail=detail))
    for step in range(int(duration/dt)+1):
        t=round(step*dt,6);remaining=max(0,duration-t)
        # Scripted disturbances are explicit scenario inputs, never hidden physics.
        if step==150:
            for angle in np.linspace(0,2*np.pi,36,endpoint=False):
                candidate=pos+2.5*np.array([np.cos(angle),np.sin(angle)])
                if not grid.reason(grid.world_to_cell(candidate)) and visible(grid,pos,candidate):enemy=candidate;enemyhp=100.;break
            event(t,'enemy_spawn',None if enemy is None else enemy.tolist())
        if step==450:hp=30.;event(t,'external_damage_injection','HP set to 30 for retreat validation')
        if step==1100:ammo=0.;event(t,'ammo_depletion_injection','Ammo set to 0 for resupply validation')
        if step==1650:outpost=False;event(t,'outpost_destroyed','Defense trigger')
        enemy_distance=math.inf if enemy is None or not visible(grid,pos,enemy) else math.dist(pos,enemy)
        state=decide(previous,hp,ammo,enemy_distance,remaining,outpost)
        if state!=previous:transitions.append(dict(t=t,source=previous,target=state,hp=hp,ammo=ammo,remaining=remaining,outpost_alive=outpost))
        if state=='PATROL':
            if math.dist(pos,patrol[patrol_index])<.08:patrol_index=1-patrol_index
            goal=patrol[patrol_index]
        elif state in ['RETREAT','RESUPPLY']:goal=supply
        elif state=='DEFEND':goal=defense
        else:goal=pos
        # Engage within 6 m from current safe position. No unchecked straight-line pursuit.
        moving=state in ['PATROL','RETREAT','RESUPPLY','DEFEND']
        if moving and (lastgoal is None or np.linalg.norm(goal-lastgoal)>.01 or state!=previous):
            planned=route(grid,pos,goal);lastgoal=goal.copy();index=0
            if planned is None:failed_routes+=1;path=np.empty((0,2));event(t,'route_unavailable',goal.tolist())
            else:path=planned
        old=pos.copy()
        if moving:pos,index=advance(pos,path,index,speed*dt)
        assert grid.verify([old,pos]) and grid.reason(grid.world_to_cell(pos)) is None
        distance+=math.dist(old,pos)
        if math.dist(pos,supply)<.2 and state in ['RETREAT','RESUPPLY']:
            hp=min(200,hp+25*dt);ammo=min(100,ammo+25*dt)
        if enemy is not None and state in ['ENGAGE','DEFEND'] and math.dist(pos,enemy)<=6 and visible(grid,pos,enemy) and ammo>=2*dt:
            ammo-=2*dt;enemyhp-=30*dt;hp=max(0,hp-1*dt)
            if enemyhp<=0:event(t,'enemy_defeated',enemy.tolist());enemy=None
        logs.append(dict(t=t,x=pos[0],y=pos[1],state=state,hp=hp,ammo=ammo,remaining=remaining,outpost_alive=int(outpost),enemy_distance=None if not np.isfinite(enemy_distance) else enemy_distance))
        if step%5==0:
            canvas=cv2.cvtColor(cv2.resize(im,(1166,648),interpolation=cv2.INTER_NEAREST),cv2.COLOR_GRAY2BGR)
            def pixel(xy):r,c=grid.world_to_cell(xy);return(c*2,r*2)
            if len(path):cv2.polylines(canvas,[np.array([pixel(p) for p in path],np.int32)],False,(190,130,70),2)
            cv2.circle(canvas,pixel(supply),9,(0,170,0),-1);cv2.putText(canvas,'SUPPLY',pixel(supply+np.array([0,.5])),0,.6,(0,130,0),2)
            cv2.circle(canvas,pixel(defense),9,(200,0,200),2)
            cv2.circle(canvas,pixel(pos),12,(0,120,255),-1)
            if enemy is not None:cv2.circle(canvas,pixel(enemy),10,(0,0,255),-1)
            panel=np.full((112,1166,3),245,np.uint8)
            cv2.putText(panel,f't={t:.1f}s   {state}   HP={hp:.0f}   Ammo={ammo:.0f}',(20,35),0,.85,(20,20,20),2)
            cv2.putText(panel,f'Remaining={remaining:.0f}s   Outpost={"alive" if outpost else "destroyed"}   A* navigation',(20,77),0,.7,(20,20,20),2)
            video.write(np.vstack([canvas,panel]))
        previous=state
    video.release()
    with (OUT/'trajectory.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(logs[0]));w.writeheader();w.writerows(logs)
    (OUT/'events.json').write_text(json.dumps(events,indent=2));(OUT/'transitions.json').write_text(json.dumps(transitions,indent=2))
    report=dict(duration_s=duration,dt_s=dt,speed_m_s=speed,distance_m=distance,collision_checks=len(logs),collision_failures=0,failed_routes=failed_routes,visited_states=sorted(set(x['state'] for x in logs)),transition_count=len(transitions),final_hp=hp,final_ammo=ammo,simulation_video_speedup=10,scenario='15s enemy;45s HP=30;110s ammo=0;165s outpost lost;last30s defend',map='provided map; registration uncertainty does not affect FSM experiment')
    assert set(report['visited_states'])==set(STATES)
    assert any(e['event']=='enemy_defeated' for e in events)
    assert any(e['source']=='RETREAT' and e['target']!='RETREAT' for e in transitions)
    assert any(e['source']=='RESUPPLY' and e['target']!='RESUPPLY' for e in transitions)
    (OUT/'summary.json').write_text(json.dumps(report,indent=2))
    fig,axes=plt.subplots(3,1,figsize=(12,10),constrained_layout=True)
    axes[0].imshow(im,cmap='gray');cells=np.array([grid.world_to_cell([a['x'],a['y']]) for a in logs]);axes[0].scatter(cells[:,1],cells[:,0],c=[STATES.index(a['state']) for a in logs],s=2,cmap='tab10');axes[0].set_title('Executed trajectory (A* and radius 0.30 m)')
    ts=[x['t'] for x in logs];axes[1].plot(ts,[x['hp'] for x in logs],label='HP');axes[1].plot(ts,[x['ammo'] for x in logs],label='Ammo');axes[1].legend()
    axes[2].step(ts,[STATES.index(x['state']) for x in logs]);axes[2].set_yticks(range(len(STATES)),STATES);axes[2].set_xlabel('Simulation time (s)');fig.savefig(OUT/'simulation_summary.png',dpi=160);plt.close(fig)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
