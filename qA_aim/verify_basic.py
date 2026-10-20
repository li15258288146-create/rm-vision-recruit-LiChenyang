"""Audit basic-task videos, CSV schema, PnP reproducibility and prediction timing."""
import csv,hashlib,json
from pathlib import Path
import cv2,numpy as np
from aim import pose
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'qA_aim'
HEADER='frame,lt_x,lt_y,lb_x,lb_y,rb_x,rb_y,rt_x,rt_y,x,y,z,yaw'.split(',')

def main():
    cfg=json.loads((ROOT/'data/armor/camera_info.json').read_text(encoding='utf-8'))
    K=np.array(cfg['camera_matrix']);dist=np.array(cfg['distortion_coefficients']);reports=[]
    for name in ['远距离平移','中距离陀螺']:
        source=ROOT/'data/armor'/f'{name}.mp4';csvfile=OUT/f'detections_{name}.csv'
        with csvfile.open(encoding='utf-8') as f:
            reader=csv.reader(f);assert next(reader)==HEADER;rows=np.array(list(reader),float)
        assert np.isfinite(rows).all() and np.all(rows[:,0]==rows[:,0].astype(int))
        pts=rows[:,1:9].reshape(-1,4,2)
        assert np.all((pts[:,:,0]>=0)&(pts[:,:,0]<1440)) and np.all((pts[:,:,1]>=0)&(pts[:,:,1]<1080))
        assert np.all(pts[:,0,1]<pts[:,1,1]) and np.all(pts[:,3,1]<pts[:,2,1])
        assert np.all(pts[:,:2,0].mean(1)<pts[:,2:,0].mean(1)) and np.all(rows[:,11]>0)
        for row,p in zip(rows[::100],pts[::100]):
            sol=pose(p,K,dist);np.testing.assert_allclose(sol['position'],row[9:12],atol=1e-9);assert abs(sol['yaw']-row[12])<1e-9
        counts=[];fps_values=[]
        crops=[];chosen=set(np.linspace(0,int(rows[:,0].max()),12,dtype=int))
        for video in [source,OUT/'figures'/f'{name}_detections.mp4']:
            cap=cv2.VideoCapture(str(video));assert cap.isOpened(),video
            fps_values.append(cap.get(cv2.CAP_PROP_FPS));n=0
            while True:
                ok,im=cap.read()
                if not ok:break
                if video==source and n in chosen:
                    framepts=pts[rows[:,0]==n]
                    if len(framepts):
                        allpts=framepts.reshape(-1,2);lo=np.maximum(allpts.min(0).astype(int)-45,0);hi=np.minimum(allpts.max(0).astype(int)+45,[1439,1079])
                        for p in framepts:
                            cv2.polylines(im,[np.round(p).astype(np.int32)],True,(0,255,0),2)
                            for j,point in enumerate(p):cv2.putText(im,str(j+1),tuple(point.astype(int)),0,.5,(255,255,0),1)
                        crop=cv2.resize(im[lo[1]:hi[1]+1,lo[0]:hi[0]+1],(360,200));cv2.putText(crop,f'frame {n}',(8,20),0,.5,(255,255,255),1);crops.append(crop)
                n+=1
            cap.release();counts.append(n)
        assert counts[0]==counts[1] and np.all(rows[:,0]<counts[0]);assert abs(fps_values[1]-150)<.1
        if len(crops)==12:cv2.imencode('.png',np.vstack([np.hstack(crops[i:i+3]) for i in range(0,12,3)]))[1].tofile(str(OUT/'figures'/f'{name}_audit.png'))
        reports.append(dict(video=name,decoded_source_frames=counts[0],decoded_output_frames=counts[1],container_fps=fps_values[0],timestamp_fps=150,detection_rows=len(rows),detected_frames=len(np.unique(rows[:,0])),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
    state=np.load(OUT/'translation_states.npz');t=state['t'];obs=state['observation'];pred=state['prediction']
    exported=np.loadtxt(OUT/'pred_远距离平移.csv',delimiter=',',skiprows=1);valid=np.isfinite(pred).all(1)
    np.testing.assert_allclose(exported,np.c_[t[valid],pred[valid]],atol=1e-12)
    np.testing.assert_allclose(t,np.arange(len(t))/150,atol=1e-12)
    ids=np.arange(len(t)-38);future=(obs[ids+37]+obs[ids+38])/2
    ok=np.isfinite(future).all(1)&np.isfinite(pred[ids]).all(1);ids=ids[ok];future=future[ok];err=pred[ids]-future
    table=np.loadtxt(OUT/'prediction_evaluation.csv',delimiter=',',skiprows=1)
    np.testing.assert_allclose(table,np.c_[t[ids],t[ids]+.25,pred[ids],future,err,np.linalg.norm(err,axis=1)],atol=1e-12)
    metrics=json.loads((OUT/'metrics.json').read_text(encoding='utf-8'));saved=next(x for x in metrics if x['video']=='远距离平移')['prediction']
    assert saved['evaluated_frames']==len(ids);np.testing.assert_allclose(saved['rmse_xyz_m'],np.sqrt(np.mean(err**2,axis=0)),atol=1e-12)
    for name in ['translation_timeseries.png','translation_xz.png','prediction_error.png','threshold_tool_window.jpg']:
        assert cv2.imdecode(np.fromfile(OUT/'figures'/name,np.uint8),cv2.IMREAD_COLOR) is not None,name
    report=dict(videos=reports,prediction=saved,prediction_rows=len(exported),checks='Full decode, CSV columns/order/bounds, sampled PnP, forecast timestamp and metrics recomputation passed.',limitations='Frame detection coverage is not precision/recall; no independent position ground truth. Existing slider-window screenshot inspected; GUI interaction not replayed by this verifier.')
    (OUT/'basic_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
