"""逐帧处理四段素材、导出识别结果，并评估平移预测。"""
import argparse,csv,json,os
from pathlib import Path
import cv2,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from aim import detect,pose,filter_positions

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'qA_aim';FIG=OUT/'figures'


def save_image(path,image):cv2.imencode('.png',image)[1].tofile(str(path))


def evaluate(observations,prediction):
    # 0.25 s = 37.5 frames, interpolate only two neighboring valid observations.
    ids=np.arange(len(observations)-38);future=(observations[ids+37]+observations[ids+38])/2
    valid=np.isfinite(future).all(1)&np.isfinite(prediction[ids]).all(1)
    ids,future=ids[valid],future[valid];error=prediction[ids]-future
    if not len(ids):
        return ids,future,dict(evaluated_frames=0,rmse_xyz_m=None,mean_euclidean_error_m=None,median_euclidean_error_m=None)
    return ids,future,dict(evaluated_frames=len(ids),rmse_xyz_m=np.sqrt(np.mean(error**2,axis=0)).tolist(),mean_euclidean_error_m=float(np.linalg.norm(error,axis=1).mean()),median_euclidean_error_m=float(np.median(np.linalg.norm(error,axis=1))))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--videos',nargs='+',default=['远距离平移','中距离陀螺','近距离陀螺','装甲板_蓝'])
    parser.add_argument('--settings',type=Path,help='Threshold JSON saved by tune_threshold.py')
    parser.add_argument('--brightness',type=int,default=None)
    parser.add_argument('--difference',type=int,default=None)
    args=parser.parse_args()
    settings=json.loads(args.settings.read_text(encoding='utf-8-sig')) if args.settings else {}
    brightness=args.brightness if args.brightness is not None else settings.get('brightness',140)
    difference=args.difference if args.difference is not None else settings.get('difference',60)
    if not all(isinstance(v,int) and 0<=v<=255 for v in [brightness,difference]):parser.error('Thresholds must be integers in [0,255]')
    for name in args.videos:
        if not (ROOT/'data/armor'/f'{name}.mp4').is_file():parser.error(f'Video not found: {name}')
    FIG.mkdir(parents=True,exist_ok=True)
    config=json.loads((ROOT/'data/armor/camera_info.json').read_text(encoding='utf-8'));K=np.array(config['camera_matrix']);dist=np.array(config['distortion_coefficients'])
    reports=[]
    for name in args.videos:
        blue=name=='装甲板_蓝';cap=cv2.VideoCapture(str(ROOT/'data/armor'/f'{name}.mp4'));count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=24 if blue else 150
        if not cap.isOpened() or count<4:raise ValueError(f'Cannot decode video: {name}')
        size=(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if not blue and size!=(config['image_width'],config['image_height']):raise ValueError(f'Calibration/image size mismatch: {size}')
        # ASCII temporary path avoids Windows OpenCV non-ASCII output path issues.
        temp=FIG/f'preview_{len(reports)}.mp4';writer=cv2.VideoWriter(str(temp),cv2.VideoWriter_fourcc(*'mp4v'),fps,(720,540))
        if not writer.isOpened():raise RuntimeError('Video writer unavailable')
        obs=np.full((count,3),np.nan);previous=None;missing=0;rows=[];detected_frames=0;pnp_residuals=[];dist_differences=[];sample_images=[]
        for frame in range(count):
            ok,image=cap.read()
            if not ok:raise RuntimeError(f'Unexpected video decode end at {frame}/{count}')
            detections,mask=detect(image,'blue' if blue else 'red',brightness,difference);valid=[]
            for detection in detections:
                points=detection['points'];sol=None if blue else pose(points,K,dist)
                if not blue and (sol is None or sol['rms']>4 or not .2<sol['position'][2]<30):continue
                row=[frame]+points.ravel().tolist()+(['']*4 if blue else sol['position'].tolist()+[sol['yaw']]);rows.append(row)
                if sol:
                    valid.append((detection,sol));pnp_residuals.append(sol['rms'])
                    if frame%15==0:
                        no=pose(points,K,np.zeros_like(dist))
                        if no:dist_differences.append([frame,float(np.linalg.norm(points.mean(0)-K[:2,2])),float(np.linalg.norm(no['position']-sol['position']))])
                poly=np.round(points).astype(np.int32);cv2.polylines(image,[poly.reshape(-1,1,2)],True,(0,255,0),2)
                for j,p in enumerate(poly):cv2.putText(image,str(j+1),tuple(p),0,.45,(255,255,0),1)
                if sol:cv2.putText(image,f"z={sol['position'][2]:.2f}m",tuple(poly[0]+[0,-8]),0,.45,(0,255,0),1)
            if (valid if not blue else detections):detected_frames+=1
            if valid:
                if previous is None:chosen=min(valid,key=lambda x:x[0]['score'])
                else:chosen=min(valid,key=lambda x:np.linalg.norm(x[1]['position']-previous))
                position=chosen[1]['position']
                if previous is None or np.linalg.norm(position-previous)<.6:
                    obs[frame]=position;previous=position;missing=0
                else:missing+=1
            else:missing+=1
            if missing>30:previous=None
            cv2.putText(image,f'frame {frame} / t={frame/fps:.3f}s',(20,35),0,.8,(0,255,0),2)
            writer.write(cv2.resize(image,(720,540)))
            if frame in [0,count//4,count//2,3*count//4]:sample_images.append(cv2.resize(image,(720,540)))
        cap.release();writer.release()
        target=FIG/f'{name}_detections.mp4';os.replace(temp,target)
        with (OUT/f'detections_{name}.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.writer(f);w.writerow('frame,lt_x,lt_y,lb_x,lb_y,rb_x,rb_y,rt_x,rt_y,x,y,z,yaw'.split(','));w.writerows(rows)
        save_image(FIG/f'{name}_samples.png',np.vstack([np.hstack(sample_images[:2]),np.hstack(sample_images[2:])]))
        report=dict(video=name,frames=count,detected_frames=detected_frames,detection_frame_fraction=detected_frames/count,detection_rows=len(rows),median_pnp_rms_px=float(np.median(pnp_residuals)) if pnp_residuals else None)
        report.update(brightness=brightness,difference=difference,timestamp_fps=fps,input_size=list(size))
        if dist_differences:
            np.savetxt(OUT/f'distortion_comparison_{name}.csv',np.array(dist_differences),delimiter=',',header='frame,radius_from_principal_point_px,position_difference_m',comments='')
        if name=='远距离平移':
            times=np.arange(count)/150;filtered,prediction,accepted=filter_positions(obs)
            ids,future,metrics=evaluate(obs,prediction);report['prediction']=metrics
            report['stationary_baseline']=evaluate(obs,obs)[2];report['accepted_observations']=int(accepted.sum())
            np.savetxt(OUT/f'pred_{name}.csv',np.c_[times[np.isfinite(prediction).all(1)],prediction[np.isfinite(prediction).all(1)]],delimiter=',',header='t,x,y,z',comments='')
            np.savez_compressed(OUT/'translation_states.npz',t=times,observation=obs,filtered=filtered,prediction=prediction,accepted=accepted)
            fig,axes=plt.subplots(3,1,figsize=(11,8),sharex=True,constrained_layout=True)
            for j,ax in enumerate(axes):
                ax.plot(times,obs[:,j],alpha=.4,label='PnP observation');ax.plot(times,filtered[:,j],label='Filtered');ax.plot(times+.25,prediction[:,j],label='Prediction at target time');ax.set_ylabel('xyz'[j]+' (m)');ax.grid(alpha=.2)
            axes[0].legend();axes[-1].set_xlabel('Time (s)');fig.savefig(FIG/'translation_timeseries.png',dpi=160);plt.close(fig)
            fig,ax=plt.subplots(figsize=(7,5));ax.plot(obs[:,0],obs[:,2],'.',ms=2,alpha=.4);ax.plot(filtered[:,0],filtered[:,2]);ax.set(xlabel='x (m)',ylabel='z (m)',title='Top view: translation track');ax.axis('equal');fig.tight_layout();fig.savefig(FIG/'translation_xz.png',dpi=160);plt.close(fig)
            error=prediction[ids]-future
            np.savetxt(OUT/'prediction_evaluation.csv',np.c_[times[ids],times[ids]+.25,prediction[ids],future,error,np.linalg.norm(error,axis=1)],delimiter=',',header='issue_t,target_t,pred_x,pred_y,pred_z,obs_x,obs_y,obs_z,error_x,error_y,error_z,error_norm',comments='')
            fig,axes=plt.subplots(4,1,figsize=(11,10),sharex=True,constrained_layout=True)
            for j in range(3):
                axes[j].plot(times[ids]+.25,future[:,j],label='Future PnP observation');axes[j].plot(times[ids]+.25,prediction[ids,j],label='Prediction');axes[j].set_ylabel('xyz'[j]+' (m)');axes[j].grid(alpha=.2)
            axes[0].legend()
            axes[3].plot(times[ids]+.25,np.linalg.norm(error,axis=1));axes[3].set(xlabel='Target time (s)',ylabel='3D discrepancy (m)');fig.savefig(FIG/'prediction_error.png',dpi=160);plt.close(fig)
            qr=[];fig,ax=plt.subplots(figsize=(10,4))
            for q,r in [(30,.001),(300,.001),(3000,.001),(300,.0001),(300,.01)]:
                _,pred,_=filter_positions(obs,q=q,r=r);idx,truth,stats=evaluate(obs,pred);qr.append(dict(q=q,r=r,**stats));ax.plot(times[idx],np.linalg.norm(pred[idx]-truth,axis=1),alpha=.6,label=f'Qacc={q}, R={r}')
            ax.set(xlabel='Prediction issue time (s)',ylabel='3D discrepancy (m)');ax.legend();fig.tight_layout();fig.savefig(FIG/'qr_comparison.png',dpi=160);plt.close(fig);(OUT/'qr_metrics.json').write_text(json.dumps(qr,indent=2))
        reports.append(report);print(json.dumps(report,ensure_ascii=True),flush=True)
    metrics_path=OUT/'metrics.json'
    old=json.loads(metrics_path.read_text(encoding='utf-8')) if metrics_path.exists() else []
    combined={r['video']:r for r in old}
    combined.update({r['video']:r for r in reports})
    metrics_path.write_text(json.dumps(list(combined.values()),ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':main()
