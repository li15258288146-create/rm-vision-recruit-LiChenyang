"""滑动条调节阈值；Space 暂停，S 保存截图与参数，Q 退出。"""
import argparse,json
from pathlib import Path
import cv2,numpy as np
from aim import detect


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video',type=Path,default=Path(__file__).resolve().parents[1]/'data/armor/远距离平移.mp4')
    parser.add_argument('--blue',action='store_true');args=parser.parse_args()
    cap=cv2.VideoCapture(str(args.video))
    if not cap.isOpened():raise ValueError(f'Cannot open video: {args.video}')
    window='Threshold tuning - Space pause / S save / Q quit';cv2.namedWindow(window)
    cv2.createTrackbar('Brightness',window,140,255,lambda x:None);cv2.createTrackbar('Color difference',window,60,255,lambda x:None)
    pause=False;image=None
    try:
        while True:
            if not pause or image is None:
                ok,image=cap.read()
                if not ok:cap.set(cv2.CAP_PROP_POS_FRAMES,0);continue
            brightness=cv2.getTrackbarPos('Brightness',window);difference=cv2.getTrackbarPos('Color difference',window)
            detections,mask=detect(image,'blue' if args.blue else 'red',brightness,difference);shown=image.copy()
            for d in detections:cv2.polylines(shown,[np.round(d['points']).astype(np.int32)],True,(0,255,0),2)
            panel=np.hstack([cv2.resize(shown,(640,480)),cv2.resize(cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR),(640,480))]);cv2.imshow(window,panel)
            key=cv2.waitKey(15)&255
            if key==ord('q') or cv2.getWindowProperty(window,cv2.WND_PROP_VISIBLE)<1:break
            if key==32:pause=not pause
            if key==ord('s'):
                out=Path(__file__).resolve().parent/'figures';out.mkdir(exist_ok=True)
                cv2.imencode('.png',panel)[1].tofile(str(out/'threshold_tool_panel.png'))
                (out/'threshold_settings.json').write_text(json.dumps(dict(brightness=brightness,difference=difference,video=str(args.video))),encoding='utf-8')
                print('Saved processing panel and settings; capture window separately if trackbar screenshot is required.')
    finally:cap.release();cv2.destroyAllWindows()


if __name__=='__main__':main()
