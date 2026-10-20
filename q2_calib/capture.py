"""保存未缩放、未叠字的摄像头原图。S 保存，Q 退出。"""
import argparse
from datetime import datetime
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()
    args.folder.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(args.camera)
    try:
        if not cap.isOpened():
            raise RuntimeError("无法打开摄像头，请检查设备编号及相机权限")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        print("S 保存原图，Q 退出。分辨率请求可能不被设备接受，以保存尺寸为准。")
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("读取摄像头失败")
            cv2.imshow("Capture - S save / Q quit", frame)
            key = cv2.waitKey(20) & 0xFF
            if key == ord("s"):
                path = args.folder / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".png")
                ok, encoded = cv2.imencode(".png", frame)
                if not ok:
                    raise RuntimeError("图像编码失败")
                encoded.tofile(str(path))
                print("保存：", path, "尺寸：", frame.shape[1::-1])
            elif key == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
