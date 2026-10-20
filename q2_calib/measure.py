"""原图点击 A4 四角进行 PnP 测距；物理顺序：左上、右上、右下、左下。"""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path

import cv2
import numpy as np

from calibrate import ROOT, read_image, save_image


def select_points(image):
    h, w = image.shape[:2]
    scale = min(1.0, 1400 / w, 850 / h)
    dw, dh = max(1, round(w * scale)), max(1, round(h * scale))
    preview = cv2.resize(image, (dw, dh))
    sx, sy = dw / w, dh / h
    points = []
    window = "A4 - TL TR BR BL / R reset / Enter confirm / Esc cancel"

    def callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append([x / sx, y / sy])

    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, callback)
    print("按纸张物理方向点击左上、右上、右下、左下；R 重选，Enter 确认，Esc 取消。")
    try:
        while True:
            canvas = preview.copy()
            for i, (x, y) in enumerate(points):
                p = (round(x * sx), round(y * sy))
                cv2.circle(canvas, p, 4, (0, 0, 255), -1)
                cv2.putText(canvas, str(i + 1), p, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow(window, canvas)
            key = cv2.waitKey(20) & 0xFF
            if key == ord("r"):
                points.clear()
            elif key in (10, 13) and len(points) == 4:
                return np.asarray(points, np.float64)
            elif key == 27 or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                raise SystemExit("已取消")
    finally:
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--true-distance", type=float, required=True, help="实测纸中心距离，m")
    parser.add_argument("--camera", type=Path,
                        default=ROOT / "results/retake_check/calibration/camera.json",
                        help="默认使用补拍后 24 张照片的标定参数")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    parser.add_argument("--points", type=Path, help="复用此前保存的 points.json，跳过点击")
    args = parser.parse_args()
    if not np.isfinite(args.true_distance) or args.true_distance <= 0:
        parser.error("实测距离必须为有限正数")
    config = json.loads(args.camera.read_text(encoding="utf-8"))
    K = np.asarray(config["camera_matrix"], np.float64)
    dist = np.asarray(config["distortion_coefficients"], np.float64)
    image = read_image(args.image)
    if list(image.shape[1::-1]) != config["image_size"]:
        raise ValueError("测距图与标定图分辨率不同")
    if args.points:
        record = json.loads(args.points.read_text(encoding="utf-8"))
        if Path(record["image"]).resolve() != args.image.resolve():
            raise ValueError("角点记录不属于当前图片")
        points = np.asarray(record["image_points"], np.float64)
    else:
        points = select_points(image)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError("需要 4 个有限的二维角点")
    h, w = image.shape[:2]
    if np.any(points < 0) or np.any(points[:, 0] >= w) or np.any(points[:, 1] >= h):
        raise ValueError("角点越出图像边界")
    contour = points.astype(np.float32).reshape(-1, 1, 2)
    if not cv2.isContourConvex(contour) or abs(cv2.contourArea(contour)) < 10:
        raise ValueError("角点应形成凸四边形，请检查点击顺序")
    obj = np.array([[-.105, -.1485, 0], [.105, -.1485, 0],
                    [.105, .1485, 0], [-.105, .1485, 0]], np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, points, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok or not np.isfinite(tvec).all() or not np.isfinite(rvec).all():
        raise RuntimeError("PnP 求解失败")
    R, _ = cv2.Rodrigues(rvec)
    if np.any((R @ obj.T + tvec)[2] <= 0):
        raise RuntimeError("角点在相机后方，请检查点序和参数")
    projected, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    rms = float(np.sqrt(np.mean(np.sum((points - projected.reshape(-1, 2)) ** 2, axis=1))))
    distance = float(np.linalg.norm(tvec))
    error = distance - args.true_distance
    args.out.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    stem = f"{args.image.stem}_{run_id}"
    points_name = stem + "_points.json"
    (args.out / points_name).write_text(json.dumps(dict(
        image=str(args.image.resolve()), image_points=points.tolist(),
        order=["top_left", "top_right", "bottom_right", "bottom_left"],
        rvec=rvec.ravel().tolist(), tvec_m=tvec.ravel().tolist()), indent=2), encoding="utf-8")
    csv_path = args.out / "measurements.csv"
    needs_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if needs_header:
            writer.writerow(["run_id", "image", "true_m", "estimated_m", "depth_z_m",
                             "signed_error_m", "abs_error_m", "relative_error_percent",
                             "reprojection_rms_px", "points_file", "camera_file"])
        writer.writerow([run_id, args.image.name, args.true_distance, distance, float(tvec[2, 0]),
                         error, abs(error), abs(error) / args.true_distance * 100, rms,
                         points_name, str(args.camera.resolve())])
    marked = image.copy()
    poly = np.round(points).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(marked, [poly], True, (0, 255, 0), 2)
    for i, p in enumerate(poly[:, 0]):
        cv2.putText(marked, str(i + 1), tuple(p), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 255), 2)
    save_image(args.out / (stem + "_marked.png"), marked)
    print(f"空间距离：{distance:.4f} m；深度：{tvec[2, 0]:.4f} m")
    print(f"绝对误差：{abs(error):.4f} m；相对误差：{abs(error)/args.true_distance*100:.2f}%")
    print(f"重投影 RMS：{rms:.3f} px；结果：{args.out.resolve()}")


if __name__ == "__main__":
    main()
