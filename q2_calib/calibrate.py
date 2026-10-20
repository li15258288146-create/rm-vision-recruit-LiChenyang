"""棋盘格标定；默认 9x6 内角点、20 mm 方格。尺寸须按打印实测修改。"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent


def read_image(path):
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"读取图像失败：{path}")
    return image


def save_image(path, image):
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"图像编码失败：{path}")
    encoded.tofile(str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=ROOT / "images/calib")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    parser.add_argument("--square-mm", type=float, default=20.0)
    parser.add_argument("--cols", type=int, default=9, help="内角点列数")
    parser.add_argument("--rows", type=int, default=6, help="内角点行数")
    parser.add_argument("--min-images", type=int, default=15,
                        help="基本实验至少 15；5 张对比实验可设为 5")
    args = parser.parse_args()
    if not np.isfinite(args.square_mm) or args.square_mm <= 0:
        parser.error("方格尺寸必须为有限正数")
    if min(args.cols, args.rows) < 3 or args.min_images < 3:
        parser.error("内角点行列数和最少图像数须至少为 3")
    if not args.images.is_dir():
        parser.error(f"图像目录不存在：{args.images}")
    args.out.mkdir(parents=True, exist_ok=True)
    pattern = (args.cols, args.rows)
    template = np.zeros((args.cols * args.rows, 3), np.float32)
    template[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2)
    template *= args.square_mm / 1000
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.001)
    objects, observations, used = [], [], []
    image_size = None
    for path in sorted(args.images.iterdir()):
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
            continue
        image = read_image(path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        if image_size is None:
            image_size = size
        elif size != image_size:
            raise ValueError(f"图像尺寸不一致：{path.name}，请勿混用分辨率")
        found, corners = cv2.findChessboardCorners(
            gray, pattern, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not found:
            print("未检测到完整棋盘格：", path.name)
            continue
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objects.append(template.copy())
        observations.append(corners)
        used.append(path)
        cv2.drawChessboardCorners(image, pattern, corners, True)
        save_image(args.out / f"corners_{path.stem}.png", image)
    if len(used) < args.min_images:
        raise RuntimeError(f"有效图像 {len(used)} 张，需要至少 {args.min_images} 张，请补拍")
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objects, observations, image_size, None, None)
    if not all(np.isfinite(x).all() for x in (np.asarray(rms), K, dist)):
        raise RuntimeError("标定结果非有限值，请检查角点及姿态多样性")
    with (args.out / "reprojection_errors.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "rms_px"])
        for path, obj, observed, rvec, tvec in zip(used, objects, observations, rvecs, tvecs):
            projected, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
            delta = observed.reshape(-1, 2) - projected.reshape(-1, 2)
            writer.writerow([path.name, float(np.sqrt(np.mean(np.sum(delta ** 2, axis=1))))])
    parameters = dict(
        image_size=list(image_size), pattern_inner_corners=list(pattern),
        square_size_m=args.square_mm / 1000, camera_matrix=K.tolist(),
        distortion_coefficients=dist.ravel().tolist(), rms_px=float(rms),
        used_images=[p.name for p in used],
        rvecs=[v.ravel().tolist() for v in rvecs],
        tvecs_m=[v.ravel().tolist() for v in tvecs])
    (args.out / "camera.json").write_text(
        json.dumps(parameters, ensure_ascii=False, indent=2), encoding="utf-8")
    sample = read_image(used[0])
    corrected = cv2.undistort(sample, K, dist, None, K)
    save_image(args.out / "undistort_comparison.png", np.hstack([sample, corrected]))
    print(f"有效图像：{len(used)}；总体 RMS：{rms:.4f} px")
    print("K =\n", K, "\n畸变系数：", dist.ravel())
    print("结果：", args.out.resolve())


if __name__ == "__main__":
    main()
