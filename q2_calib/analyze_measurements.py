"""从真实测距 CSV 生成误差图和汇总；同一图片多次标点默认只取最后一次。"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=ROOT / "results/measurements.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "results/distance_analysis")
    args = parser.parse_args()
    if not args.csv.exists():
        parser.error("尚无测距记录，请先运行 measure.py 完成真实测量")
    with args.csv.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        parser.error("测距 CSV 没有数据")
    latest = {}
    for row in rows:
        latest[row["image"]] = row
    used = list(latest.values())
    if len({row["camera_file"] for row in used}) != 1:
        parser.error("记录混用了不同相机参数文件，请分开分析各个实验")
    true = np.array([float(row["true_m"]) for row in used])
    estimated = np.array([float(row["estimated_m"]) for row in used])
    if not np.isfinite(true).all() or not np.isfinite(estimated).all() or np.any(true <= 0):
        parser.error("距离应为有限值，真实距离必须大于零")
    order = np.argsort(true)
    true, estimated = true[order], estimated[order]
    error = estimated - true
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {
        "raw_rows": len(rows), "unique_images": len(used),
        "deduplication": "last CSV row per image; repeated clicks are not independent captures",
        "camera_file": used[0]["camera_file"],
        "true_distance_min_m": float(true.min()),
        "true_distance_max_m": float(true.max()),
        "mean_absolute_error_m": float(np.mean(np.abs(error))),
        "rmse_m": float(np.sqrt(np.mean(error ** 2))),
        "mean_signed_error_m": float(np.mean(error)),
        "mean_absolute_relative_error_percent": float(np.mean(np.abs(error) / true) * 100),
        "note": "Number of physical distance groups must be checked against experiment notes.",
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (args.out / "evaluated_measurements.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(used[0]))
        writer.writeheader()
        writer.writerows(used)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    axes[0].scatter(true, estimated)
    limits = [min(true.min(), estimated.min()), max(true.max(), estimated.max())]
    axes[0].plot(limits, limits, "k--", label="Ideal")
    axes[0].set_ylabel("Estimated distance (m)")
    axes[0].legend()
    axes[1].scatter(true, np.abs(error) * 1000)
    axes[1].set_ylabel("Absolute error (mm)")
    axes[2].scatter(true, np.abs(error) / true * 100)
    axes[2].set_ylabel("Absolute relative error (%)")
    for ax in axes:
        ax.set_xlabel("Measured reference distance (m)")
        ax.grid(alpha=.25)
    fig.savefig(args.out / "distance_errors.png", dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2))
    print("Saved:", args.out.resolve())


if __name__ == "__main__":
    main()
