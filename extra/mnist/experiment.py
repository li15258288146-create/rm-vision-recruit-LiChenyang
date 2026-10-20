"""Independent MNIST classification experiment; no other project data is used."""
import argparse
import csv
import gzip
import hashlib
import json
import platform
import random
import struct
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
FILES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(16 * 7 * 7, 32), nn.ReLU(),
            nn.Linear(32, 10),
        )

    def forward(self, x):
        return self.layers(x)


def load_data():
    folder = ROOT / "data"
    folder.mkdir(exist_ok=True)
    provenance = []
    arrays = {}
    for name, expected in FILES.items():
        path = folder / name
        url = "https://ossci-datasets.s3.amazonaws.com/mnist/" + name
        if not path.exists():
            print("Downloading", url, flush=True)
            temp = path.with_suffix(".part")
            urllib.request.urlretrieve(url, temp)
            if hashlib.md5(temp.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Download checksum mismatch: {name}")
            temp.replace(path)
        raw = path.read_bytes()
        assert hashlib.md5(raw).hexdigest() == expected, name
        provenance.append({"file": name, "url": url, "md5": expected,
                           "sha256": hashlib.sha256(raw).hexdigest()})
        content = gzip.decompress(raw)
        magic, count = struct.unpack(">II", content[:8])
        if magic == 2051:
            rows, cols = struct.unpack(">II", content[8:16])
            assert (rows, cols) == (28, 28)
            arrays[name] = np.frombuffer(content, np.uint8, offset=16).copy().reshape(count, 1, 28, 28)
        else:
            assert magic == 2049
            arrays[name] = np.frombuffer(content, np.uint8, offset=8).copy()
            assert len(arrays[name]) == count
    def images(prefix):
        return torch.from_numpy(arrays[prefix + "-images-idx3-ubyte.gz"]).float() / 255
    def labels(prefix):
        return torch.from_numpy(arrays[prefix + "-labels-idx1-ubyte.gz"]).long()
    return images("train"), labels("train"), images("t10k"), labels("t10k"), provenance


def evaluate(model, x, y):
    model.eval()
    correct = 0
    loss = 0.
    with torch.inference_mode():
        for start in range(0, len(y), 256):
            logits = model(x[start:start+256])
            correct += int((logits.argmax(1) == y[start:start+256]).sum())
            loss += float(nn.functional.cross_entropy(logits, y[start:start+256], reduction="sum"))
    return loss / len(y), correct / len(y)


def main():
    # ONNX exporter logs contain Unicode symbols, even in redirected Windows output.
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--checkpoint", type=Path, help="Skip training, evaluate/export/benchmark existing state_dict")
    args = parser.parse_args()
    assert args.epochs > 0
    # Explicit UTC+8 avoids relying on the host's timezone database.
    from datetime import timezone, timedelta
    now = datetime.now(timezone(timedelta(hours=8)))
    out = ROOT / "runs" / now.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    x, y, tx, ty, provenance = load_data()
    assert len(x) == 60000 and len(tx) == 10000
    perm = torch.randperm(60000, generator=torch.Generator().manual_seed(args.seed))
    train_idx, val_idx = perm[:55000], perm[55000:]
    np.savez_compressed(out / "split_indices.npz", train=train_idx.numpy(), validation=val_idx.numpy())
    model = SmallCNN()
    history = []
    best_accuracy = -1.
    checkpoint = out / "model.pt"
    train_start = time.perf_counter()
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
        torch.save(model.state_dict(), checkpoint)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        for epoch in range(1, args.epochs + 1):
            start = time.perf_counter()
            model.train()
            order = train_idx[torch.randperm(len(train_idx))]
            total_loss = 0.
            for batch in order.split(128):
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.cross_entropy(model(x[batch]), y[batch])
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach()) * len(batch)
            val_loss, accuracy = evaluate(model, x[val_idx], y[val_idx])
            row = dict(epoch=epoch, train_loss=total_loss/len(train_idx),
                       validation_loss=val_loss, validation_accuracy=accuracy,
                       seconds=time.perf_counter()-start)
            history.append(row)
            print(json.dumps(row), flush=True)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), checkpoint)
    train_seconds = time.perf_counter() - train_start
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    torch.set_num_threads(1)
    onnx_path = out / "model.onnx"
    torch.onnx.export(
        model, (torch.zeros(2, 1, 28, 28),), str(onnx_path),
        input_names=["image"], output_names=["logits"], opset_version=18,
        dynamo=True, external_data=False,
        dynamic_shapes={"x": {0: torch.export.Dim("batch", min=1, max=256)}},
    )
    onnx.checker.check_model(onnx.load(str(onnx_path)))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(onnx_path), sess_options=options,
                                  providers=["CPUExecutionProvider"])
    confusion = np.zeros((10, 10), dtype=np.int64)
    correct_pt = correct_ort = agreement = 0
    max_abs = 0.
    with torch.inference_mode():
        for start in range(0, len(ty), 256):
            batch = tx[start:start+256]
            pt = model(batch).numpy()
            ox = session.run(None, {"image": batch.numpy()})[0]
            np.testing.assert_allclose(pt, ox, rtol=1e-4, atol=1e-4)
            max_abs = max(max_abs, float(np.abs(pt - ox).max()))
            p, q = pt.argmax(1), ox.argmax(1)
            truth = ty[start:start+256].numpy()
            correct_pt += int((p == truth).sum())
            correct_ort += int((q == truth).sum())
            agreement += int((p == q).sum())
            np.add.at(confusion, (truth, p), 1)
    np.savetxt(out / "confusion_matrix.csv", confusion, fmt="%d", delimiter=",")
    # Same preloaded float32 inputs; no disk IO, normalization or setup in timing.
    timings = []
    summaries = []
    with torch.inference_mode():
        for size in (1, 32):
            inputs = [tx[i:i+size].contiguous() for i in range(0, 32*size, size)]
            numpy_inputs = [v.numpy() for v in inputs]
            calls = {
                "pytorch": lambda i: model(inputs[i % len(inputs)]),
                "onnxruntime": lambda i: session.run(None, {"image": numpy_inputs[i % len(inputs)]}),
            }
            for call in calls.values():
                for i in range(100):
                    call(i)
            for repeat in range(5):
                names = list(calls) if repeat % 2 == 0 else list(reversed(calls))
                for name in names:
                    for i in range(500):
                        t0 = time.perf_counter_ns()
                        calls[name](i)
                        elapsed = (time.perf_counter_ns() - t0) / 1e6
                        timings.append([size, name, repeat, i, elapsed])
            for name in calls:
                ms = np.array([row[4] for row in timings if row[0] == size and row[1] == name])
                summaries.append(dict(batch=size, engine=name, samples=len(ms),
                                      mean_ms=float(ms.mean()), median_ms=float(np.median(ms)),
                                      p95_ms=float(np.percentile(ms, 95)),
                                      images_per_second=float(size*1000/ms.mean())))
    with (out / "timings.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["batch", "engine", "repeat", "iteration", "milliseconds"])
        writer.writerows(timings)
    try:
        cpu = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
            text=True).strip()
    except (OSError, subprocess.SubprocessError):
        cpu = platform.processor()
    results = {
        "scope": "Independent MNIST handwritten-digit classification only",
        "started_at": now.isoformat(), "seed": args.seed, "cpu": cpu,
        "platform": platform.platform(), "python": platform.python_version(),
        "versions": {"torch": torch.__version__, "numpy": np.__version__,
                     "onnx": onnx.__version__, "onnxruntime": ort.__version__},
        "split": {"train": 55000, "validation": 5000, "test": 10000},
        "preprocessing": "float32 grayscale / 255, NCHW, no augmentation",
        "parameters": sum(p.numel() for p in model.parameters()),
        "history": history, "training_seconds": train_seconds,
        "checkpoint_source": str(args.checkpoint) if args.checkpoint else "best validation accuracy",
        "test": {"pytorch_correct": correct_pt, "onnx_correct": correct_ort,
                 "count": 10000, "prediction_agreement": agreement/10000,
                 "max_absolute_logit_difference": max_abs},
        "benchmark": summaries, "inference_threads": 1, "warmup_calls": 100,
        "repeats": 5, "calls_per_repeat": 500, "providers": session.get_providers(),
        "data": provenance,
        "artifacts": {p.name: {"bytes": p.stat().st_size,
                              "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                      for p in (checkpoint, onnx_path)},
    }
    (out / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    (ROOT / "LATEST.txt").write_text(out.name + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False), flush=True)
    print("RESULT_DIRECTORY", out, flush=True)


if __name__ == "__main__":
    main()
