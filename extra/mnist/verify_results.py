"""Recompute recorded metrics and check saved model predictions independently."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from experiment import ROOT, SmallCNN, load_data


def main():
    run = ROOT / "runs" / (ROOT / "LATEST.txt").read_text().strip()
    result = json.loads((run / "results.json").read_text(encoding="utf-8"))
    for name, info in result["artifacts"].items():
        assert hashlib.sha256((run / name).read_bytes()).hexdigest() == info["sha256"]
    split = np.load(run / "split_indices.npz")
    assert len(split["train"]) == 55000 and len(split["validation"]) == 5000
    np.testing.assert_array_equal(np.sort(np.concatenate(list(split.values()))), np.arange(60000))
    with (run / "timings.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 10000
    for item in result["benchmark"]:
        values = np.array([float(r["milliseconds"]) for r in rows
                           if int(r["batch"]) == item["batch"] and r["engine"] == item["engine"]])
        assert len(values) == 2500 and (values > 0).all()
        for key, actual in [("mean_ms", values.mean()), ("median_ms", np.median(values)),
                            ("p95_ms", np.percentile(values, 95))]:
            assert np.isclose(item[key], actual, atol=1e-12)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    model = SmallCNN().eval()
    model.load_state_dict(torch.load(run / "model.pt", weights_only=True))
    onnx.checker.check_model(onnx.load(str(run / "model.onnx")))
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = opts.inter_op_num_threads = 1
    session = ort.InferenceSession(str(run / "model.onnx"), opts, providers=["CPUExecutionProvider"])
    _, _, x, y, _ = load_data()
    correct = [0, 0]
    matrix = np.zeros((10, 10), dtype=int)
    with torch.inference_mode():
        for start in range(0, len(y), 256):
            pt = model(x[start:start+256]).numpy()
            ox = session.run(None, {"image": x[start:start+256].numpy()})[0]
            np.testing.assert_allclose(pt, ox, rtol=1e-4, atol=1e-4)
            np.testing.assert_array_equal(pt.argmax(1), ox.argmax(1))
            truth = y[start:start+256].numpy()
            correct[0] += int((pt.argmax(1) == truth).sum())
            correct[1] += int((ox.argmax(1) == truth).sum())
            np.add.at(matrix, (truth, pt.argmax(1)), 1)
        for size in (1, 32):
            np.testing.assert_allclose(model(x[:size]).numpy(),
                                       session.run(None, {"image": x[:size].numpy()})[0],
                                       rtol=1e-4, atol=1e-4)
    assert correct == [result["test"]["pytorch_correct"], result["test"]["onnx_correct"]]
    np.testing.assert_array_equal(matrix, np.loadtxt(run / "confusion_matrix.csv", delimiter=","))
    check = dict(status="passed", test_correct=correct, test_count=len(y),
                 timing_rows=len(rows), checks=["model SHA256", "dataset MD5", "split partition",
                 "all timing statistics", "ONNX structure", "full test-set predictions",
                 "confusion matrix", "batch 1 and 32 numerical parity"])
    (run / "verification.json").write_text(json.dumps(check, indent=2), encoding="utf-8")
    print(json.dumps(check, indent=2))


if __name__ == "__main__":
    main()
