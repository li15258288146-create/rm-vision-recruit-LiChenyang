# MNIST 独立替代实验

本实验仅分类公开 MNIST 手写数字，不接入 A 题视频或自瞄系统，不宣称完成原题装甲板数字识别。完整结果与实验限制见 [附加题报告](../REPORT.md)。

## 运行

在项目根目录 PowerShell 执行，现有 `.venv` 已安装所需依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install -r extra/mnist/requirements.txt
.\.venv\Scripts\python.exe -u extra/mnist/experiment.py
.\.venv\Scripts\python.exe extra/mnist/verify_results.py
```

从零建立独立环境时用 Python 3.13：

```powershell
py -3.13 -m venv .venv_mnist
.\.venv_mnist\Scripts\python.exe -m pip install -r extra/mnist/requirements.txt
.\.venv_mnist\Scripts\python.exe extra/mnist/experiment.py
```

每次运行新建 `runs/时间戳/`，成功后才更新 `LATEST.txt`。首次下载四个 MNIST gzip 文件并校验 MD5；以后直接使用保留数据，不需重新下载。不依赖 torchvision，也不需要 GPU。

重新评估已有模型、导出并测速，不训练：

```powershell
.\.venv\Scripts\python.exe extra/mnist/experiment.py --checkpoint extra/mnist/runs/20261020_183707/model.pt
```

原报告引用固定目录 `runs/20261020_183707/`，新一次运行不会改变原报告中的数字。`verify_results.py` 检查 LATEST 指向的结果。

## 文件

- `experiment.py`：下载/校验、IDX 解析、固定划分、训练、按验证集选模型、导出、测试、测速。
- `verify_results.py`：重新加载两种模型，全测试集检查，重算计时统计并校验文件 SHA256。
- `data/`：复现实验所需的四个原始压缩数据文件，不应当作缓存排除。
- `runs/20261020_183707/`：本次成功结果，含权重、单文件 ONNX、划分索引、原始计时 CSV、混淆矩阵、JSON 和核查记录。
- `experiment_console.log` 与 `runs/20261020_183627/`：首次导出失败记录；训练完成但没有成功导出，不能作为最终结果。
- `experiment_success_console.log`：完整成功运行日志。

初次导出因 Windows GBK 编码不能打印导出器的 Unicode 日志而中止，脚本现显式使用 UTF-8。导出器提示跳过未安装 torchvision 的检测算子，本 CNN 不使用这些算子；是否成功以 ONNX 结构检查和全测试集数值比较为准。

实验固定种子和确定性算法，但不同硬件/库版本仍可能有数值差异；时延尤其会随机器负载、温度和功耗状态变化。报告中训练耗时含验证和保存，不含数据下载/解析、模型构建与导出。
