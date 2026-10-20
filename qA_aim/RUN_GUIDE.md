# A题已有程序运行说明

本文整理自已有 BASIC_REPORT.md，不新增算法、参数配置或运行功能。保留项目根目录下 data/armor 与 qA_aim 的相对位置；依赖见 requirements.txt。已有环境为 Python 3.13。执行批处理会更新已有输出，需保留快照时请先备份。

## 已有运行步骤


在D:\RM1或完整项目根目录执行，无需启动摄像头，输入是题目视频：

```powershell
.\.venv\Scripts\python.exe -m pip install -r qA_aim/requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s qA_aim -p "test_*.py"
.\.venv\Scripts\python.exe qA_aim/run_aim.py --videos 远距离平移 中距离陀螺
.\.venv\Scripts\python.exe qA_aim/verify_basic.py
```

阈值工具：

```powershell
.\.venv\Scripts\python.exe qA_aim/tune_threshold.py
```

Space暂停在当前帧，拖动Brightness和Color difference查看掩码与框选；S保存figures/threshold_settings.json和处理面板，Q退出。S保存的面板不含系统滑动条，已有threshold_tool_window.jpg为此前完整窗口截图，本轮检查了图像，未重新执行鼠标滑动操作。

保存后批处理：

```powershell
.\.venv\Scripts\python.exe qA_aim/run_aim.py --videos 远距离平移 中距离陀螺 --settings qA_aim/figures/threshold_settings.json
```

或直接传 `--brightness 140 --difference 60`。重新调参会更新结果和统计，报告中的数字为本次默认阈值实验。


## 运行结果的适用范围

两段基本题视频的完整解码和输出复核记录见 basic_validation.json；近距离与蓝色视频为前次保存结果，本次文档整理没有重跑。已有6项测试通过的记录来自基本题复核，本次未重新执行测试。小陀螺中心和角速度估计没有实现，不存在对应运行入口。
