# B 题：环境建图与路径规划

本目录为 B 题提交材料。**最终结论以 [FINAL_REPORT.md](FINAL_REPORT.md) 为准**；其他阶段报告保留迭代过程。

## 阅读入口

1. [最终答题报告](FINAL_REPORT.md)：完成情况、方案、结果、六个必答问题与局限。
2. [运行说明](RUN_GUIDE.md)：环境、快速检查、完整复现及故障排查。
3. [代码导读与答辩提纲](DEFENSE_GUIDE.md)：模块职责、讲解顺序、常见追问。
4. [AI 使用记录](ai_log.md)：五个关键交互及错误纠正，个人体会请本人核实后补充。

## 三类结果，提交时不要混用

| 用途 | 路径与地图 | 结论 |
|---|---|---|
| 基本题主提交 | 本目录 path1.csv～path5.csv；题目给定地图 | 第 1～5 组可达；第 6 组目标净空不足，第 7 组越界，不伪造路径 |
| 独立建图实验 | results/optimized/astar/、dijkstra/；registered_map.yaml | 10 条路径自建图通过，但这些路线交叉到给定图失败 |
| 最新双地图优化 | results/robust/astar/、dijkstra/；fused_map.yaml | 同时通过给定、自建、融合图，共 30 次检查通过；不冒充纯点云建图 |

第 3 组最新路线名义检查通过，但 27 种离散配准扰动中仅 18 种整条通过；另 9 种在原始起点就不满足净空。端点和半径 0.30 m 不变。

![最新路径与风险](results/robust/paths_and_risk.png)

## 快速复核

在包含 data/、tools/、qB_sentry/ 的项目根目录执行：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s qB_sentry -p "test_*.py"
.\.venv\Scripts\python.exe qB_sentry/verify_submission.py
```

验证既有文件，不重建点云，结果写入 results/submission_verification.json。

## 历史材料

[基础实验](BASELINE_REPORT.md)、[早期发挥实验](ADVANCED_REPORT.md)、[独立地图优化](OPTIMIZATION_REPORT.md)、[双地图优化细节](ROBUST_BOUNDARIES_REPORT.md)。旧版自建图曾用灰度 205 表示未知，与阈值不匹配，旧日志不能证明未知区被正确阻挡；当前使用 128。

本目录整理不代表已上传 GitHub，也不替代公共题或 A 题提交。
