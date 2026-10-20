# B 题运行说明

## 环境与目录

命令均在项目根目录执行。拷贝时保留结构：

```text
项目根目录/
├── data/rmuc_2025_map.pcd
├── data/map/（rmuc_2025.pgm、rmuc_2025.yaml、queries.json）
├── tools/check_path.py
└── qB_sentry/（源码、文档、path1～5.csv、results/）
```

本机已使用 Python 3.13 运行。直接依赖见 requirements.txt，本机版本快照见 requirements-lock.txt；锁定文件不保证其他平台都有对应 wheel。当前 .venv 已可用，不必重复创建。新 Windows 环境：

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r qB_sentry/requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

Linux 用 `python3 -m venv .venv`，后续解释器替换为 `.venv/bin/python`；Linux 全流程尚未实测。

## 先检查已有结果

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s qB_sentry -p "test_*.py"
.\.venv\Scripts\python.exe qB_sentry/verify_submission.py
```

24 项单元测试。提交验证覆盖主提交 5 条、基本题两算法 10 条、自建图 10 条、最新路径三图 30 个组合，共 55 次官方检查。另检查原端点、最大点距、整段密采样，以及第 6、7 组失败原因。结果写入 results/submission_verification.json，任何断言失败会以非零状态退出。

单独检查第 3 组最新路线：

```powershell
.\.venv\Scripts\python.exe tools/check_path.py data/map/rmuc_2025.yaml qB_sentry/results/robust/astar/path3.csv
```

官方检查器失败也可能返回退出码 0，必须看末行“通过/没通过”；验证脚本已处理。第 6、7 组不生成伪造的 path CSV。

## 分阶段重新运行

命令会更新对应结果目录；如需保留本次提交快照，先复制整个项目到另一个目录运行。点云和配准计算较耗时，依赖版本、浮点和并行计算可能造成细微差异，重跑后应重新验证和更新报告。

### 基本题

```powershell
.\.venv\Scripts\python.exe qB_sentry/run_planning.py
```

输出根目录 path1～5.csv、results/summary.csv/json、搜索图及检查日志，额外包含曼哈顿和两倍启发式实验。

### 从点云开始重建

按依赖顺序运行：

```powershell
.\.venv\Scripts\python.exe qB_sentry/build_map.py
.\.venv\Scripts\python.exe qB_sentry/terrain_advanced.py
.\.venv\Scripts\python.exe qB_sentry/register_map.py
.\.venv\Scripts\python.exe qB_sentry/reconstruct_map.py
.\.venv\Scripts\python.exe qB_sentry/refine_registration.py
.\.venv\Scripts\python.exe qB_sentry/evaluate_optimized.py
.\.venv\Scripts\python.exe qB_sentry/audit_reconstruction.py
.\.venv\Scripts\python.exe qB_sentry/robust_boundaries.py
```

build_map 生成调平点云；早期 register_map 提供双朝向初始化；reconstruct_map 是最终独立建图；refine_registration 精配准；evaluate/audit 验证独立结果，robust_boundaries 生成融合地图及容错路径。

**缓存注意：每次重新生成或替换 results/pointcloud/leveled.ply 后，应先将 results/optimized/point_geometry.npz 移入备份目录，再运行 reconstruct_map.py。** 缓存没有输入哈希校验，不能配新点云使用旧缓存。首次无缓存时无需操作；只重跑 robust_boundaries 不涉及此缓存。

独立图评估保留了交叉到给定图的失败日志，这是实验事实；最新融合路径应通过三图检查。表面删点验证不能当作真实遮挡区域的真值验证。

### 仿真与动画

完成上述基本目录生成后：

```powershell
.\.venv\Scripts\python.exe qB_sentry/decision_sim.py
.\.venv\Scripts\python.exe qB_sentry/visualize_advanced.py
```

输出 results/decision/simulation.mp4、状态图和日志，以及 results/advanced/search_replay.mp4。仿真使用给定图，不包含定位误差、实车动力学或完整比赛规则。

最后再运行前述测试和提交验证。历史报告数字是保存的实验快照，不会随重跑自动更新；新运行后应以 JSON 为准修改报告。

## 常见问题与提交

- 找不到资源：仅复制 qB_sentry 不足以复现，需保留 data/tools 结构。
- Open3D 安装失败：核对 Python 与平台的 wheel 支持；本机验证环境是 Python 3.13。
- 未知区域可走：核对 negate 和阈值；128 是当前未知值，205 在 free_thresh=0.25 时会被判为空闲。
- 第 3 组部分扰动失败：固定起点限制，不要改端点或半径绕过。
- 检查失败：先确认 YAML 对应的地图类别，再查看出界、碰撞、跳跃日志。
- 隐藏数据：planner.search 可复用，示例驱动默认读题目路径。verify_submission 的固定 SHA256 仅用于保护本题输入，不是任意数据测试接口；新数据需调用 GridMap/search 或编写新输入驱动。
- 提交材料包括全部 .py、依赖文件、文档、主路径以及 results 中图表/日志/地图/视频。大体积 PLY、NPZ 是重建中间结果，可由原始 PCD 重新生成；若不提交，须说明不能直接跳过重建阶段。
- GitHub 仓库名、个人信息及完整考核根目录 README 按原题要求填写；不要伪造持续提交历史。本次未代为发布仓库。
