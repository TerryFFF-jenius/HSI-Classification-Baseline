# 自适应空间 residual gate：本地正确性报告

任务状态：**授权的实现及本地正确性验证已完成，待主线程最终审核；训练仍未授权**。初轮曾因缺少实际 checkpoint 而标为部分完成；现已通过用户提供的 exec_03 epoch 192 真实 checkpoint 定向严格加载补验。没有进入训练阶段，不新增导师阶段编号。原始 `verification.json`、`commands.json`、`command_history.json` 保留初轮原始结果，不追改为本轮运行记录；本轮结果独立记录在 `checkpoint_verification.json`。

## 真实 checkpoint 定向补验

用户下载并指定的只读输入为 `D:\HSI_Baseline\checkpoint_inputs\exec03_serial\checkpoint_epoch_192.pth`。远程来源仅作为 provenance 记录：`/root/HSI_Baseline_honghu_diag_20261018/diagnostics/honghu_single_seed_20261018/runs_20261018_exec_03/gscvit_dssr_spatial/checkpoint_epoch_192.pth`，本线程没有连接 AutoDL 或重新下载。

验证前、验证后的 SHA256 均为 `606ef81f050cda8171dd0effe05f77ef6a24c12289942f5e076f253cf0183304`。使用 `torch.load(..., map_location='cpu', weights_only=True)` 读取原始 106 键 state_dict；没有重命名、过滤、补键或使用 strict=False。冻结旧实现和新实现 gate=False 模型均 strict=True 成功，无 missing/unexpected keys。两边空间状态显式开启，强度 0.5；模型均 eval()，仅 inference_mode 合成输入。

| B | 输入形状 | 输出形状 | max_abs_diff | 判定 |
| --- | --- | --- | ---: | --- |
| 1 | `[1,1,15,7,7]` | `[1,22]` | 0.0 | PASS |
| 2 | `[2,1,15,7,7]` | `[2,22]` | 0.0 | PASS |

沿用 float32 `atol=1e-6, rtol=1e-5`；输出全部有限，eval 后两边参数及 buffers 仍与加载的 state_dict 相同。Python 3.13.9、PyTorch 2.13.0+cpu、CPU、1 线程。实际补验命令及退出码：

```cmd
C:\Users\PC\Anaconda3\python.exe diagnostics\spatial_residual_gate_20261019\verify_checkpoint.py
```

退出码 **0**，完整运行命令、环境、严格加载结果、数值和三个实际加载源码 SHA256 见 `checkpoint_verification.json`。三个模型源码哈希与初轮已验收版本相同；本轮未修改 gate 实现或重跑整个初轮合成套件。原 `before.json` 保持不变；audit.py 只将上述精确路径且哈希匹配的 checkpoint 作为授权输入单独列出，不放宽其他文件检查。checkpoint **不纳入 Git 暂存建议**。

未来训练设计仍需控制共享权重初始化和训练随机状态：开启 gate 时创建卷积会消耗随机数，随后清零不会恢复随机状态，不能仅以 seed 相同认定训练对照完全一致。本轮未因此修改实现。

本任务只验证代码契约，不判断精度收益。已有 Rp/Rs=2.296、cos=0.0667 及 0.5→0.25 单 seed 无收益的结果仅作为用户提供的背景，未重新搜证、计算或推导因果结论。

## 实现与文件

修改以下三个文件：

- `compare/gscvit.py`：仅串联 `GSCViTDSSRSpatialWrapper` 增加 `spatial_residual_gate=False`，显式开启才注册卷积。关闭直接保留原空间输出。
- `models/__init__.py`：最小增加参数传递；非串联模型显式要求 gate 时抛出错误，防止静默无效。
- `diagnostics/honghu_single_seed_20261018/diagnostic_train.py`：只准备未来 `--spatial-residual-gate`，默认关闭，记录配置并传入工厂；非串联模型显式开启提前报错。训练流程、loss、learning rate、batch、epochs、seed、scheduler 均未改。

新增 `verify_spatial_residual_gate.py` 和本目录下的配置、冻结参考、审计工具及证据文件。具体 SHA256 见 `manifest.json` 的 `artifact_sha256`；三个实际加载的模型源码路径和哈希同时记录在 `verification.json.runtime_sources`。`manifest.json` 不对自身作循环哈希。

公式严格为：

```text
R_spatial = F_spatial - F_spectral
gate_input = concat(F_spectral, R_spatial, beta, dim=1)
gate_logits = Conv2d(129, 1, kernel_size=1)(gate_input)
g = 2 * sigmoid(gate_logits)
F_out = F_spectral + g * R_spatial
```

`F_spectral/F_spatial/R_spatial/F_out=[B,64,9,9]`，`beta/g=[B,1,9,9]`，`gate_input=[B,129,9,9]`。gate 位于光谱状态、空间状态完成原顺序恢复之后、原 GSSA 之前；没有增加归一化、loss、新状态分支或一般 State Fusion。

开启时新增 130 个参数，权重与 bias 全零初始化：

| state_dict 键 | 形状 | 数量 |
| --- | --- | ---: |
| `spatial_residual_gate_conv.weight` | `[1,129,1,1]` | 129 |
| `spatial_residual_gate_conv.bias` | `[1]` | 1 |

关闭时不创建这两个键，不改变旧键及形状，也不额外消耗卷积初始化随机数。默认 `spatial_state_strength=0.5`。工厂原有 `spatial_state_enabled=False` 与串联 wrapper 原有 `True` 均保留；开启 gate 不隐式开启空间状态，合成测试显式传入 `spatial_state_enabled=True`。

## 实际命令与环境

所有终端均使用 Windows CMD。工作目录 `D:\HSI_Baseline`，未创建分支或 worktree。

| 实际命令 | 退出码 | 结果 |
| --- | ---: | --- |
| `python diagnostics\spatial_residual_gate_20261019\audit.py before` | 0 | 修改前参考与 3,226 个现有文件的 SHA256 清单 |
| `python diagnostics\spatial_residual_gate_20261019\run_checks.py` | 1 | 系统 Python 3.13.6 无 torch；测试主体未执行 |
| `C:\Users\PC\Anaconda3\python.exe diagnostics\spatial_residual_gate_20261019\run_checks.py` | 0 | 合成检查和 `git diff --check` 均为 0 |
| `C:\Users\PC\Anaconda3\python.exe verify_spatial_residual_gate.py` | 0 | 上一行子进程实际执行的检查命令 |
| `git diff --check` | 0 | 上一行运行器执行；仅有 Git 行尾提示 |
| `python diagnostics\spatial_residual_gate_20261019\audit.py after` | 0 | 3,223 个纳入审计的其余旧文件哈希保持一致，无越界改动 |

成功运行环境：Python **3.13.9 (Anaconda)**，PyTorch **2.13.0+cpu**，CPU，float32，1 个 PyTorch 计算线程。未下载或安装依赖。`command_history.json` 保留首次环境失败与后续成功的真实 stdout/stderr/退出码；`commands.json` 为最近一轮结果。

前置只读检查使用 `git status --short`、`git branch --show-current`、`git rev-parse HEAD`、`rg`、`dir`、`type`、`more` 和 `where`。若干包含引号的 `rg`/`python -c` 命令遭 CMD 参数解析错误（组合命令退出码 1，部分被后续成功命令掩盖为 0）；不计为通过。随后用脚本避免引号问题。没有把这些失败解释为代码问题。

起始分支：`codex/spatial025-adaptation`。提交：`f6cbfd03e965840e8838a7fbde46f1479cec3946`。起始工作树已 dirty，包括 `.gitignore`、`feature_diagnosis.py` 的修改以及多个未跟踪文件；完整起始状态在 `before.json`，最终状态在 `manifest.json`。没有覆盖或回滚这些内容。未发现适用的 `AGENTS.md`；已读取项目 README 及相关当前实现说明。

## 逐项验收与数值

预先在脚本中声明、成功运行前固定的 float32 容差：`atol=1e-6, rtol=1e-5`。以下是数值等价性检查，不推广为逐 bit 等价保证。

| 验收项 | 结果与关键输出 |
| --- | --- |
| 1. 默认关闭与独立旧参考 | PASS。冻结的是修改前完整 `gscvit.py`，SHA256 `499b6e6f4629df2ac0b4d63c7bae310f90adfd2ae5cd8307272913e798566690`，单独加载旧类；路由源码哈希保持不变。串联、TSSR、DSSR、空间单路及 GSCViT 在 B=1/2、eval 与 train-mode 合成前向中 `max_abs_diff=0`。train-mode 只检查 dropout/BN 兼容，不是训练。串联的空间状态开启/关闭都覆盖；默认初始化 state_dict 数值也一致。 |
| 2. 零初始化与接口 | PASS。B=1/2 均 `g_min=g_max=1`，所有预期形状正确。直接 route 的 F_out 与 F_spatial/旧参考 `max_abs_diff=0`；完整分类器最大差值 B=1 为 `1.9371509552e-7`，B=2 为 `1.1920928955e-7`，均在容差内。不声称逐 bit 等价。 |
| 3. 范围、自适应、样本隔离 | PASS。受控非零卷积同时连接光谱、residual、beta 三块，空间/样本变化得到 `g∈[0.6200510263,1.7615940571]`；单样本与合批 gate/output 差值均 0，改变另一样本差值 0，交换样本对应交换输出。全串联 route 单独/合批差值 `2.3841857910e-7`，改变另一样本对本样本差值 0。仅改共享 bias 得到统一 gate `1.3363754749`，未作为空间自适应证据；极端 logits 饱和值为 0、2，有限。 |
| 4. 梯度与权重不变 | PASS。B=1/2、零权重/受控非零权重均覆盖完整合成前向 backward；输入、beta、两个 gate 参数及全部 13 个空间状态参数都有有限梯度；预期 gate 梯度非零。逐参数记录见 verification.json。backward 前后所有模型参数完全不变，未创建优化器或调用 optimizer.step。 |
| 5. beta、排列、ordered_gain | PASS。gate 要求 `[B,1,9,9]`，错误维度/批大小/通道数拒绝；既有 SpatialStateRouter 的 3D/4D beta 接口仍一致。光谱 8 元、空间 81 元 permutation/inverse 均完整，两种复合顺序均为恒等。复用既有 ordered_gain 常量 residual 回归，并补充 B=1/2 生产尺寸独立公式检查，最大差值 `2.3841857910e-7`；源码未改。 |
| 6. checkpoint | **PASS（本轮补齐）**。指定 exec_03 epoch 192 实际 checkpoint 的 106 个键，分别 strict=True 加载至冻结旧实现和新 gate=False 模型，无缺失/多余键。B=1/2 eval 合成输出最大差值均 0，文件前后哈希一致。初轮内存结构测试及 gate=True 缺少两个 gate 键的预期失败结果继续保留，不能把旧 checkpoint 称为包含 gate。 |
| 7. 静态检查及训练参数传递 | PASS。授权源码用 compile 检查；AST 仅提取 argparse 声明与 build_model 调用，测试 CLI 缺省 False、显式 True、0.5 强度及参数传递。未导入/运行训练入口。非串联 gate opt-in 拒绝。Git diff 检查通过。 |
| 范围与资产保护 | PASS。3,226 个纳入审计的旧文件中，仅上述三份授权源码改变，另 3,223 个 SHA256 完全一致，无越界新增/修改/删除；新增内容限授权测试脚本与本专用目录。未修改 dynamic_router、main_train、main_test、旧 checkpoint/split/PCA/validation batch/结果文件。审计排除 .git、__pycache__、.idea、.vscode；专用目录独立列入 artifact_sha256。 |

零初始化时卷积权重 W=0，因此通过 logits 对 gate_input 的附加梯度为零，这是预期结果。空间状态主路径仍传梯度，gate 权重与 bias 可收到非零梯度。把 F_spectral、F_spatial、beta 当独立叶子做直接验证时，`dF_out/dF_spectral=0`、`dF_out/dF_spatial=I`、`dF_out/dbeta=0`；完整网络中的上游参数仍可经空间状态路径得到梯度。测试没有要求所有参数或所有梯度元素都非零。

## 限制、边界及交接

初轮真实 checkpoint 限制已由本轮指定文件补验解除，结论仅覆盖该 SHA256 的 checkpoint 与上述 CPU 合成输入，不代表真实数据精度或训练收益。交用户转主线程审核，不自动推进训练。

没有训练/微调、test 集评估、真实数据推理、AutoDL 操作、新 split、PCA、实验 checkpoint 写入、gate 结构/超参扫描、一般 State Fusion 或 Git 写操作。所有 forward/backward 均为本地合成张量；受控权重只用于正确性 fixture，不构成优化训练。资产审计只做原字节哈希比对，没有重新挖掘历史实验内容或证据。

建议提交标题：`feat: add opt-in adaptive spatial residual gate for serial DSSR`

建议用户自行精确暂存下列文件，不执行 `git add .`，不包含本任务前已有的 dirty/untracked 内容：

```text
compare/gscvit.py
models/__init__.py
diagnostics/honghu_single_seed_20261018/diagnostic_train.py
verify_spatial_residual_gate.py
diagnostics/spatial_residual_gate_20261019/README.md
diagnostics/spatial_residual_gate_20261019/gate_config.json
diagnostics/spatial_residual_gate_20261019/reference_gscvit_before.py
diagnostics/spatial_residual_gate_20261019/audit.py
diagnostics/spatial_residual_gate_20261019/run_checks.py
diagnostics/spatial_residual_gate_20261019/before.json
diagnostics/spatial_residual_gate_20261019/verification.json
diagnostics/spatial_residual_gate_20261019/commands.json
diagnostics/spatial_residual_gate_20261019/command_history.json
diagnostics/spatial_residual_gate_20261019/manifest.json
diagnostics/spatial_residual_gate_20261019/verify_checkpoint.py
diagnostics/spatial_residual_gate_20261019/checkpoint_verification.json
```

以上只是建议文件范围，没有执行 Git 暂存、提交或推送，也没有自动向其他线程发消息。
