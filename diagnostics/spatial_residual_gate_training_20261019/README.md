# gate 单模型训练方案与运行准备

日期：2026-10-19（Asia/Shanghai）。任务状态：准备完成待主线程审核。不是导师新阶段编号。

**training_authorized=false；status=prepared_not_executed。准备检查通过不构成训练授权。**

## 范围与边界

本地开始 HEAD 为 d8962e0b2f2f3281878f08fb4906b075101ec713，分支 codex/spatial025-adaptation；工作树原本有无关 dirty/untracked，已保留。未 fetch/pull，未创建分支/worktree，未 Git add/commit/push，未独立确认远程状态。适用父目录及目标目录未发现 AGENTS.md。

只修改旧 diagnostic_train.py 的 gate 显式开启分支与运行记录；其余新增文件均在本目录。compare/gscvit.py、compare/dynamic_router.py、models/__init__.py、main_train.py、main_test.py、旧 gate 验收材料、原配置/manifest/README、共享资产均未修改。未重新检查真实 checkpoint strict 加载，沿用已验收结论；未做历史 split/PCA/运行身份搜索。

本地只做 CPU float32 合成构建、forward/backward、随机状态/CLI/预检/比较 fixture 检查。没有 optimizer.step，没有训练循环，没有真实数据前向、validation/test、新 split/PCA、真实 checkpoint 反序列化或 AutoDL 访问。资产只逐字节散列及读取原 JSON 元数据。

## 冻结的未来候选

唯一拟议预算为 **1 组 × 200 epochs = 200 epochs**，gscvit_dssr_spatial 从随机初始化训练。gate=True，spatial_state_enabled=True，spatial_state_strength=0.5；spectral_groups=8、route_strength=0.5、route_temperature=1.0、router_variant=full。不微调，不 warm-start，不重训 0.5 参照，不重复 0.25，不拓展一般 State Fusion。

HongHu，training_seed=20261017，split_seed=20261018，22 类；patch=7，输入 [B,1,15,7,7]。batch=32；AdamW lr=3e-4、weight_decay=0.005，所有参数同组、无 gate 专属 lr 或冻结 backbone。保持原 CrossEntropyLoss。

配置历史标签 scheduler=poly **仅是标签**。实际公式严格保留 LambdaLR(lambda epoch: 0.95 ** ((epoch - 1) / 10 + 1))，每个 epoch 末 scheduler.step；没有改成另一种调度。

PCA 复用原 full-scene 资产：15 维，whiten=True，random_state=20261017，禁止重新 fit/train-only。整场景特征参与过既有 PCA，这是 transductive 限制；不能声称 test 场景特征从未参与预处理。test 标签及指标仍封存。沿用 train/validation=0.01/0.01、原 split/样本顺序、原 DataLoader 专用 generator、num_workers=0、shuffle/drop_last/pin_memory 等设置。

每 epoch 持久化 train loss 与学习率；每 2 epochs validation，总计 100 条 OA/AA/Kappa；OA 最大 checkpoint，>= 平局选较晚记录。无 early stopping 或新 checkpoint 规则。gate 专用日志补充 epoch_history.jsonl、train_config.json 逐轮更新、最终 selected_epoch/全精度 selected_validation/checkpoint SHA256、真实命令/退出码。非有限 logits/loss/gradient/validation metric 的既有拦截保留；新增 gate-only 参数/buffer 与累计 epoch loss 拦截和记录。

## 初始化与随机状态

使用唯一方法 isolated_gate_off_reference_from_same_preconstruction_rng_restore_reference_poststate_v1：保存构建前 Python/NumPy/CPU 和已初始化 CUDA RNG；从该状态构建内存 gate-off 参照，保存其构建后状态；恢复同一构建前状态并构建 gate-on；逐键验证共享参数/buffer 的键/shape/dtype/数值 exact、130 个新增参数全零；最后恢复真正的 gate-off 构建后状态。没有借用训练 checkpoint、没有任意重置 seed。参照不 forward、不训练。

alignment.py 验证模型构建不消费 Python/NumPy/CUDA RNG 和传入的 DataLoader generator。gate=False 只多出一个函数别名赋值，不调用对齐、不多构建、不重置 seed。合成检查还从修改前训练脚本快照提取原构造表达式：gate-off 新旧只调用一次 factory，共享 state 与构建后 RNG exact；B=1/2 train-mode 输出、buffer/RNG exact。

gate-on/off 的 train-mode forward 比较使用独立模型副本，先保存同一个 RNG，分别恢复后 forward/backward；不是连续 forward 冒充同一 dropout 序列。buffer 变化以副本比对；未来训练模型未被合成测试使用。CPU float32 采用 atol=1e-6、rtol=1e-5，即逐元素 |a-b| <= 1e-6 + 1e-5*|b|；不能把最大绝对误差必须低于 atol 当作此准则。初始化 tensor/RNG 用 exact。所有参数梯度 finite，gate 梯度非零；未更新参数。

本地 CUDA 未验证，CPU 成功不代表 GPU 成功。synthetic_checks.py --device cuda:0 是用户未来单独运行的合成入口，同样不训练、不真实数据 forward。CUDA 失败就停止并保留证据，不放宽容差。

## 预先固定的判定

主指标是 validation OA，比较各自按原规则选中的最佳 checkpoint。参照仅用已验收 exec_03 gate-off strength=0.5 的 epoch192：OA=95.1383501422，AA=89.1850859033，Kappa=93.8338075617；checkpoint SHA256=606ef81f050cda8171dd0effe05f77ef6a24c12289942f5e076f253cf0183304。

OA 下降或持平：本次筛选未显示 OA 收益，停止；AA/Kappa 即使更高也如实伴随报告，不包装为 OA 假设验证成功。OA 提高：只记单 seed 验证集探索性候选，报告 AA/Kappa、曲线、数值健康与限制；不宣称跨 seed、因果或正式提升。微小提升不追加事后阈值或显著性包装。两种结果均不自动多 seed、test、结构/强度扫描或新训练。

Rp/Rs≈2.30 不等于残差过大有害；0.5→0.25 单次无收益不再重复。gate 权重非零不能证明有效按样本/位置适配。本计划不提出 gate 实际分布采集，额外真实前向预算=0；未来需要另行明确采样范围/前向次数并交主线程审核。

比较入口只读取完整 train_config.json 和 checkpoint 字节哈希，不加载模型或数据。要求 200 epoch/100 validation、冻结参数/资产一致、原调度曲线和 >= tie 规则、checkpoint 哈希匹配；生成 comparison.json 与 curves.csv。当前只用标记 SYNTHETIC 的 fixture 测试，无真实候选结果，尚未读取 AutoDL 原始完整记录。fixture 不可通过正式比较入口冒充结果。

## 用户未来操作步骤（全部尚未执行）

1. 将本目录的 FINAL_REPORT.md、FILE_SHA256.json、最终 verification_final 证据交主线程审核。审核通过也不自动训练。
2. 用户按精确暂存清单自行提交/同步，然后实际确认新 commit 和远端目录。这里不猜新 SHA，不声称 AutoDL 已就绪。不覆盖现有结果或失败目录。
3. 在用户确认的远端 Python 环境先运行合成 CUDA 检查，输出到一个新的空目录；再运行 launch.py 的默认只读预检，指定 actual --expected-commit。预检报告应放在拟议训练输出目录之外。
4. 主线程审核冻结计划、实际源码路径/哈希、commit/dirty、7 个资产身份、初始化/RNG/参数组证据、CUDA 环境和输出路径。执行保护要求用户同步后的工作树干净；原本本地 dirty 无需清除。将授权/预检/CUDA 报告及运行输出放在代码工作树之外，避免这些新文件造成 dirty。不要删除或覆盖既有无关文件来满足检查。
5. 只有收到用户明确未来训练授权后，另存 approval_template.json 为单独授权记录。template 本身保持 false；填 approved_by/approved_at、approved_commit、plan_sha256、完整 runtime_file_sha256、实际 assets/output 路径、CUDA 报告路径及 SHA256，status=approved_for_one_run、training_authorized=true。该文件是操作保护而非密码学授权签名。未授权不要填 true。
6. 用户未来用 launch.py --execute --approval <已审核文件> 执行一次。启动器对批准文件旁的 consumed 标记独占创建，失败也消耗此尝试，禁止自动重跑；非空 output 立即拒绝。直接运行 diagnostic_train.py 开启 gate 会在读数据前被拒绝。未通过检查就停止交主线程；不要更改代码绕过保护。
7. 未来若一次运行成功，用户提供既有 exec_03 和候选的完整 train_config.json 及两份 checkpoint 路径，调用 compare_runs.py；仅散列 checkpoint，读取日志作比较，无额外前向。主线程按上述 OA 规则决定，任务自身不延伸。

### 跨平台参数表

| 入口 | 参数 | 作用 |
|---|---|---|
| synthetic_checks.py | --device cpu 或 cuda:0；--output-dir 新空目录 | 合成构建/forward/backward/RNG/fixture，无训练 |
| launch.py | --project-root；--assets-dir；--output-dir；--expected-commit；--report 新文件 | 默认仅只读预检，不创建训练输出 |
| launch.py | 上述参数 + --execute --approval | 仅未来用户明确授权后允许一次 |
| compare_runs.py | --reference-config；--candidate-config；--reference-checkpoint；--candidate-checkpoint；--output-dir | 未来完整日志比较、哈希与曲线 |

已知远端资产路径仅为用户提供：/root/HSI_Baseline_honghu_diag_20261018/diagnostics/honghu_single_seed_20261018/assets；参照目录为同一旧根下 diagnostics/honghu_single_seed_20261018/runs_20261018_exec_03/gscvit_dssr_spatial/。本阶段没有访问它们。

### 本地 Windows CMD 复查入口（仅准备）

cd /d D:\HSI_Baseline

C:\Users\PC\Anaconda3\python.exe -B diagnostics\spatial_residual_gate_training_20261019\run_local_checks.py --evidence-name USER_CHOSEN_NEW_CHECK_DIRECTORY

该检查入口只运行 CPU 合成、只读资产预检和预期拒绝测试，绝不调用训练入口；目录名必须新建。此处不附远端 Linux 命令，避免混入本地 CMD 列表。

## 停止条件与限制

无明确训练授权，或 CUDA 未验证，或 commit/dirty/路径/源码/配置/资产不匹配，或任何初始化/RNG/finite 检查失败，或输出非空，立即停止并留存原结果。所有失败启动目录/consumed 标记保留，不自动清理重试训练。本地普通合成检查可另建目录修复复查。

不保证跨 PyTorch 版本/硬件得到相同随机初始化数值；公平对齐承诺是同运行条件下 gate-on/off exact，未来环境必须自行预检记录。没有可用于 warm-start 的本阶段模型 checkpoint。没有任何训练性能结论。

FILE_SHA256.json 是本目录所有交付文件及唯一旧文件修改的完整 SHA256 清单（仅排除自身以避免自引用）；FINAL_REPORT.md 含精确审查结论；STAGING_PATHS.txt 是明确暂存候选列表，不含资产/权重/无关变更。建议 commit 标题：Prepare one-run spatial gate training with RNG-aligned initialization。所有 Git 写操作由用户未来自行决定，本阶段未执行。
