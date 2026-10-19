"""Summarize final local evidence and hash this explicit delivery boundary only."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TRAINER = ROOT / 'diagnostics/honghu_single_seed_20261018/diagnostic_train.py'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def create(path, content):
    with Path(path).open('x', encoding='utf-8', newline=chr(10)) as f:
        f.write(content)

def main():
    evidence = HERE / 'verification_final'
    result = read(evidence / 'synthetic_cpu/synthetic_report.json')
    boundary = read(evidence / 'boundary_report.json')
    preflight = read(evidence / 'preflight_local.json')
    baseline = read(HERE / 'baseline.json')
    assert result['status'] == 'passed' and boundary['status'] == 'preparation_checks_passed'
    assert not boundary['future_target_created'] and not boundary['full_reference_config_read']
    for rel, record in boundary['protected_files'].items():
        assert sha(ROOT / rel) == record['before']
    for rel, expected in result['runtime_file_sha256'].items():
        assert sha(ROOT / rel) == expected, rel
    git = subprocess.run(['git', '--no-optional-locks', 'diff', '--', 'diagnostics/honghu_single_seed_20261018/diagnostic_train.py'], cwd=ROOT, capture_output=True, text=True, encoding='utf-8', errors='replace')
    assert git.returncode == 0
    create(HERE / 'trainer_diff.patch', git.stdout)
    summary = {
        'task_status': '准备完成待主线程批准', 'status': 'prepared_not_executed', 'training_authorized': False,
        'evidence_directory': 'verification_final', 'head': boundary['head'], 'branch': preflight['runtime_code_identity']['git']['branch'],
        'worktree_clean': False, 'remote_verified': False, 'future_commit': None, 'autodl_ready_claimed': False,
        'budget': preflight['plan']['budget'], 'protected_files_unchanged': boundary['protected_count'],
        'default_regression': result['default_regression'], 'default_train_outputs': result['default_train_mode_outputs'],
        'alignment': result['alignment'], 'synthetic_order_exact': result['synthetic_index_order_exact'],
        'synthetic_cases': [{k:r[k] for k in ('batch', 'max_abs_logits_difference', 'max_fraction_of_allowed_error', 'loss_off', 'loss_on', 'parameters_exactly_unchanged', 'all_parameter_gradients_finite', 'optimizer_membership')} for r in result['cases']],
        'fixtures': result['fixtures'], 'cuda_verified': False, 'real_data_forward_calls': 0, 'optimizer_step_calls': 0,
        'training_loop_entered': False, 'test_evaluated': False, 'assets_verified_count': len(preflight['assets']),
        'exact_local_reference_config_exists': boundary['exact_local_reference_config_exists'], 'full_reference_config_read': False,
        'actual_commands': boundary['commands'], 'trainer_before_sha256': baseline['trainer_before_sha256'], 'trainer_after_sha256': sha(TRAINER),
        'actual_finalize_command': [sys.executable, *sys.argv],
        'prior_evidence': {'checks_cpu_01': 'numerical check passed; JSON newline serializer defect; superseded, raw files retained',
                           'preflight_local_01.json': 'hash checks passed; JSON newline serializer defect; superseded, raw file retained',
                           'verification_02': 'passed, before direct RNG equality and additional regression checks',
                           'verification_03': 'passed, before accumulated-loss check and device identity extension'},
        'future_execution': 'User must review, commit/synchronize, verify CUDA and identity, then provide explicit separate approval. No execution in this task.'}
    create(HERE / 'delivery_summary.json', json.dumps(summary, ensure_ascii=False, indent=2) + chr(10))
    a = result['alignment']
    rows = []
    for r in result['cases']:
        grads = r['gradient_max_abs']
        rows.append(f"| {r['batch']} | {r['max_abs_logits_difference']:.17g} | {r['max_fraction_of_allowed_error']:.17g} | {r['loss_on']:.17g} | {grads['spatial_residual_gate_conv.weight']:.17g} | {grads['spatial_residual_gate_conv.bias']:.17g} |")
    table = chr(10).join(rows)
    commands = chr(10).join('- 命令（argv 精确数组）：' + json.dumps(r['actual_command'], ensure_ascii=False) + f"；退出码 {r['exit_code']}（预期 {r['expected_exit_code']}）。" for r in boundary['commands'])
    core = chr(10).join(f"| {rel} | {value} |" for rel, value in result['runtime_file_sha256'].items())
    report = f'''# 最终交付：gate 单模型训练方案与运行准备

2026-10-19（Asia/Shanghai）。**准备完成待主线程批准；training_authorized=false；status=prepared_not_executed。** 本结果不构成训练授权，不是导师新阶段编号。执行分支在此停止，由用户转交主线程。

## 实际身份与边界

- 实际开始/结束 HEAD：{boundary['head']}；分支 codex/spatial025-adaptation。工作树不干净，原有无关 dirty/untracked 保留；未独立确认远程状态。
- 新增仅在 diagnostics/spatial_residual_gate_training_20261019/；唯一旧文件修改是 diagnostics/honghu_single_seed_20261018/diagnostic_train.py。未扩大代码范围。
- 定向父路径检查未发现适用 AGENTS.md。没有 fetch/pull、分支/worktree、Git add/commit/push；没有 AutoDL、训练、真实样本 forward、validation/test、PCA/split 生成、checkpoint 加载或 optimizer.step。
- baseline.json 与 verification_final/boundary_report.json 比对 {boundary['protected_count']} 个受保护文件 SHA256 全部一致：核心模型、main_train/main_test、已验收 gate 目录文件、原配置/manifest/README/共享资产以及两份既有 dirty 文件。
- 本地精确参照路径 train_config.json 存在：{boundary['exact_local_reference_config_exists']}；未读 AutoDL 全量记录。正式结果比较尚未进行，只测标记清楚的合成 fixture。

## 实现与默认路径回归

初始化采用保存/恢复同一构建前状态、隔离 gate-off 内存参照、恢复真实参照构建后 RNG 的单一方法。共享 tensor 逐键/shape/dtype/exact 数值一致：{a['shared_initial_state']['tensor_count']} 个 state_dict tensor，共 {a['shared_initial_state']['numel']} 个元素（含 buffer）；只增加 weight [1,129,1,1] 和 bias [1]，合计 130，全零，初始 g=1。

共享初始摘要 SHA256：{a['shared_initial_state']['sha256']}。

gate-off 构建后 CPU RNG：{a['reference_rng_after']['cpu']}。

未对齐 gate-on 构建后 CPU RNG：{a['raw_gate_rng_after']['cpu']}。

对齐 gate-on 构建后 CPU RNG：{a['aligned_rng_after']['cpu']}。使用 torch.equal 对 RNG tensor exact 检查；摘要用于追溯。Python/NumPy RNG 和 DataLoader 专用 generator 未被构造诊断消费；同 generator 状态的 103 个模拟索引序列 exact，相应 SHA256 为 {result['index_order_sha256']}。没有任意重新 seed 掩盖构造差异。

旧 trainer 的 11 个原函数在去除新增 gate-only guard/logging 和 factory 别名后 AST 相同；保留原数据准备、DataLoader、loss、AdamW、LambdaLR、epoch 次数、validation 频率及 >= checkpoint 选择。旧/新 gate=False 实际构造都只调用 factory 一次；初始化、构建后 RNG exact；B=1/2 train-mode 输出最大差 0，buffer 与 forward 后 RNG exact。

新增 gate-only 行为：显式授权/源码资产身份 guard（读数据之前）、对齐 factory、参数组唯一性验证、运行身份/配置/摘要记录、逐 epoch JSONL/配置持久化、非有限参数/buffer 与累计 loss 检测。完整旧训练脚本 diff 见 trainer_diff.patch。旧脚本原 SHA256={baseline['trainer_before_sha256']}；修改后={sha(TRAINER)}。

## 实际合成数值

环境：Python {result['environment']['python']}；PyTorch {result['environment']['pytorch']}；CPU float32，单线程。CUDA **未验证**。独立副本 train-mode 比较前分别恢复同一 RNG，B=1/2 检查初始 dropout/BN 路径；所有共享和 gate 参数梯度 finite，gate 梯度非零，参数在 backward 前后 exact 不变。AdamW 1 个参数组、90 个参数 tensor，每个参数且每个 gate 参数恰好一次，lr=3e-4、weight_decay=0.005；step 调用 0。

| B | logits 最大绝对差 | 允许逐元素误差最大占比 | gate loss | gate weight 梯度最大绝对值 | gate bias 梯度最大绝对值 |
|---|---|---|---|---|---|
{table}

容差预先固定 atol=1e-6、rtol=1e-5；使用 |a-b| <= atol + rtol*|b|，两例全部通过。B=1 最大绝对差高于 atol 但仍在组合误差界内，没有放宽容差。输出 [B,22]，g_min=g_max=1。初始化/RNG exact 与 forward 近似等价分别判断。

## 预检与证据

- 7 个已知共享资产全部哈希/字节大小匹配；三条用户给定锚定 SHA256 也匹配；剩余值逐项来自原 file_manifest.json，不猜测。
- 空目录只读预检不创建训练目标；非空目录拒绝 exit=1；错误 commit 拒绝 exit=1；实际源码导入路径/哈希、Git 状态、完整计划、Python/PyTorch/CUDA 记录齐全。
- 合成 fixture 验证资产篡改拒绝、inert approval 拒绝、直接 gate 训练 guard 拒绝、OA 下降/持平/上升、平局较晚记录选择、错误 tie 和残缺曲线拒绝、fixture 不可冒充真实结果。
- 两条虚拟 epoch 记录成功持久化；手动调用健康 hook（没有 step）验证 NaN 参数和无限累计 loss 均停止并记录。没有通过循环训练生成这些记录。
- 新脚本在内存 compile；新脚本 AST 无 optimizer.step/torch.load/np.load；合成期同时把这些入口与 trainer.main/build_split_tensors 替换成拒绝函数。

实际最终复查命令为 C:\\Users\\PC\\Anaconda3\\python.exe -B diagnostics\\spatial_residual_gate_training_20261019\\run_local_checks.py --evidence-name verification_final，退出码 0。子命令与输出均原样保存在 verification_final/commands.json：

{commands}

早期一次补丁封装语法错误、一次换行补丁格式错误均发生在工具执行前，未写入；已修正。首轮 checks_cpu_01 与 preflight_local_01 数值/资产检查成功，但发现 JSON 尾换行序列化缺陷，原文件保留并标为 superseded；verification_02/03 通过后补了直接 RNG exact、默认输出回归、累计 loss 检查。最终依据只用 verification_final，全部 JSON 经解析校验。探查时默认 Python 缺 torch/einops，改用既有 Anaconda 环境；未安装依赖。

## 未来预算、配置和结论规则

拟议唯一候选 1 组/200 epochs；随机初始化 gscvit_dssr_spatial，gate=True、spatial enabled=True、strength=0.5、groups=8、route strength=0.5、temperature=1.0、full。HongHu，training_seed=20261017，split_seed=20261018，22 类，patch7，[B,1,15,7,7]，batch32，AdamW 3e-4/0.005。原 LambdaLR 公式原样保留，不因 poly 标签换公式。每轮 loss、隔轮 validation 100 条，OA 最大，>= 平局取后，无 early stopping。

PCA 只复用既有 full-scene 15 维/whiten/random_state=20261017 资产；train/validation=0.01/0.01，split/顺序/DataLoader 不变。整场景 PCA 属于 transductive 限制，不能声称 test 场景特征未参与预处理；test 标签/指标封存。旧 epoch192 仅比较/兼容参照，绝不是 warm-start。

主指标固定 validation OA；伴随 AA/Kappa 和全曲线。OA<=已验收参照：本次未显示 OA 收益，停止；AA/Kappa 改善也不转移主指标。OA>参照：仅单 seed 验证集探索性候选，不做跨 seed/因果/正式性能宣称，不设置事后显著性或新阈值。两种情况均不自动训练/test/扫描。gate 权重非零不证明有效适配；额外真实 forward 预算为 0，未提出 gate 分布采集。实现验收与训练收益验收严格分开。

## 用户未来操作（尚未执行）

详细参数表与本地 CMD 入口在 README.md。用户先转主线程审核本包；随后自行确定精确提交/同步 SHA 与 AutoDL 路径。新 SHA 未知，不作猜测。未来用户先跑 synthetic_checks.py --device cuda:0 到新的空目录，再跑 launch.py 默认只读预检；将报告/授权/训练输出放在代码工作树之外以保持用户同步后的工作树干净。

只有明确未来训练授权后另建审批文件，记录批准者/时间、真实 commit/路径、计划和源码哈希、CUDA 合成证据；原 template 保持 false。launch.py --execute 才可能执行一次，consumed 标记独占且失败也保留，不自动重跑。非空目录/错 commit/dirty/资产或配置不符/非有限/对齐不符立即停止交主线程。当前没有真实训练结果、未训练 checkpoint、CUDA 结论或 AutoDL 已就绪承诺。

用户未来完成唯一候选后，compare_runs.py 只读取原参照与候选的完整 JSON、核验两个 checkpoint 字节哈希，输出 comparison.json 和 curves.csv；不 forward。不重复固定 0.25，无新 seed/结构/强度、State Fusion 或历史搜证。

## 文件与 SHA256

精确全量清单为 FILE_SHA256.json（本目录全部文件及唯一修改的旧 trainer；只排除该清单自身避免自引用）。STAGING_PATHS.txt 给出建议逐文件暂存范围：可运行脚本、方案、最终报告/摘要/最终证据及原始 trainer 快照；不含资产、权重、fixture .npz 或无关变更，不建议整体 git add diagnostics。建议 commit 标题：Prepare one-run spatial gate training with RNG-aligned initialization。未执行 Git 写操作。

| 核心文件 | SHA256 |
|---|---|
{core}

## 验收结论

冻结预算/配置、初始化/RNG、公平默认路径、CLI 参数传递、optimizer 唯一性、输出/梯度、asset 与防覆盖 fixture、比较 fixture、日志及非有限处理、本地边界保护：全部通过。CUDA 和训练性能未验收；AutoDL 文件/新提交待用户实际确认。此处停止，准备完成不等于获准训练。
'''
    create(HERE / 'FINAL_REPORT.md', report)
    # Explicit staging recommendation; arbitrary byte fixtures and superseded evidence stay local.
    recommended = [p for p in HERE.iterdir() if p.is_file() and p.suffix in ('.py', '.json', '.md', '.txt', '.patch') and p.name not in ('preflight_local_01.json', 'FILE_SHA256.json', 'STAGING_PATHS.txt')]
    recommended += [evidence / x for x in ('commands.json', 'boundary_report.json', 'preflight_local.json', 'synthetic_cpu/synthetic_report.json')]
    recommended += [TRAINER, HERE / 'STAGING_PATHS.txt', HERE / 'FILE_SHA256.json']
    create(HERE / 'STAGING_PATHS.txt', chr(10).join(sorted(str(p.relative_to(ROOT)).replace(chr(92), '/') for p in recommended)) + chr(10))
    files = [p for p in HERE.rglob('*') if p.is_file() and p.name != 'FILE_SHA256.json'] + [TRAINER]
    inventory = {str(p.relative_to(ROOT)).replace(chr(92), '/'): {'sha256': sha(p), 'bytes': p.stat().st_size} for p in sorted(files)}
    create(HERE / 'FILE_SHA256.json', json.dumps({'scope': 'new delivery directory plus sole modified trainer; excludes this manifest itself', 'files': inventory}, ensure_ascii=False, indent=2) + chr(10))
    # All final evidence JSON must parse. Superseded malformed first outputs remain explicitly excluded.
    for path in evidence.rglob('*.json'):
        if 'FAKE_BYTES_NOT_DATA' not in path.parts:
            read(path)
    print(json.dumps({'status': 'prepared_not_executed', 'training_authorized': False, 'delivery_file_count': len(inventory),
                      'manifest_sha256': sha(HERE / 'FILE_SHA256.json'), 'report_sha256': sha(HERE / 'FINAL_REPORT.md'),
                      'cases': summary['synthetic_cases'], 'local_reference_config_exists': boundary['exact_local_reference_config_exists']}, ensure_ascii=False))

if __name__ == '__main__':
    main()
