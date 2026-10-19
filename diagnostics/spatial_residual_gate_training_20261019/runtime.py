"""Read-only preparation checks and gate-specific run records; never loads arrays."""
import hashlib
import json
import math
from pathlib import Path
import sys
import platform

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PLAN_PATH = HERE / 'plan.json'
NEW_REL = 'diagnostics/spatial_residual_gate_training_20261019/'
RUNTIME_REL = ('compare/gscvit.py', 'compare/dynamic_router.py', 'models/__init__.py',
               'diagnostics/honghu_single_seed_20261018/diagnostic_train.py',
               *(NEW_REL + x for x in ('alignment.py', 'runtime.py', 'launch.py', 'synthetic_checks.py', 'compare_runs.py', 'plan.json')))
ACTIVE_LAUNCH = None

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write_new(path, payload):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write(chr(10))

def assert_empty_output(path, assets=None):
    path = Path(path).resolve()
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError('Refusing non-empty output; preserve failed runs: ' + str(path))
    if assets is not None:
        assets = Path(assets).resolve()
        if path == assets or assets in path.parents or path in assets.parents:
            raise ValueError('Output must be disjoint from assets')
    return path

def validate_assets(assets, anchors):
    assets = Path(assets).resolve()
    for name, expected in anchors.items():
        if sha256(assets / name) != expected:
            raise ValueError('Pinned asset SHA256 mismatch: ' + name)
    manifest = read_json(assets / 'file_manifest.json')
    required = {'split_indices.npz', 'sample_index_map.npz', 'pca_transform.npz', 'pca_scene.npz', 'validation_batch.npz', 'split_manifest.json'}
    if set(manifest) != required:
        raise ValueError('Unexpected asset manifest entries')
    observed = {}
    for name, record in manifest.items():
        p = assets / name
        found = {'bytes': p.stat().st_size, 'sha256': sha256(p)}
        if found != record:
            raise ValueError('Asset hash/size mismatch: ' + name)
        observed[name] = found
    observed['file_manifest.json'] = {'bytes': (assets / 'file_manifest.json').stat().st_size, 'sha256': sha256(assets / 'file_manifest.json')}
    return observed

def validate_split_metadata(split, plan):
    expected = {'diagnostic_id': 'honghu_single_seed_20261018', 'dataset': 'HongHu',
                'training_seed': plan['training_seed'], 'split_seed': plan['split_seed'], 'train_ratio': 0.01, 'val_ratio': 0.01}
    for key, value in expected.items():
        if split.get(key) != value:
            raise ValueError('Split metadata mismatch: ' + key)
    for key, value in {'pca_components': 15, 'pca_whiten': True, 'pca_random_state': 20261017,
                       'pca_fit_scope': 'full_scene_before_patch_creation_and_split', 'patch_size': 7}.items():
        if split['preprocessing'].get(key) != value:
            raise ValueError('Preprocessing mismatch: ' + key)

def runtime_identity(trainer, root):
    import compare.dynamic_router
    import compare.gscvit
    import models
    from diagnostics.spatial_residual_gate_training_20261019 import alignment
    identity = trainer.collect_runtime_code_identity(root)
    if identity['collection_status'] != 'complete':
        raise RuntimeError('Incomplete runtime code/Git identity')
    for rel in RUNTIME_REL:
        path = ROOT / rel
        if path.resolve() != (Path(root) / rel).resolve():
            raise ValueError('Actual package root differs: ' + rel)
        identity['files'][rel] = {'path': str(path.resolve()), 'sha256': sha256(path), 'matches_project_root': True}
    for mod in (compare.dynamic_router, compare.gscvit, models, sys.modules[__name__]):
        if ROOT not in Path(mod.__file__).resolve().parents:
            raise ValueError('Imported module outside project: ' + mod.__name__)
    identity['actual_imports'] = {}
    for module, rel in ((alignment, NEW_REL + 'alignment.py'), (sys.modules[__name__], NEW_REL + 'runtime.py'),
                        (trainer, 'diagnostics/honghu_single_seed_20261018/diagnostic_train.py'),
                        (compare.gscvit, 'compare/gscvit.py'), (compare.dynamic_router, 'compare/dynamic_router.py'),
                        (models, 'models/__init__.py')):
        actual = Path(module.__file__).resolve()
        if actual != (ROOT / rel).resolve():
            raise ValueError('Unexpected imported source: ' + module.__name__)
        identity['actual_imports'][module.__name__] = {'path': str(actual), 'sha256': sha256(actual)}
    return identity

def preflight(trainer, root, assets, output, expected_commit):
    plan = read_json(PLAN_PATH)
    if plan['training_authorized'] is not False or plan['status'] != 'prepared_not_executed':
        raise ValueError('Do not change the frozen preparation plan to authorize training')
    output = assert_empty_output(output, assets)
    identity = runtime_identity(trainer, root)
    if identity['git']['head'] != expected_commit:
        raise ValueError('Actual commit differs from user-confirmed commit')
    records = validate_assets(assets, plan['asset_anchors'])
    validate_split_metadata(read_json(Path(assets) / 'split_manifest.json'), plan)
    import torch
    return {'status': 'prepared_not_executed', 'training_authorized': False, 'plan': plan,
            'plan_sha256': sha256(PLAN_PATH), 'runtime_code_identity': identity, 'assets': records,
            'output_dir': str(output), 'assets_dir': str(Path(assets).resolve()),
            'environment': {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
                            'pytorch': torch.__version__, 'cuda_version': torch.version.cuda,
                            'cuda_available': torch.cuda.is_available(),
                            'selected_device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
                            'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
                            'cuda_initialized': torch.cuda.is_initialized()},
            'array_or_checkpoint_loaded': False, 'forward_executed': False}

def validate_approval(approval_path, report):
    approval = read_json(approval_path)
    identity = report['runtime_code_identity']
    expected_files = {p: r['sha256'] for p, r in identity['files'].items()}
    checks = {'training_authorized': True, 'status': 'approved_for_one_run',
              'approved_commit': identity['git']['head'], 'plan_sha256': report['plan_sha256'],
              'budget': report['plan']['budget'], 'runtime_file_sha256': expected_files}
    for k, expected in checks.items():
        if approval.get(k) != expected:
            raise PermissionError('Missing/mismatched future approval: ' + k)
    if not approval.get('approved_by') or not approval.get('approved_at'):
        raise PermissionError('Future reviewer and approval date required')
    if not identity['git']['worktree_clean']:
        raise PermissionError('Execution requires synchronized clean worktree; preparation may remain dirty')
    for k in ('assets_dir', 'output_dir'):
        if not approval.get(k) or Path(approval[k]).resolve() != Path(report[k]).resolve():
            raise PermissionError('Approval path mismatch: ' + k)
    cuda_path = Path(approval.get('cuda_synthetic_report') or '')
    if not cuda_path.is_file() or sha256(cuda_path) != approval.get('cuda_synthetic_report_sha256'):
        raise PermissionError('Reviewed CUDA synthetic report required')
    cuda = read_json(cuda_path)
    if cuda.get('status') != 'passed' or cuda.get('device') != 'cuda:0' or cuda.get('training_loop_entered') is not False:
        raise PermissionError('CUDA synthetic validation has not passed')
    if cuda.get('cuda_verified') is not True or cuda.get('optimizer_step_calls') != 0 or cuda.get('real_data_forward_calls') != 0:
        raise PermissionError('CUDA report violates the synthetic-only preflight boundary')
    if cuda.get('runtime_file_sha256') != expected_files or cuda.get('git_head') != identity['git']['head']:
        raise PermissionError('CUDA synthetic evidence does not match runtime')
    import torch
    if not torch.cuda.is_available() or cuda['environment']['pytorch'] != torch.__version__ or cuda['environment']['cuda_version'] != torch.version.cuda or cuda['environment']['device_name'] != torch.cuda.get_device_name(0):
        raise PermissionError('CUDA report environment does not match execution environment')
    return approval

def guard_training(trainer, args):
    if ACTIVE_LAUNCH is None or args.gate_launch_id != ACTIVE_LAUNCH['id']:
        raise PermissionError('Gate training requires reviewed launch.py --execute entry')
    if args.model_name != 'gscvit_dssr_spatial' or args.spatial_state_strength != 0.5 or not args.spatial_residual_gate:
        raise ValueError('Gate run does not match frozen model/strength')
    approval = read_json(args.gate_approval)
    report = preflight(trainer, args.project_root, args.assets_dir, args.output_dir, approval.get('approved_commit'))
    validate_approval(args.gate_approval, report)
    return {'launch_id': args.gate_launch_id, 'approval_path': str(Path(args.gate_approval).resolve()),
            'approval_sha256': sha256(args.gate_approval), 'approval': approval, 'preflight': report}

def start_record(output, context, command):
    write_new(Path(output) / 'gate_run_identity.json', {'gate_context': context, 'actual_command': command,
              'status': 'authorized_start_pre_data', 'training_authorized': True})

def install_health_hook(optimizer, model, output, run_config, write_config):
    def check_parameters(*unused):
        import torch
        bad = [name for name, p in model.named_parameters() if not torch.isfinite(p).all()]
        bad += [name for name, b in model.named_buffers() if not torch.isfinite(b).all()]
        if bad:
            run_config['run_status'] = 'failed_nonfinite'
            run_config['nonfinite_events'].append({'stage': 'post_optimizer', 'names': bad})
            write_config(output, run_config)
            raise FloatingPointError('Nonfinite parameter/buffer after optimizer; stopping')
    return optimizer.register_step_post_hook(check_parameters)

def record_epoch(output, record, run_config, best_checkpoint, elapsed, write_config):
    # Gate-only durability; optimizer/scheduler/loss/checkpoint rules untouched.
    if not math.isfinite(record['train_loss']):
        run_config['run_status'] = 'failed_nonfinite'
        run_config['nonfinite_events'].append({'epoch': record['epoch'], 'stage': 'epoch_loss', 'message': 'nonfinite accumulated epoch loss'})
        record['train_loss'] = None
        write_config(output, run_config)
        raise FloatingPointError('Nonfinite accumulated epoch loss')
    with (Path(output) / 'epoch_history.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(record, allow_nan=False) + chr(10))
        f.flush()
    run_config['best_checkpoint'] = best_checkpoint
    run_config['elapsed_seconds'] = elapsed
    if best_checkpoint:
        run_config['best_checkpoint_path'] = str((Path(output) / best_checkpoint).resolve())
        run_config['best_checkpoint_sha256'] = sha256(Path(output) / best_checkpoint)
    write_config(output, run_config)
