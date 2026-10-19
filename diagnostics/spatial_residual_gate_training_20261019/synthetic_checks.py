"""Synthetic CPU/CUDA checks only. No trainer.main, real data, checkpoint or step."""
import argparse
import ast
import contextlib
import copy
import hashlib
import json
import io
from pathlib import Path
import random
import sys
import types
from unittest.mock import patch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from models import build_model
from diagnostics.honghu_single_seed_20261018 import diagnostic_train as trainer
from diagnostics.spatial_residual_gate_training_20261019 import alignment, runtime, compare_runs

ATOL, RTOL = 1e-6, 1e-5
OPTIONS = dict(spectral_groups=8, route_strength=0.5, route_temperature=1.0,
               router_variant='full', spatial_state_enabled=True, spatial_state_strength=0.5)

def forbidden(*args, **kwargs):
    raise AssertionError('Forbidden during preparation: data/checkpoint/optimizer step/training main')

def source_functions(path):
    return {n.name: n for n in ast.parse(Path(path).read_text(encoding='utf-8')).body if isinstance(n, ast.FunctionDef)}

def static_default_regression():
    before = source_functions(HERE / 'diagnostic_train_before.txt')
    after = source_functions(trainer.__file__)
    class StripGate(ast.NodeTransformer):
        def visit_If(self, node):
            text = ast.unparse(node.test)
            if text in ('args.spatial_residual_gate', "getattr(args, 'spatial_residual_gate', False)"):
                return None
            return self.generic_visit(node)
        def visit_Assign(self, node):
            if any(isinstance(t, ast.Name) and t.id == 'model_factory' for t in node.targets):
                return None
            return self.generic_visit(node)
        def visit_Name(self, node):
            if node.id == 'model_factory':
                node.id = 'build_model'
            return node
        def visit_Expr(self, node):
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == 'add_argument':
                if node.value.args and isinstance(node.value.args[0], ast.Constant) and node.value.args[0].value in ('--gate-approval', '--gate-launch-id'):
                    return None
            return self.generic_visit(node)
    for name, original in before.items():
        normalized = StripGate().visit(copy.deepcopy(after[name]))
        if ast.dump(original, include_attributes=False) != ast.dump(normalized, include_attributes=False):
            raise AssertionError('Default-path AST changed: ' + name)
    return {'all_original_functions_ast_equal_after_removing_gate_only_additions': True,
            'original_function_count': len(before), 'optimizer_scheduler_loss_dataloader_checkpoint_logic_unchanged': True}

def isolated_constructor(gate, device, generator, old=False):
    functions = source_functions(HERE / 'diagnostic_train_before.txt' if old else trainer.__file__)
    body = functions['main'].body
    start = next(i for i,n in enumerate(body) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == ('model' if old else 'model_factory') for t in n.targets))
    end = next(i for i,n in enumerate(body[start:], start) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'model' for t in n.targets))
    snippet = ast.Module(body=copy.deepcopy(body[start:end+1]), type_ignores=[])
    args = types.SimpleNamespace(model_name='gscvit_dssr_spatial', spatial_state_strength=0.5, spatial_residual_gate=gate)
    calls = []
    def factory(*a, **kw):
        calls.append({'args': list(a), 'kwargs': kw})
        return build_model(*a, **kw)
    ns = {'build_model': factory, 'args': args, 'device': torch.device(device), 'generator': generator, 'spatial_enabled': True}
    exec(compile(ast.fix_missing_locations(snippet), str(trainer.__file__), 'exec'), ns)
    return ns['model'], args, calls

def cli_checks():
    statements = source_functions(trainer.__file__)['main'].body
    stop = next(i for i,n in enumerate(statements) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'root' for t in n.targets))
    snippet = compile(ast.fix_missing_locations(ast.Module(body=copy.deepcopy(statements[:stop]), type_ignores=[])), '<isolated-cli>', 'exec')
    common = ['diagnostic_train.py', '--project-root', str(ROOT), '--assets-dir', 'synthetic-not-opened', '--output-dir', 'synthetic-not-created', '--model-name']
    outputs = []
    for gate in (False, True):
        ns = {'argparse': argparse, 'MODEL_FLAGS': trainer.MODEL_FLAGS}
        with patch.object(sys, 'argv', common + ['gscvit_dssr_spatial'] + (['--spatial-residual-gate'] if gate else [])):
            exec(snippet, ns)
        assert ns['args'].spatial_residual_gate is gate
        assert ns['args'].spatial_state_strength == 0.5
        outputs.append(vars(ns['args']))
    with patch.object(sys, 'argv', common + ['gscvit_dssr'] + ['--spatial-residual-gate']):
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                exec(snippet, {'argparse': argparse, 'MODEL_FLAGS': trainer.MODEL_FLAGS})
        except SystemExit as exc:
            assert exc.code == 2
        else:
            raise AssertionError('Nonserial gate CLI accepted')
    return {'default_disabled': True, 'explicit_enabled': True, 'nonserial_rejected': True, 'parsed': outputs, 'trainer_main_called': False}

def assert_state_exact(a, b):
    assert set(a) == set(b)
    for name in a:
        assert a[name].shape == b[name].shape and a[name].dtype == b[name].dtype and torch.equal(a[name], b[name]), name

def fixture_checks(output):
    fixtures = output / 'fixtures'
    fixtures.mkdir()
    # These are labelled arbitrary byte fixtures, not generated HongHu/PCA assets.
    names = ['split_indices.npz', 'sample_index_map.npz', 'pca_transform.npz', 'pca_scene.npz', 'validation_batch.npz', 'split_manifest.json']
    fake = fixtures / 'FAKE_BYTES_NOT_DATA'
    fake.mkdir()
    manifest = {}
    for name in names:
        path = fake / name
        path.write_bytes(('SYNTHETIC HASH FIXTURE ONLY: ' + name).encode())
        manifest[name] = {'bytes': path.stat().st_size, 'sha256': runtime.sha256(path)}
    runtime.write_new(fake / 'file_manifest.json', manifest)
    anchors = {n: runtime.sha256(fake / n) for n in ('file_manifest.json', 'split_manifest.json', 'sample_index_map.npz')}
    assert len(runtime.validate_assets(fake, anchors)) == 7
    (fake / 'pca_scene.npz').write_bytes(b'SYNTHETIC TAMPER')
    try:
        runtime.validate_assets(fake, anchors)
    except ValueError:
        pass
    else:
        raise AssertionError('Tampered asset accepted')
    try:
        runtime.assert_empty_output(fake)
    except FileExistsError:
        pass
    else:
        raise AssertionError('Non-empty output accepted')
    empty = fixtures / 'empty_target'
    assert runtime.assert_empty_output(empty) == empty.resolve() and not empty.exists()
    plan = runtime.read_json(runtime.PLAN_PATH)
    synthetic_assets = {n: dict(bytes=123, sha256=h) for n,h in plan['asset_anchors'].items()}
    def log(gate, delta=0.0):
        history = []
        best = 0.0
        for epoch in range(1, 201):
            row = {'epoch': epoch, 'train_loss': 2.0 / epoch, 'learning_rate': 3e-4 * 0.95 ** ((epoch - 1) / 10 + 1)}
            if epoch % 2 == 0:
                metrics = dict(plan['reference']['validation'])
                metrics['oa'] += delta if gate else 0
                if epoch != 192:
                    metrics['oa'] -= 1
                row['validation'] = metrics
                row['checkpoint_saved'] = metrics['oa'] >= best
                if row['checkpoint_saved']:
                    best = metrics['oa']
                    row['checkpoint_sha256'] = 'a' * 64 if gate else plan['reference']['sha256']
            history.append(row)
        return {'SYNTHETIC_FIXTURE_ONLY': True, 'model_name': plan['model_name'], 'dataset': 'HongHu', 'seed': 20261017,
                'split_seed': 20261018, 'spatial_state_enabled': True, 'run_status': 'completed',
                'config': dict(plan['config'], spatial_residual_gate=gate), 'nonfinite_events': [],
                'assets_file_manifest_sha256': plan['asset_anchors']['file_manifest.json'], 'assets': synthetic_assets,
                'history': history, 'best_checkpoint': 'checkpoint_epoch_192.pth',
                'best_checkpoint_sha256': 'a'*64 if gate else plan['reference']['sha256'],
                'selected_epoch': 192, 'selected_validation': history[191]['validation'],
                'initialization_alignment': {'shared_state_exact': True, 'cpu_rng_exact': True},
                'optimizer_membership': {'gate_parameters_exactly_once': True},
                'training_authorized': True if gate else None,
                'gate_preparation': {'approval': {'training_authorized': True}, 'SYNTHETIC_NOT_AUTHORIZATION': True}}
    old = log(False)
    cases = {}
    for name, delta in [('decrease', -0.1), ('tie', 0.0), ('increase', 0.1)]:
        new = log(True, delta)
        cases[name] = compare_runs.compare(old, new, fixture=True)
    bad = log(True)
    bad['history'].pop()
    try:
        compare_runs.compare(old, bad, fixture=True)
    except ValueError:
        pass
    else:
        raise AssertionError('Partial comparison log accepted')
    tied = log(True)
    tied['history'][193]['validation'] = dict(tied['history'][191]['validation'])
    tied['history'][193]['checkpoint_saved'] = True
    tied['history'][193]['checkpoint_sha256'] = 'b' * 64
    tied['best_checkpoint'] = 'checkpoint_epoch_194.pth'
    tied['best_checkpoint_sha256'] = 'b' * 64
    tied['selected_epoch'] = 194
    tied['selected_validation'] = tied['history'][193]['validation']
    assert compare_runs.validate_run(tied, plan, True)['epoch'] == 194
    wrong_tie = copy.deepcopy(tied)
    wrong_tie['history'][193]['checkpoint_saved'] = False
    try:
        compare_runs.validate_run(wrong_tie, plan, True)
    except ValueError:
        pass
    else:
        raise AssertionError('Earlier checkpoint on tie accepted')
    try:
        compare_runs.compare(old, log(True))
    except ValueError:
        pass
    else:
        raise AssertionError('Fixtures accepted as real evidence')
    runtime.write_new(fixtures / 'SYNTHETIC_reference_config.json', old)
    runtime.write_new(fixtures / 'SYNTHETIC_candidate_config.json', log(True, 0.1))
    runtime.write_new(fixtures / 'SYNTHETIC_comparison_cases.json', cases)
    try:
        runtime.validate_approval(HERE / 'approval_template.json', {'runtime_code_identity': {'files': {}, 'git': {'head': 'fixture'}}, 'plan_sha256': 'fixture', 'plan': plan})
    except PermissionError:
        pass
    else:
        raise AssertionError('Unapproved template accepted')
    args = types.SimpleNamespace(gate_launch_id=None)
    try:
        runtime.guard_training(trainer, args)
    except PermissionError:
        pass
    else:
        raise AssertionError('Direct gate invocation bypassed guard')
    # Exercise durable epoch writes and a health hook directly, without a training loop or optimizer.step.
    durable = fixtures / 'synthetic_logging'
    durable.mkdir()
    config = {'history': [], 'nonfinite_events': [], 'run_status': 'started'}
    row1 = {'epoch': 1, 'train_loss': 2.0, 'learning_rate': 0.000285}
    row2 = {'epoch': 2, 'train_loss': 1.0, 'learning_rate': 0.00028, 'validation': {'oa': 50.0, 'aa': 45.0, 'kappa': 40.0}}
    config['history'].append(row1)
    runtime.record_epoch(durable, row1, config, None, 0.1, trainer.write_run_config)
    config['history'].append(row2)
    runtime.record_epoch(durable, row2, config, None, 0.2, trainer.write_run_config)
    lines = (durable / 'epoch_history.jsonl').read_text().splitlines()
    assert [json.loads(s)['epoch'] for s in lines] == [1, 2]
    assert runtime.read_json(durable / 'train_config.json')['history'] == [row1, row2]
    class HookCapture:
        def register_step_post_hook(self, hook):
            self.hook = hook
            return hook
    toy = torch.nn.Linear(2, 1)
    hook_target = HookCapture()
    runtime.install_health_hook(hook_target, toy, durable, config, trainer.write_run_config)
    hook_target.hook()
    with torch.no_grad():
        toy.weight.fill_(float('nan'))
    try:
        hook_target.hook()
    except FloatingPointError:
        pass
    else:
        raise AssertionError('Nonfinite parameter was accepted')
    assert runtime.read_json(durable / 'train_config.json')['run_status'] == 'failed_nonfinite'
    bad_loss = {'epoch': 3, 'train_loss': float('inf'), 'learning_rate': 0.00027}
    config['history'].append(bad_loss)
    try:
        runtime.record_epoch(durable, bad_loss, config, None, 0.3, trainer.write_run_config)
    except FloatingPointError:
        pass
    else:
        raise AssertionError('Nonfinite accumulated loss accepted')
    assert runtime.read_json(durable / 'train_config.json')['nonfinite_events'][-1]['stage'] == 'epoch_loss'
    return {'hash_fixture_passed': True, 'tamper_rejected': True, 'nonempty_output_rejected': True,
            'empty_output_not_created': True, 'inert_approval_rejected': True, 'comparison_cases': ['decrease', 'tie', 'increase'],
            'partial_history_rejected': True, 'real_autodl_config_read': False, 'later_checkpoint_tie_rule': True,
            'incorrect_tie_rejected': True, 'fixtures_not_accepted_as_real': True, 'direct_training_guard_rejected': True,
            'two_epoch_records_persisted_without_training': True, 'nonfinite_hook_recorded_and_raised_without_step': True,
            'nonfinite_accumulated_loss_recorded_and_rejected': True}

def run(device, output):
    torch.set_num_threads(1)
    if device == 'cuda:0':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable; do not infer CUDA success from CPU')
        torch.cuda.init()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(20261017)
    np.random.seed(20261017)
    torch.manual_seed(20261017)
    pre = alignment.capture_rng()
    gen = torch.Generator().manual_seed(20261017)
    gen_pre = gen.get_state().clone()
    off, _, old_calls = isolated_constructor(False, device, gen, old=True)
    off_post = alignment.capture_rng()
    alignment.restore_rng(pre)
    default, _, default_calls = isolated_constructor(False, device, gen)
    assert_state_exact(off.state_dict(), default.state_dict())
    assert alignment.rng_equal(off_post, alignment.capture_rng())
    assert old_calls == default_calls and len(default_calls) == 1
    default_outputs = []
    for batch in (1, 2):
        original_copy, default_copy = copy.deepcopy(off).train(), copy.deepcopy(default).train()
        x = torch.randn(batch, 1, 15, 7, 7, generator=torch.Generator().manual_seed(20261017 + batch)).to(device)
        state = alignment.capture_rng()
        with torch.no_grad():
            original_logits = original_copy(x)
        original_after = alignment.capture_rng()
        alignment.restore_rng(state)
        with torch.no_grad():
            default_logits = default_copy(x)
        assert torch.equal(original_logits, default_logits)
        assert_state_exact(original_copy.state_dict(), default_copy.state_dict())
        assert alignment.rng_equal(original_after, alignment.capture_rng())
        default_outputs.append({'batch': batch, 'max_abs_difference': 0.0, 'buffers_exact': True, 'rng_exact': True})
        del original_copy, default_copy
    del default
    alignment.restore_rng(pre)
    on, args, on_calls = isolated_constructor(True, device, gen)
    assert [c['kwargs']['spatial_residual_gate'] for c in on_calls] == [False, True]
    assert all({k:c['kwargs'][k] for k in OPTIONS} == OPTIONS for c in on_calls)
    assert alignment.rng_equal(off_post, alignment.capture_rng())
    assert torch.equal(gen.get_state(), gen_pre)
    shared = {k:v for k,v in on.state_dict().items() if k not in alignment.GATE_KEYS}
    assert_state_exact(off.state_dict(), shared)
    assert set(on.state_dict()) - set(off.state_dict()) == alignment.GATE_KEYS
    # Separate loader generators; consume only synthetic indices and restore global RNG.
    samples = []
    for unused in (0, 1):
        g = torch.Generator()
        g.set_state(gen_pre)
        loader = DataLoader(TensorDataset(torch.arange(103)), batch_size=32, shuffle=True, generator=g, num_workers=0)
        samples.append(torch.cat([batch[0] for batch in loader]))
    assert torch.equal(samples[0], samples[1])
    assert torch.equal(gen.get_state(), gen_pre)
    assert alignment.rng_equal(off_post, alignment.capture_rng())
    cases = []
    for batch in (1, 2):
        a, b = copy.deepcopy(off).train(), copy.deepcopy(on).train()
        pa = {k:v.detach().clone() for k,v in a.named_parameters()}
        pb = {k:v.detach().clone() for k,v in b.named_parameters()}
        optimizer = torch.optim.AdamW(b.parameters(), lr=3e-4, weight_decay=0.005)
        membership = alignment.optimizer_membership(b, optimizer)
        input_gen = torch.Generator().manual_seed(20261017 + batch)
        x = torch.randn(batch, 1, 15, 7, 7, generator=input_gen).to(device)
        y = torch.arange(batch, device=device) % 22
        forward_rng = alignment.capture_rng()
        out_a = a(x)
        loss_a = torch.nn.functional.cross_entropy(out_a, y)
        loss_a.backward()
        rng_after_a = alignment.capture_rng()
        gate_outputs = []
        handle = b.spatial_residual_gate_conv.register_forward_hook(lambda m, i, out: gate_outputs.append((2 * torch.sigmoid(out)).detach()))
        alignment.restore_rng(forward_rng)
        out_b = b(x)
        loss_b = torch.nn.functional.cross_entropy(out_b, y)
        loss_b.backward()
        handle.remove()
        assert alignment.rng_equal(rng_after_a, alignment.capture_rng())
        torch.testing.assert_close(out_a, out_b, atol=ATOL, rtol=RTOL)
        assert torch.isfinite(loss_a) and torch.isfinite(loss_b)
        grads = {}
        for name, parameter in b.named_parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            assert torch.equal(parameter.detach(), pb[name]), name
            grads[name] = float(parameter.grad.abs().max())
        for name, parameter in a.named_parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            assert torch.equal(parameter.detach(), pa[name]), name
        for name, buffer in a.named_buffers():
            torch.testing.assert_close(buffer, dict(b.named_buffers())[name], atol=ATOL, rtol=RTOL)
        assert gate_outputs and all(torch.equal(g, torch.ones_like(g)) for g in gate_outputs)
        assert all(grads[k] > 0 for k in alignment.GATE_KEYS)
        cases.append({'batch': batch, 'train_mode': True, 'logits_shape': list(out_b.shape),
                      'max_abs_logits_difference': float((out_a-out_b).abs().max().detach()),
                      'max_fraction_of_allowed_error': float(((out_a-out_b).abs() / (ATOL + RTOL * out_b.abs())).max().detach()),
                      'loss_off': float(loss_a.detach()), 'loss_on': float(loss_b.detach()),
                      'same_dropout_rng_replayed': True, 'shared_buffers_close': True,
                      'all_parameter_gradients_finite': True, 'parameters_exactly_unchanged': True,
                      'gradient_max_abs': grads, 'gate_min': min(float(g.min()) for g in gate_outputs),
                      'gate_max': max(float(g.max()) for g in gate_outputs), 'optimizer_membership': membership})
        del a, b, optimizer
    identity = runtime.runtime_identity(trainer, ROOT)
    return {'status': 'passed', 'device': device, 'tolerance': {'atol': ATOL, 'rtol': RTOL},
            'training_authorized': False, 'training_loop_entered': False, 'optimizer_step_calls': 0,
            'real_data_forward_calls': 0, 'checkpoint_load_calls': 0,
            'environment': {'python': sys.version, 'executable': sys.executable, 'pytorch': torch.__version__,
                            'cuda_version': torch.version.cuda, 'device_name': torch.cuda.get_device_name(0) if device == 'cuda:0' else 'CPU', 'threads': torch.get_num_threads()},
            'cuda_verified': device == 'cuda:0', 'git_head': identity['git']['head'],
            'runtime_file_sha256': {p:r['sha256'] for p,r in identity['files'].items()},
            'runtime_code_identity': identity, 'default_regression': static_default_regression(),
            'default_train_mode_outputs': default_outputs,
            'cli': cli_checks(), 'alignment': args.gate_initialization,
            'default_constructor_calls': 1, 'gate_constructor_calls': 2,
            'synthetic_index_order_exact': True, 'index_order_sha256': hashlib.sha256(samples[0].numpy().tobytes()).hexdigest(),
            'cases': cases, 'fixtures': fixture_checks(output), 'actual_command': [sys.executable, *sys.argv]}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'cuda:0'], default='cpu')
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    output = runtime.assert_empty_output(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if args.device == 'cuda:0':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable')
        torch.cuda.init()
    outer = alignment.capture_rng()
    try:
        with patch.object(torch.optim.AdamW, 'step', forbidden), patch.object(np, 'load', forbidden), patch.object(torch, 'load', forbidden), patch.object(trainer, 'main', forbidden), patch.object(trainer, 'build_split_tensors', forbidden):
            result = run(args.device, output)
        runtime.write_new(output / 'synthetic_report.json', result)
        print(json.dumps({'status': result['status'], 'device': args.device, 'cases': [{k:r[k] for k in ('batch','max_abs_logits_difference','loss_on')} for r in result['cases']], 'report': str(output / 'synthetic_report.json')}))
    except BaseException as exc:
        runtime.write_new(output / 'failure.json', {'status': 'failed', 'exception': repr(exc), 'actual_command': [sys.executable, *sys.argv], 'training_started': False})
        raise
    finally:
        alignment.restore_rng(outer)

if __name__ == '__main__':
    main()
