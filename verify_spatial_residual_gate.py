"""CPU synthetic correctness only; no datasets, checkpoint IO, or optimizer.

Run from the repository root: python verify_spatial_residual_gate.py
Tolerance declared before execution: float32 atol=1e-6, rtol=1e-5.
The frozen pre-edit source is an independent regression oracle. It is not a
historical experiment checkpoint and must not be reported as one.
"""
import ast
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import traceback

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
sys.dont_write_bytecode = True
import torch
from models import build_model
from compare import gscvit, dynamic_router

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'diagnostics/spatial_residual_gate_20261019'
ATOL, RTOL = 1e-6, 1e-5
REPORT = {'tolerance': {'dtype': 'float32', 'atol': ATOL, 'rtol': RTOL}, 'checks': []}
NEW_KEYS = {'spatial_residual_gate_conv.weight': [1, 129, 1, 1],
            'spatial_residual_gate_conv.bias': [1]}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def record(name, **details):
    REPORT['checks'].append({'name': name, 'status': 'PASS', **details})
    print('[PASS]', name, json.dumps(details, sort_keys=True))


def finite(tensor):
    require(tensor is not None and torch.isfinite(tensor).all().item(), 'missing/nonfinite tensor')


def close(actual, expected):
    finite(actual)
    finite(expected)
    require(actual.shape == expected.shape, 'shape mismatch')
    difference = (actual - expected).abs().max().item()
    require(torch.allclose(actual, expected, atol=ATOL, rtol=RTOL), f'difference {difference}')
    return difference


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_reference():
    before = json.loads((OUT / 'before.json').read_text(encoding='utf-8'))
    path = OUT / 'reference_gscvit_before.py'
    require(sha(path) == before['reference_sha256'] == before['files']['compare/gscvit.py']['sha256'],
            'frozen reference no longer matches captured source')
    require(sha(ROOT / 'compare/dynamic_router.py') == before['files']['compare/dynamic_router.py']['sha256'],
            'dynamic_router changed')
    spec = importlib.util.spec_from_file_location('gate_pre_edit_reference', path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    record('independent_reference', source=str(path), sha256=sha(path),
           unchanged_router_sha256=sha(ROOT / 'compare/dynamic_router.py'))
    return reference


def permutation_check(router, count):
    p, inv = router.last_permutation, router.last_inverse_permutation
    require(p is not None and inv is not None, 'missing permutation')
    expected = torch.arange(count).expand_as(p)
    require(p.shape == inv.shape and p.shape[1] == count, 'permutation shape')
    require(torch.equal(p.sort(dim=1).values, expected), 'permutation incomplete')
    require(torch.equal(inv.sort(dim=1).values, expected), 'inverse incomplete')
    require(torch.equal(p.gather(1, inv), expected), 'p composed with inverse')
    require(torch.equal(inv.gather(1, p), expected), 'inverse composed with p')


def serial(gate=False, spatial=True):
    return build_model('gscvit_dssr_spatial', 15, 22, 7,
                       spatial_state_enabled=spatial, spatial_residual_gate=gate).eval()


def legacy_regressions(reference):
    cases = [('gscvit', 'GSCViTWrapper', {}),
             ('gscvit_tssr', 'GSCViTTSSRWrapper', {}),
             ('gscvit_dssr', 'GSCViTDSSRWrapper', {}),
             ('gscvit_spatial', 'GSCViTSpatialWrapper', {'spatial_state_enabled': True}),
             ('gscvit_spatial', 'GSCViTSpatialWrapper', {'spatial_state_enabled': False}),
             ('gscvit_dssr_spatial', 'GSCViTDSSRSpatialWrapper', {'spatial_state_enabled': True}),
             ('gscvit_dssr_spatial', 'GSCViTDSSRSpatialWrapper', {'spatial_state_enabled': False})]
    for name, cls, options in cases:
        torch.manual_seed(612)
        old = getattr(reference, cls)(15, 22, 7, **options)
        torch.manual_seed(612)
        new = build_model(name, 15, 22, 7, **options)
        old_state, new_state = old.state_dict(), new.state_dict()
        require(list(old_state) == list(new_state), 'default state keys changed')
        require(all(old_state[k].shape == new_state[k].shape for k in old_state), 'state shape changed')
        require(all(torch.equal(old_state[k], new_state[k]) for k in old_state), 'default initialization changed')
        loaded = new.load_state_dict(old_state, strict=True)
        require(not loaded.missing_keys and not loaded.unexpected_keys, 'strict load incomplete')
        require(not any('spatial_residual_gate' in k for k in new_state), 'gate parameters created by default')
        differences = {}
        for batch in (1, 2):
            x = torch.randn(batch, 1, 15, 7, 7)
            for train_mode in (False, True):
                old.train(train_mode)
                new.train(train_mode)
                with torch.no_grad():
                    torch.manual_seed(613)
                    expected = old(x)
                    torch.manual_seed(613)
                    actual = new(x)
                differences[f'B{batch}_train{train_mode}'] = close(actual, expected)
            if name == 'gscvit_dssr_spatial':
                features = torch.randn(batch, 64, 9, 9)
                with torch.no_grad():
                    old_out = old._route(features)
                    new_out = new._route(features)
                differences[f'B{batch}_route'] = max(close(a, b) for a, b in zip(new_out, old_out))
                if options['spatial_state_enabled']:
                    for router in (new.state_router, new.spatial_state_router):
                        permutation_check(router, 8 if router is new.state_router else 81)
        record('legacy_' + name, options=options, state_key_count=len(old_state), strict=True,
               max_abs_diff=differences)


def capture(model):
    values = {}
    def hook(name):
        def save(module, args, output):
            values[name] = output
            if name == 'logits':
                values['gate_input'] = args[0]
                if args[0].requires_grad:
                    args[0].retain_grad()
            if name == 'spectral':
                values['beta'] = args[2]
                if args[2].requires_grad:
                    args[2].retain_grad()
        return save
    handles = [model.state_router.register_forward_hook(hook('spectral')),
               model.spatial_state_router.register_forward_hook(hook('spatial')),
               model.spatial_residual_gate_conv.register_forward_hook(hook('logits'))]
    return values, handles


def zero_and_checkpoint(reference):
    old = reference.GSCViTDSSRSpatialWrapper(15, 22, 7).eval()
    enabled = serial(True)
    old_state = old.state_dict()
    extra = {k: list(v.shape) for k, v in enabled.state_dict().items() if k not in old_state}
    require(extra == NEW_KEYS, 'unexpected new parameters')
    require(sum(enabled.state_dict()[k].numel() for k in extra) == 130, 'parameter count')
    try:
        enabled.load_state_dict(old_state, strict=True)
    except RuntimeError as exc:
        error = str(exc)
        require(all(k in error for k in NEW_KEYS), 'strict failure did not identify gate keys')
    else:
        raise AssertionError('old state silently accepted with gate enabled')
    # Explicit initialization, not a claim that the old state contains gate weights.
    expanded = dict(old_state)
    expanded.update({k: enabled.state_dict()[k].clone() for k in NEW_KEYS})
    enabled.load_state_dict(expanded, strict=True)
    require(all(torch.count_nonzero(enabled.state_dict()[k]).item() == 0 for k in NEW_KEYS), 'nonzero initialization')
    record('checkpoint_structure', new_parameters=extra, new_parameter_count=130,
           old_into_enabled_strict_error=error, explicit_zero_extension_strict=True,
           actual_historical_checkpoint_loaded=False,
           limitation='Pre-edit in-memory state_dict only; does not replace historical checkpoint loading.')
    for batch in (1, 2):
        values, handles = capture(enabled)
        with torch.no_grad():
            f = torch.randn(batch, 64, 9, 9)
            result, alpha, beta = enabled._route(f)
            expected, _, _ = old._route(f)
            g = 2 * torch.sigmoid(values['logits'])
            require(torch.equal(g, torch.ones_like(g)), 'initial g is not exactly one')
            require(g.shape == beta.shape == (batch, 1, 9, 9), 'beta/g shape')
            require(values['gate_input'].shape == (batch, 129, 9, 9), 'gate input shape')
            require(result.shape == (batch, 64, 9, 9), 'output shape')
            cat = torch.cat((values['spectral'], values['spatial'] - values['spectral'], beta), dim=1)
            require(torch.equal(cat, values['gate_input']), 'gate concatenation content/order')
            diff_spatial = close(result, values['spatial'])
            diff_reference = close(result, expected)
            for router, count in ((enabled.state_router, 8), (enabled.spatial_state_router, 81)):
                permutation_check(router, count)
            x = torch.randn(batch, 1, 15, 7, 7)
            diff_logits = close(enabled(x), old(x))
        for handle in handles:
            handle.remove()
        record('zero_gate_B' + str(batch), g_min=g.min().item(), g_max=g.max().item(),
               gate_shape=list(g.shape), gate_input_shape=[batch, 129, 9, 9],
               output_shape=list(result.shape), max_abs_diff_spatial=diff_spatial,
               max_abs_diff_reference=diff_reference, max_abs_diff_classifier=diff_logits,
               spectral_and_spatial_permutations='complete; both compositions identity')


def controlled_adaptation():
    model = serial(True)
    conv = model.spatial_residual_gate_conv
    spectral = torch.zeros(2, 64, 9, 9)
    grid = torch.linspace(-1, 1, 81).reshape(9, 9)
    spectral[0, 0], spectral[1, 0] = grid, grid.flip(0) + 0.7
    spatial = spectral + 0.25
    beta = torch.stack((grid + 1, grid.flip(1) + 1)).unsqueeze(1)
    # Controlled nonzero coefficients from all three input blocks.
    with torch.no_grad():
        conv.weight.zero_()
        conv.weight[0, 0, 0, 0] = 0.8
        conv.weight[0, 64, 0, 0] = -0.4
        conv.weight[0, 128, 0, 0] = 0.6
        conv.bias.fill_(0.1)
        def evaluate(s, p, b):
            data = torch.cat((s, p - s, b), 1)
            g = 2 * torch.sigmoid(conv(data))
            out = model._gate_spatial_residual(s, p, b)
            close(out, s + g * (p - s))
            finite(g)
            require(g.min() >= 0 and g.max() <= 2, 'gate out of range')
            return out, g
        output, gate = evaluate(spectral, spatial, beta)
        expected_logits = 0.8 * spectral[:, :1] - 0.4 * (spatial - spectral)[:, :1] + 0.6 * beta + 0.1
        close(gate, 2 * torch.sigmoid(expected_logits))
        require(all(gate[i].max() - gate[i].min() > 0.1 for i in (0, 1)), 'no spatial adaptation')
        require(not torch.allclose(gate[0], gate[1]), 'fixture lacks sample variation')
        single_diffs = []
        for i in (0, 1):
            one, one_gate = evaluate(spectral[i:i+1], spatial[i:i+1], beta[i:i+1])
            single_diffs.append(max(close(one, output[i:i+1]), close(one_gate, gate[i:i+1])))
        s2, p2, b2 = spectral.clone(), spatial.clone(), beta.clone()
        s2[1] += 50
        p2[1] -= 70
        b2[1] += 30
        changed, changed_gate = evaluate(s2, p2, b2)
        unaffected = max(close(changed[:1], output[:1]), close(changed_gate[:1], gate[:1]))
        swapped, swapped_gate = evaluate(spectral.flip(0), spatial.flip(0), beta.flip(0))
        close(swapped, output.flip(0))
        close(swapped_gate, gate.flip(0))
        conv.weight.zero_()
        conv.bias.fill_(0.7)
        _, bias_gate = evaluate(spectral, spatial, beta)
        require(torch.unique(bias_gate).numel() == 1, 'bias-only gate should be uniform')
        conv.bias.fill_(1000)
        _, upper = evaluate(spectral, spatial, beta)
        conv.bias.fill_(-1000)
        _, lower = evaluate(spectral, spatial, beta)
    record('controlled_adaptation', gate_shape=list(gate.shape), g_min=gate.min().item(), g_max=gate.max().item(),
           sample_single_max_abs_diff=single_diffs, changed_other_sample_max_abs_diff=unaffected,
           batch_swap=True, bias_only_uniform_value=bias_gate.flatten()[0].item(),
           saturation_endpoints=[lower.min().item(), upper.max().item()])
    # Full route sample isolation additionally covers recurrent spatial execution.
    with torch.no_grad():
        conv.weight.fill_(0.01)
        conv.bias.zero_()
        features = torch.randn(2, 64, 9, 9)
        full = model._route(features)[0]
        separate = torch.cat([model._route(features[i:i+1])[0] for i in (0, 1)])
        split_diff = close(full, separate)
        altered = features.clone()
        altered[1] *= -3
        isolated = close(model._route(altered)[0][:1], full[:1])
    record('full_route_sample_isolation', separate_max_abs_diff=split_diff, other_sample_max_abs_diff=isolated)


def gradients():
    for batch in (1, 2):
        for nonzero in (False, True):
            model = serial(True)
            if nonzero:
                with torch.no_grad():
                    model.spatial_residual_gate_conv.weight.fill_(0.01)
            values, handles = capture(model)
            inputs = torch.randn(batch, 1, 15, 7, 7, requires_grad=True)
            before = {k: p.detach().clone() for k, p in model.named_parameters()}
            output = model(inputs)
            output.square().mean().backward()  # Synthetic scalar probe, no added training loss.
            finite(inputs.grad)
            require(inputs.grad.abs().sum() > 0, 'input gradient is zero')
            finite(values['beta'].grad)
            require(values['beta'].grad.abs().sum() > 0, 'beta gradient is zero')
            summaries = {}
            for name, p in model.named_parameters():
                require(torch.equal(before[name], p.detach()), 'backward modified weight: ' + name)
                if p.grad is not None:
                    finite(p.grad)
                if name.startswith(('spatial_state_router.', 'spatial_residual_gate_conv.')):
                    finite(p.grad)
                    summaries[name] = {'max_abs': p.grad.abs().max().item(), 'nonzero': bool(torch.count_nonzero(p.grad))}
            require(all(summaries[k]['nonzero'] for k in NEW_KEYS), 'gate parameter lacks expected gradient')
            finite(values['gate_input'].grad)
            extra = values['gate_input'].grad.abs().max().item()
            require((extra > 0) if nonzero else (extra == 0), 'gate input gradient property violated')
            for handle in handles:
                handle.remove()
            record(f'gradients_B{batch}_nonzero{nonzero}', input_max_abs=inputs.grad.abs().max().item(),
                   beta_max_abs=values['beta'].grad.abs().max().item(), gate_input_extra_max_abs=extra,
                   parameter_gradients=summaries, all_weights_unchanged=True, optimizer_step=False)
    # Independent direct-input test: at W=0, dFout/dFs=0, dFout/dFp=I,
    # and beta gets zero additional gradient through the gate.
    model = serial(True)
    s = torch.randn(1, 64, 9, 9, requires_grad=True)
    p = torch.randn_like(s, requires_grad=True)
    beta = torch.ones(1, 1, 9, 9, requires_grad=True)
    model._gate_spatial_residual(s, p, beta).sum().backward()
    require(torch.count_nonzero(s.grad) == 0 and torch.count_nonzero(beta.grad) == 0, 'extra direct gradient')
    require(torch.equal(p.grad, torch.ones_like(p)), 'spatial direct gradient')
    record('zero_gate_direct_derivatives', spectral=0, beta=0, spatial=1,
           note='Full upstream gradients still flow through spatial state; zero gate input gradient is expected.')


def interfaces_and_static():
    require(inspect.signature(build_model).parameters['spatial_residual_gate'].default is False, 'factory default')
    sig = inspect.signature(gscvit.GSCViTDSSRSpatialWrapper)
    require(sig.parameters['spatial_residual_gate'].default is False, 'wrapper default')
    require(sig.parameters['spatial_state_strength'].default == 0.5, 'strength default')
    require(inspect.signature(build_model).parameters['spatial_state_strength'].default == 0.5, 'factory strength')
    default = build_model('gscvit_dssr_spatial', 15, 22, 7)
    require(default.spatial_residual_gate is False and not hasattr(default, 'spatial_residual_gate_conv'), 'default module')
    require(default.spatial_state_router.enabled is False, 'pre-existing factory spatial default changed')
    for name in ('gscvit', 'gscvit_tssr', 'gscvit_dssr', 'gscvit_spatial'):
        try:
            build_model(name, 15, 22, 7, spatial_residual_gate=True)
        except ValueError:
            pass
        else:
            raise AssertionError('gate accepted by nonserial model')
    model = serial(True)
    f = torch.randn(2, 64, 9, 9)
    beta4 = torch.rand(2, 1, 9, 9)
    with torch.no_grad():
        four = model.spatial_state_router(f, beta4)
        three = model.spatial_state_router(f, beta4[:, 0])
        beta_diff = close(four, three)
        permutation_check(model.spatial_state_router, 81)
    for bad in (beta4[:, 0], beta4[:1], beta4.expand(-1, 2, -1, -1)):
        try:
            model._gate_spatial_residual(f, four, bad)
        except ValueError:
            pass
        else:
            raise AssertionError('gate accepted malformed beta')
    off = serial(True, spatial=False)
    with torch.no_grad():
        routed, alpha, beta = off.dynamic_router.route(f)
        expected = off.state_router(routed, alpha, beta)
        off_diff = close(off._route(f)[0], expected)
    path = ROOT / 'diagnostics/honghu_single_seed_20261018/diagnostic_train.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    # Execute only extracted argparse declarations, not import/main/training code.
    import argparse
    declarations = [node for node in main.body if isinstance(node, ast.Expr)
                    and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
                    and isinstance(node.value.func.value, ast.Name)
                    and node.value.func.value.id == 'parser' and node.value.func.attr == 'add_argument']
    parser = argparse.ArgumentParser()
    env = {'parser': parser, 'MODEL_FLAGS': {'gscvit_dssr_spatial': True}}
    exec(compile(ast.Module(body=declarations, type_ignores=[]), str(path), 'exec'), env)
    argv = ['--project-root', '.', '--assets-dir', 'UNUSED', '--model-name', 'gscvit_dssr_spatial', '--output-dir', 'UNUSED']
    parsed_default = parser.parse_args(argv)
    parsed_on = parser.parse_args(argv + ['--spatial-residual-gate'])
    require(parsed_default.spatial_residual_gate is False and parsed_on.spatial_residual_gate is True, 'CLI opt-in')
    require(parsed_default.spatial_state_strength == 0.5, 'CLI strength')
    calls = [n for n in ast.walk(main) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'build_model']
    require(len(calls) == 1, 'unexpected build_model call count')
    kw = {k.arg: k.value for k in calls[0].keywords}
    require(ast.unparse(kw['spatial_residual_gate']) == 'args.spatial_residual_gate', 'CLI flag not forwarded')
    # Execute the isolated build_model expression against a recorder to check kwargs.
    captured = {}
    def recorder(*args, **kwargs):
        captured.update(kwargs)
    for args in (parsed_default, parsed_on):
        captured.clear()
        eval(compile(ast.Expression(calls[0]), str(path), 'eval'),
             {'build_model': recorder, 'args': args, 'spatial_enabled': True})
        require(captured['spatial_residual_gate'] is args.spatial_residual_gate, 'argument propagation')
    sources = ['compare/gscvit.py', 'models/__init__.py',
               'diagnostics/honghu_single_seed_20261018/diagnostic_train.py',
               'verify_spatial_residual_gate.py',
               'diagnostics/spatial_residual_gate_20261019/audit.py']
    for source in sources:
        compile((ROOT / source).read_text(encoding='utf-8'), source, 'exec')
    record('interfaces_and_static', factory_and_CLI_defaults=False, spatial_strength_default=0.5,
           nonserial_opt_in_rejected=True, old_spatial_beta_3d_4d_max_abs_diff=beta_diff,
           gate_beta_strict_4d=True, spatial_disabled_max_abs_diff=off_diff,
           isolated_argparse_and_call_test=True, training_entry_executed=False, compiled=sources)


def ordered_gain_regression():
    # Existing independently derived constant-residual oracle; do not edit router.
    from verify_spatial_dssr import check_ordered_gain_regression
    check_ordered_gain_regression()
    # Add production-size, distinct sample orders and both inverse compositions.
    for batch in (1, 2):
        router = dynamic_router.SpatialStateRouter(64)
        f = torch.randn(batch, 64, 9, 9)
        beta = torch.linspace(0.1, 1.9, 81).reshape(1, 1, 9, 9).repeat(batch, 1, 1, 1)
        if batch == 2:
            beta[1] = beta[1].flip(-1)
        residual = torch.linspace(-0.3, 0.4, 64)
        with torch.no_grad():
            router.from_state.weight.zero_()
            router.from_state.bias.copy_(residual)
            actual = router(f, beta)
            expected = f + 0.5 * torch.tanh(router.mix) * beta * residual.reshape(1, 64, 1, 1)
            diff = close(actual, expected)
            permutation_check(router, 81)
            if batch == 2:
                require(not torch.equal(router.last_permutation[0], router.last_permutation[1]), 'same sample order')
        record('ordered_gain_production_B' + str(batch), max_abs_diff=diff, both_inverse_compositions=True)


def main():
    torch.set_num_threads(1)
    torch.manual_seed(20261019)  # Synthetic fixture only; no experiment seed change.
    REPORT['environment'] = {'python': sys.version, 'executable': sys.executable,
                             'pytorch': torch.__version__, 'device': 'cpu', 'threads': 1}
    REPORT['command'] = [sys.executable, *sys.argv]
    REPORT['runtime_sources'] = {}
    for rel, module in [('compare/gscvit.py', gscvit), ('compare/dynamic_router.py', dynamic_router),
                        ('models/__init__.py', sys.modules['models'])]:
        loaded = Path(module.__file__).resolve()
        require(loaded == (ROOT / rel).resolve(), 'loaded source path mismatch')
        REPORT['runtime_sources'][rel] = {'path': str(loaded), 'sha256': sha(loaded)}
    print(json.dumps(REPORT['environment']))
    print('Predeclared tolerance:', ATOL, RTOL)
    try:
        reference = load_reference()
        legacy_regressions(reference)
        zero_and_checkpoint(reference)
        controlled_adaptation()
        gradients()
        interfaces_and_static()
        ordered_gain_regression()
        REPORT['synthetic_status'] = 'PASS'
        REPORT['task_status'] = 'PARTIAL: actual historical checkpoint strict load not performed'
    except Exception:
        REPORT['synthetic_status'] = 'FAIL'
        REPORT['error'] = traceback.format_exc()
        raise
    finally:
        (OUT / 'verification.json').write_text(json.dumps(REPORT, indent=2), encoding='utf-8')
    print('SYNTHETIC CORRECTNESS: PASS; historical checkpoint acceptance remains partial')


if __name__ == '__main__':
    main()
