"""Only the user-specified historical checkpoint, CPU synthetic eval, read-only."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
CHECKPOINT = ROOT / 'checkpoint_inputs/exec03_serial/checkpoint_epoch_192.pth'
EXPECTED = '606ef81f050cda8171dd0effe05f77ef6a24c12289942f5e076f253cf0183304'
ATOL, RTOL = 1e-6, 1e-5


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    report = {'command': [sys.executable, *sys.argv], 'status': 'FAIL',
              'checkpoint': str(CHECKPOINT), 'expected_sha256': EXPECTED,
              'source': '/root/HSI_Baseline_honghu_diag_20261018/diagnostics/honghu_single_seed_20261018/runs_20261018_exec_03/gscvit_dssr_spatial/checkpoint_epoch_192.pth',
              'tolerance': {'atol': ATOL, 'rtol': RTOL}, 'training_authorized': False}
    try:
        report['sha256_before'] = sha(CHECKPOINT)
        assert report['sha256_before'] == EXPECTED, 'checkpoint hash mismatch'
        import torch
        from models import build_model
        import compare.gscvit
        import compare.dynamic_router
        torch.set_num_threads(1)
        report['environment'] = {'python': sys.version, 'pytorch': torch.__version__,
                                 'device': 'cpu', 'dtype': 'float32', 'threads': 1}
        previous = json.loads((OUT / 'manifest.json').read_text(encoding='utf-8'))
        report['runtime_sources'] = {}
        for rel, module in [('compare/gscvit.py', compare.gscvit),
                            ('compare/dynamic_router.py', compare.dynamic_router),
                            ('models/__init__.py', sys.modules['models'])]:
            path = Path(module.__file__).resolve()
            assert path == (ROOT / rel).resolve()
            assert sha(path) == previous['source_sha256'][rel], 'model source changed since accepted verification'
            report['runtime_sources'][rel] = {'path': str(path), 'sha256': sha(path)}
        reference = OUT / 'reference_gscvit_before.py'
        before = json.loads((OUT / 'before.json').read_text(encoding='utf-8'))
        assert sha(reference) == before['reference_sha256']
        report['frozen_reference_sha256'] = sha(reference)
        spec = importlib.util.spec_from_file_location('checkpoint_frozen_reference', reference)
        old_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old_module)
        old = old_module.GSCViTDSSRSpatialWrapper(15, 22, 7, spatial_state_enabled=True,
                                                  spatial_state_strength=0.5).cpu().eval()
        new = build_model('gscvit_dssr_spatial', 15, 22, 7, spatial_state_enabled=True,
                          spatial_state_strength=0.5, spatial_residual_gate=False).cpu().eval()
        assert not hasattr(new, 'spatial_residual_gate_conv')
        # No key renaming, filtering, extension, or fallback loading.
        with CHECKPOINT.open('rb') as stream:
            state = torch.load(stream, map_location='cpu', weights_only=True)
        assert isinstance(state, dict) and all(torch.is_tensor(v) for v in state.values()), 'expected raw state_dict'
        report['checkpoint_key_count'] = len(state)
        report['strict_loading'] = {}
        for label, model in [('frozen_old', old), ('new_gate_disabled', new)]:
            loaded = model.load_state_dict(state, strict=True)
            assert not loaded.missing_keys and not loaded.unexpected_keys
            assert all(torch.equal(model.state_dict()[k], v) for k, v in state.items())
            report['strict_loading'][label] = {'strict': True, 'missing_keys': loaded.missing_keys,
                                               'unexpected_keys': loaded.unexpected_keys}
        generator = torch.Generator(device='cpu').manual_seed(20261019)
        report['synthetic_seed'] = 20261019
        report['comparisons'] = []
        with torch.inference_mode():
            for batch in (1, 2):
                inputs = torch.randn(batch, 1, 15, 7, 7, generator=generator)
                expected, actual = old(inputs), new(inputs)
                assert expected.shape == actual.shape == (batch, 22)
                assert torch.isfinite(expected).all() and torch.isfinite(actual).all()
                diff = (actual - expected).abs().max().item()
                assert torch.allclose(actual, expected, atol=ATOL, rtol=RTOL)
                report['comparisons'].append({'batch': batch, 'input_shape': list(inputs.shape),
                    'output_shape': list(actual.shape), 'max_abs_diff': diff, 'finite': True,
                    'allclose': True, 'eval': not old.training and not new.training})
        for model in (old, new):
            assert all(torch.equal(model.state_dict()[k], v) for k, v in state.items())
        report['model_state_unchanged_after_eval'] = True
        report['status'] = 'PASS'
    except Exception:
        report['error'] = traceback.format_exc()
    finally:
        report['sha256_after'] = sha(CHECKPOINT) if CHECKPOINT.is_file() else None
        report['checkpoint_unchanged'] = report['sha256_after'] == report.get('sha256_before') == EXPECTED
        if not report['checkpoint_unchanged']:
            report['status'] = 'FAIL'
        report['exit_code'] = 0 if report['status'] == 'PASS' else 1
        (OUT / 'checkpoint_verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report, indent=2))
    return report['exit_code']


if __name__ == '__main__':
    sys.exit(main())
