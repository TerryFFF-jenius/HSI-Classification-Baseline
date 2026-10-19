"""Compare complete JSON logs and checkpoint hashes only; no model or data forward."""
import argparse
import copy
import csv
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from diagnostics.spatial_residual_gate_training_20261019 import runtime

def validate_run(run, plan, gate):
    expected = dict(plan['config'], spatial_residual_gate=gate)
    config = dict(run['config'])
    config.setdefault('spatial_residual_gate', False)  # Original pre-gate record means gate disabled.
    for k, v in expected.items():
        if config.get(k) != v:
            raise ValueError('Frozen configuration differs: ' + k)
    for k, value in {'model_name': plan['model_name'], 'dataset': 'HongHu', 'seed': 20261017,
                     'split_seed': 20261018, 'spatial_state_enabled': True, 'run_status': 'completed'}.items():
        if run.get(k) != value:
            raise ValueError('Incomplete/incompatible run: ' + k)
    if run.get('nonfinite_events'):
        raise ValueError('Numerical health failed')
    if run.get('assets_file_manifest_sha256') != plan['asset_anchors']['file_manifest.json']:
        raise ValueError('Asset manifest identity differs')
    for name, sha in plan['asset_anchors'].items():
        if run['assets'][name]['sha256'] != sha:
            raise ValueError('Asset identity differs: ' + name)
    history = run['history']
    if [r['epoch'] for r in history] != list(range(1, 201)):
        raise ValueError('Need all 200 epoch records')
    validations = []
    best = 0.0
    for record in history:
        if not math.isfinite(record['train_loss']) or not math.isfinite(record['learning_rate']):
            raise ValueError('Nonfinite curve entry')
        expected_lr = 3e-4 * 0.95 ** ((record['epoch'] - 1) / 10 + 1)
        if not math.isclose(record['learning_rate'], expected_lr, rel_tol=1e-12, abs_tol=1e-16):
            raise ValueError('Observed scheduler curve differs from actual frozen LambdaLR')
        if ('validation' in record) != (record['epoch'] % 2 == 0):
            raise ValueError('Need exactly 100 even-epoch validation records')
        if 'validation' in record:
            metrics = record['validation']
            if any(not math.isfinite(metrics[k]) for k in ('oa', 'aa', 'kappa')):
                raise ValueError('Nonfinite validation metric')
            saved = metrics['oa'] >= best
            if record.get('checkpoint_saved') is not saved:
                raise ValueError('Checkpoint selection violates >= tie rule')
            if saved:
                best = metrics['oa']
                if not isinstance(record.get('checkpoint_sha256'), str) or len(record['checkpoint_sha256']) != 64:
                    raise ValueError('Missing saved checkpoint SHA256')
            validations.append(record)
    selected = max(validations, key=lambda r: (r['validation']['oa'], r['epoch']))
    if run['best_checkpoint'] != f"checkpoint_epoch_{selected['epoch']:03d}.pth":
        raise ValueError('Best checkpoint does not match full history')
    if run['best_checkpoint_sha256'] != selected['checkpoint_sha256']:
        raise ValueError('Selected checkpoint hash differs')
    if gate:
        context = run.get('gate_preparation', {})
        if run.get('training_authorized') is not True or context.get('approval', {}).get('training_authorized') is not True:
            raise ValueError('Missing authorized gate run identity')
        if run.get('selected_epoch') != selected['epoch'] or run.get('selected_validation') != selected['validation']:
            raise ValueError('Gate final selected record missing or inconsistent')
        if run.get('initialization_alignment', {}).get('shared_state_exact') is not True or run['initialization_alignment'].get('cpu_rng_exact') is not True:
            raise ValueError('Missing initialization/RNG alignment evidence')
        if run.get('optimizer_membership', {}).get('gate_parameters_exactly_once') is not True:
            raise ValueError('Missing gate optimizer membership check')
    return selected

def compare(reference, candidate, *, fixture=False):
    if not fixture and (reference.get('SYNTHETIC_FIXTURE_ONLY') or candidate.get('SYNTHETIC_FIXTURE_ONLY')):
        raise ValueError('Synthetic fixtures cannot be reported as real results')
    plan = runtime.read_json(runtime.PLAN_PATH)
    old = validate_run(reference, plan, False)
    new = validate_run(candidate, plan, True)
    for name in reference['assets']:
        if name not in candidate['assets'] or any(reference['assets'][name][k] != candidate['assets'][name][k] for k in ('sha256', 'bytes')):
            raise ValueError('Shared assets differ between records: ' + name)
    if old['epoch'] != 192 or reference['best_checkpoint_sha256'] != plan['reference']['sha256']:
        raise ValueError('Reference is not accepted exec_03 epoch192')
    for key, value in plan['reference']['validation'].items():
        if abs(old['validation'][key] - value) > 5.1e-11:
            raise ValueError('Reference metrics differ from accepted rounded values')
    delta = {k: new['validation'][k] - old['validation'][k] for k in ('oa', 'aa', 'kappa')}
    rule = 'oa_greater' if delta['oa'] > 0 else 'oa_less_or_equal'
    return {'synthetic_fixture': fixture, 'primary_metric': 'validation OA', 'delta_percentage_points': delta,
            'reference_selected': old, 'candidate_selected': new,
            'decision': plan['decision_rules'][rule], 'numerical_health': 'all recorded metrics/loss finite; no recorded nonfinite events',
            'limitations': ['single seed', 'validation exploratory screening', 'full-scene transductive PCA',
                            'no test evaluation', 'weights nonzero do not prove effective adaptation'],
            'followup_training_authorized': False}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-config', required=True)
    parser.add_argument('--candidate-config', required=True)
    parser.add_argument('--reference-checkpoint', required=True, help='Hash bytes only; never deserialize')
    parser.add_argument('--candidate-checkpoint', required=True, help='Hash bytes only; never deserialize')
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    output = runtime.assert_empty_output(args.output_dir)
    old, new = runtime.read_json(args.reference_config), runtime.read_json(args.candidate_config)
    result = compare(old, new)
    for config, path in ((old, args.reference_checkpoint), (new, args.candidate_checkpoint)):
        if runtime.sha256(path) != config['best_checkpoint_sha256']:
            raise ValueError('Actual checkpoint byte hash differs from full config')
    exit_path = Path(args.candidate_config).parent / 'launch_exit.json'
    if runtime.read_json(exit_path).get('exit_code') != 0:
        raise ValueError('Candidate launch did not complete successfully')
    result['input_files'] = {str(Path(p).resolve()): runtime.sha256(p) for p in (args.reference_config, args.candidate_config, args.reference_checkpoint, args.candidate_checkpoint, exit_path)}
    result['actual_command'] = [sys.executable, *sys.argv]
    output.mkdir(parents=True, exist_ok=True)
    runtime.write_new(output / 'comparison.json', result)
    with (output / 'curves.csv').open('x', encoding='utf-8', newline='') as f:
        fields = ['run', 'epoch', 'train_loss', 'learning_rate', 'oa', 'aa', 'kappa']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for label, config in (('reference_exec_03', old), ('gate_candidate', new)):
            for record in config['history']:
                writer.writerow(dict(run=label, **{k: record[k] for k in ('epoch', 'train_loss', 'learning_rate')},
                                     **{k: record.get('validation', {}).get(k) for k in ('oa', 'aa', 'kappa')}))
    print(json.dumps(result, ensure_ascii=False))

if __name__ == '__main__':
    main()
