"""Record bounded preparation commands and their real exit codes; no training entry."""
import argparse
import ast
import json
from pathlib import Path
import subprocess
import sys
import hashlib

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-name', required=True)
    args = parser.parse_args()
    if Path(args.evidence_name).name != args.evidence_name or args.evidence_name in ('.', '..'):
        raise ValueError('Use a simple evidence directory name')
    output = HERE / args.evidence_name
    if output.exists():
        raise FileExistsError(output)
    output.mkdir()
    baseline = json.loads((HERE / 'baseline.json').read_text(encoding='utf-8'))
    baseline_head = baseline['head']['stdout'].strip()
    synthetic_dir = output / 'synthetic_cpu'
    future = output / 'unused_future_target'
    launch = [sys.executable, '-B', str(HERE / 'launch.py'), '--project-root', str(ROOT),
              '--assets-dir', str(ROOT / 'diagnostics/honghu_single_seed_20261018/assets'),
              '--output-dir', str(future), '--expected-commit', baseline_head]
    jobs = [
        ([sys.executable, '-B', str(HERE / 'synthetic_checks.py'), '--device', 'cpu', '--output-dir', str(synthetic_dir)], 0),
        (launch + ['--report', str(output / 'preflight_local.json')], 0),
        (launch[:launch.index('--output-dir') + 1] + [str(synthetic_dir)] + launch[launch.index('--expected-commit'):], 1),
        (launch[:-1] + ['0' * 40], 1),
        (['git', '--no-optional-locks', 'diff', '--check', '--', 'diagnostics/honghu_single_seed_20261018/diagnostic_train.py'], 0),
    ]
    records = []
    for command, expected in jobs:
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding='utf-8', errors='replace')
        records.append({'actual_command': command, 'exit_code': result.returncode, 'expected_exit_code': expected,
                        'stdout': result.stdout, 'stderr': result.stderr})
        (output / 'commands.json').write_text(json.dumps(records, indent=2, ensure_ascii=False) + chr(10), encoding='utf-8')
        if result.returncode != expected:
            print(json.dumps(records[-1], ensure_ascii=False))
            raise RuntimeError('Preparation command returned unexpected exit code; evidence retained')
        print(json.dumps({'command': Path(command[2]).name if len(command) > 2 and command[1] == '-B' else command, 'exit_code': result.returncode, 'expected': expected}))
    protected = {rel: {'before': expected, 'after': digest(ROOT / rel)} for rel, expected in baseline['protected'].items()}
    assert all(v['before'] == v['after'] for v in protected.values()), 'Protected file changed'
    paths = list(HERE.glob('*.py')) + [ROOT / 'diagnostics/honghu_single_seed_20261018/diagnostic_train.py']
    for path in paths:
        compile(path.read_text(encoding='utf-8'), str(path), 'exec')
    # Verify source boundary: new scripts cannot call optimizer.step or deserialize assets/checkpoints.
    forbidden = []
    for path in HERE.glob('*.py'):
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == 'step' or (node.func.attr == 'load' and isinstance(node.func.value, ast.Name) and node.func.value.id in ('np', 'torch')):
                    forbidden.append([str(path), node.lineno])
    assert not forbidden, forbidden
    reference_local = ROOT / 'diagnostics/honghu_single_seed_20261018/runs_20261018_exec_03/gscvit_dssr_spatial/train_config.json'
    status = subprocess.run(['git', '--no-optional-locks', 'status', '--porcelain=v1', '--untracked-files=all'], cwd=ROOT, capture_output=True, text=True, encoding='utf-8', errors='replace')
    head = subprocess.run(['git', '--no-optional-locks', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True)
    assert head.returncode == 0 and head.stdout.strip() == baseline_head
    report = {'status': 'preparation_checks_passed', 'training_authorized': False, 'training_status': 'prepared_not_executed',
              'commands': records, 'protected_files': protected, 'protected_count': len(protected),
              'compiled_in_memory': [str(p.relative_to(ROOT)) for p in paths], 'forbidden_calls': forbidden,
              'head': head.stdout.strip(), 'git_status_command_exit_code': status.returncode, 'status_short': status.stdout.splitlines(),
              'future_target_created': future.exists(), 'remote_verified': False, 'autodl_accessed': False,
              'exact_local_reference_config_path': str(reference_local), 'exact_local_reference_config_exists': reference_local.is_file(),
              'full_reference_config_read': False, 'comparison_basis': 'synthetic fixtures only',
              'actual_command': [sys.executable, *sys.argv]}
    assert not future.exists()
    (output / 'boundary_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + chr(10), encoding='utf-8')
    print(json.dumps({'status': report['status'], 'protected_file_count': len(protected), 'evidence_dir': str(output), 'training_started': False}))

if __name__ == '__main__':
    main()
