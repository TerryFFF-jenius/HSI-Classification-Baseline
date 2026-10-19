"""Capture only the agreed source/asset boundary; never loads array/checkpoint data."""
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()

def git(*args):
    p = subprocess.run(['git', '--no-optional-locks', '-C', str(ROOT), *args], capture_output=True, text=True, encoding='utf-8', errors='replace')
    return {'command': p.args, 'exit_code': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}

def main():
    target = HERE / 'baseline.json'
    if target.exists():
        raise FileExistsError(target)
    protected = ['compare/gscvit.py', 'compare/dynamic_router.py', 'models/__init__.py', 'main_train.py', 'main_test.py', '.gitignore', 'diagnostics/honghu_single_seed_20261018/feature_diagnosis.py']
    # Exact accepted evidence directory only: byte hashes, no historical search.
    protected += [str(p.relative_to(ROOT)).replace('\\', '/') for p in (ROOT / 'diagnostics/spatial_residual_gate_20261019').iterdir() if p.is_file()]
    protected += [str(p.relative_to(ROOT)).replace('\\', '/') for p in (ROOT / 'diagnostics/honghu_single_seed_20261018').glob('*.json')]
    protected += ['diagnostics/honghu_single_seed_20261018/README.md']
    protected += [str(p.relative_to(ROOT)).replace('\\', '/') for p in (ROOT / 'diagnostics/honghu_single_seed_20261018/assets').iterdir() if p.is_file()]
    trainer = ROOT / 'diagnostics/honghu_single_seed_20261018/diagnostic_train.py'
    original = HERE / 'diagnostic_train_before.txt'
    with original.open('xb') as f:
        f.write(trainer.read_bytes())
    report = {'head': git('rev-parse', 'HEAD'), 'branch': git('branch', '--show-current'), 'status': git('status', '--porcelain=v1', '--untracked-files=all'), 'remote_verified': False, 'protected': {p: digest(ROOT / p) for p in protected}, 'trainer_before_sha256': digest(trainer), 'original_source': original.name, 'training_authorized': False, 'status_label': 'prepared_not_executed'}
    target.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'protected_file_count': len(protected), 'head': report['head']['stdout'].strip(), 'trainer_before_sha256': report['trainer_before_sha256']}))

if __name__ == '__main__':
    main()
