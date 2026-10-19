"""Local provenance capture; never loads experiment assets or checkpoints."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCES = ['compare/gscvit.py', 'compare/dynamic_router.py', 'models/__init__.py']
AUTHORIZED_INPUT = 'checkpoint_inputs/exec03_serial/checkpoint_epoch_192.pth'
AUTHORIZED_HASH = '606ef81f050cda8171dd0effe05f77ef6a24c12289942f5e076f253cf0183304'
EDITABLE = {'compare/gscvit.py', 'models/__init__.py',
            'diagnostics/honghu_single_seed_20261018/diagnostic_train.py',
            'verify_spatial_residual_gate.py'}


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args):
    run = subprocess.run(['git', *args], cwd=ROOT, capture_output=True)
    return {'command': ['git', *args], 'exit_code': run.returncode,
            'stdout': run.stdout.decode('utf-8', errors='replace'),
            'stderr': run.stderr.decode('utf-8', errors='replace')}


def inventory():
    records = {}
    for base, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', '.idea', '.vscode'}]
        for name in files:
            path = Path(base) / name
            rel = path.relative_to(ROOT).as_posix()
            if rel.startswith('diagnostics/spatial_residual_gate_20261019/'):
                continue
            records[rel] = {'size': path.stat().st_size, 'sha256': sha(path)}
    return records


def main():
    if sys.argv[1] == 'before':
        target = OUT / 'reference_gscvit_before.py'
        if target.exists() or (OUT / 'before.json').exists():
            raise RuntimeError('refusing to overwrite pre-edit evidence')
        target.write_bytes((ROOT / SOURCES[0]).read_bytes())
        record = {'head': git('rev-parse', 'HEAD'), 'branch': git('branch', '--show-current'),
                  'status': git('status', '--short'), 'files': inventory(),
                  'reference_sha256': sha(target)}
        (OUT / 'before.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
        print('Captured pre-edit source and SHA256 inventory:', len(record['files']), 'files')
    elif sys.argv[1] == 'after':
        before = json.loads((OUT / 'before.json').read_text(encoding='utf-8'))
        current = inventory()
        changed = [rel for rel, old in before['files'].items() if current.get(rel) != old]
        unexpected = sorted(set(changed) - EDITABLE)
        added = sorted(set(current) - set(before['files']))
        authorized_inputs = {}
        if AUTHORIZED_INPUT in current:
            if current[AUTHORIZED_INPUT]['sha256'] != AUTHORIZED_HASH:
                raise AssertionError('authorized checkpoint hash mismatch')
            authorized_inputs[AUTHORIZED_INPUT] = {
                **current[AUTHORIZED_INPUT], 'git_staging': 'exclude',
                'source': '/root/HSI_Baseline_honghu_diag_20261018/diagnostics/honghu_single_seed_20261018/runs_20261018_exec_03/gscvit_dssr_spatial/checkpoint_epoch_192.pth',
                'authorization': 'User supplied exact local path and SHA256 for read-only strict-load supplement.'}
        unexpected_added = sorted(set(added) - EDITABLE - set(authorized_inputs))
        files = {rel: sha(ROOT / rel) for rel in sorted(EDITABLE) if (ROOT / rel).exists()}
        for path in sorted(OUT.rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts and path.name != 'manifest.json':
                files[path.relative_to(ROOT).as_posix()] = sha(path)
        record = {'status': 'PASS' if not unexpected and not unexpected_added else 'FAIL',
                  'audit_command': [sys.executable, *sys.argv],
                  'audit_exit_code': 0 if not unexpected and not unexpected_added else 1,
                  'inventory_exclusions': ['.git', '__pycache__', '.idea', '.vscode',
                                           'diagnostics/spatial_residual_gate_20261019/'],
                  'git': [git('rev-parse', 'HEAD'), git('branch', '--show-current'), git('status', '--short')],
                  'changed_existing_files': changed, 'added_files_outside_dedicated_folder': added,
                  'authorized_verification_inputs': authorized_inputs,
                  'unexpected_modified_or_deleted': unexpected, 'unexpected_added': unexpected_added,
                  'protected_existing_files_verified': len(before['files']) - len(changed),
                  'artifact_sha256': files,
                  'source_sha256': {rel: sha(ROOT / rel) for rel in SOURCES},
                  'manifest_self_hash': 'Excluded to avoid self-reference; hash externally if needed.'}
        (OUT / 'manifest.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
        print(json.dumps({k: v for k, v in record.items() if k not in {'artifact_sha256', 'git'}}, indent=2))
        if unexpected or unexpected_added:
            raise AssertionError('files outside allowed scope changed')
    else:
        raise ValueError('use before or after')


if __name__ == '__main__':
    main()
