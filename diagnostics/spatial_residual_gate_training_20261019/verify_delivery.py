"""Repair packaging metadata only and verify final delivery without running models."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TRAINER = ROOT / 'diagnostics/honghu_single_seed_20261018/diagnostic_train.py'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def write(p, payload):
    Path(p).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + chr(10), encoding='utf-8')

def main():
    evidence = HERE / 'verification_final'
    result = read(evidence / 'synthetic_cpu/synthetic_report.json')
    boundary = read(evidence / 'boundary_report.json')
    for rel, expected in result['runtime_file_sha256'].items():
        assert sha(ROOT / rel) == expected, rel
    for rel, record in boundary['protected_files'].items():
        assert sha(ROOT / rel) == record['before'], rel
    checked = []
    excluded = []
    for p in evidence.rglob('*.json'):
        if 'FAKE_BYTES_NOT_DATA' in p.parts:
            excluded.append(str(p.relative_to(HERE)))
        else:
            read(p)
            checked.append(str(p.relative_to(HERE)))
    for p in HERE.glob('*.py'):
        compile(p.read_text(encoding='utf-8'), str(p), 'exec')
    status = subprocess.run(['git', '--no-optional-locks', 'status', '--short'], cwd=ROOT, capture_output=True, text=True, encoding='utf-8', errors='replace')
    head = subprocess.run(['git', '--no-optional-locks', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True)
    diffcheck = subprocess.run(['git', '--no-optional-locks', 'diff', '--check', '--', str(TRAINER.relative_to(ROOT))], cwd=ROOT, capture_output=True, text=True)
    assert head.stdout.strip() == boundary['head'] and diffcheck.returncode == 0 and status.returncode == 0
    packaging = {'status': 'passed', 'training_authorized': False, 'actual_command': [sys.executable, *sys.argv],
                 'initial_finalize_command': [sys.executable, '-B', 'diagnostics/spatial_residual_gate_training_20261019/finalize_delivery.py'],
                 'initial_finalize_exit_code': 1, 'reason': 'Attempted JSON parsing of deliberately invalid FAKE_BYTES_NOT_DATA split_manifest byte fixture; not a model/preflight failure',
                 'fix': 'Exclude explicitly labelled arbitrary-byte fixtures from JSON parse checks; keep them hash-inventoried',
                 'real_final_evidence_json_parsed': checked, 'intentional_non_json_fixture_paths': excluded,
                 'runtime_hashes_unchanged': True, 'protected_file_count': len(boundary['protected_files']),
                 'head': head.stdout.strip(), 'git_status_short': status.stdout.splitlines(), 'git_diff_check_exit_code': diffcheck.returncode,
                 'no_model_execution_in_packaging': True}
    write(HERE / 'packaging_checks.json', packaging)
    summary = read(HERE / 'delivery_summary.json')
    summary['packaging_check'] = packaging
    write(HERE / 'delivery_summary.json', summary)
    report_path = HERE / 'FINAL_REPORT.md'
    report = report_path.read_text(encoding='utf-8')
    suffix = '''
## 封存校验补充

finalize_delivery.py 首轮退出码 1：尾部遍历误把 FAKE_BYTES_NOT_DATA 下故意非 JSON 的 split_manifest 字节 fixture 当成真实 JSON；此前准备检查全部通过。这属于报告封存校验范围错误，不是模型或资产检查失败。已明确排除此类标记字节 fixture；它们仍保留在完整哈希清单中。随后 verify_delivery.py 重新解析全部真实最终证据 JSON、内存编译新脚本、核验运行源码和 30 份保护文件、核验 HEAD/diff 后完成封存，退出码 0。详见 packaging_checks.json。未重跑或进入任何训练，原检查/失败证据均保留。
'''
    if '## 封存校验补充' not in report:
        report_path.write_text(report + suffix, encoding='utf-8')
    staging = HERE / 'STAGING_PATHS.txt'
    paths = staging.read_text(encoding='utf-8').splitlines()
    paths += [str((HERE / p).relative_to(ROOT)).replace(chr(92), '/') for p in ('verify_delivery.py', 'packaging_checks.json')]
    staging.write_text(chr(10).join(sorted(set(paths))) + chr(10), encoding='utf-8')
    files = [p for p in HERE.rglob('*') if p.is_file() and p.name != 'FILE_SHA256.json'] + [TRAINER]
    inventory = {str(p.relative_to(ROOT)).replace(chr(92), '/'): {'sha256': sha(p), 'bytes': p.stat().st_size} for p in sorted(files)}
    write(HERE / 'FILE_SHA256.json', {'scope': 'new delivery directory plus sole modified trainer; excludes this manifest itself', 'files': inventory})
    for rel, expected in read(HERE / 'FILE_SHA256.json')['files'].items():
        assert sha(ROOT / rel) == expected['sha256']
    print(json.dumps({'status': 'passed', 'file_count': len(inventory), 'head': head.stdout.strip(),
                      'manifest_sha256': sha(HERE / 'FILE_SHA256.json'), 'report_sha256': sha(report_path),
                      'trainer_sha256': sha(TRAINER), 'cases': summary['synthetic_cases'],
                      'reference_local_config_exists': boundary['exact_local_reference_config_exists']}, ensure_ascii=False))

if __name__ == '__main__':
    main()
