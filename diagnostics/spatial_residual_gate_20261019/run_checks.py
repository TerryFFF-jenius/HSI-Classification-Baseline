"""Run and record local synthetic verification, never a training entry point."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
commands = [[sys.executable, 'verify_spatial_residual_gate.py'], ['git', 'diff', '--check']]
records = []
history_path = OUT / 'command_history.json'
history = json.loads(history_path.read_text(encoding='utf-8')) if history_path.exists() else []
previous = OUT / 'commands.json'
if previous.exists() and not history:
    history.append(json.loads(previous.read_text(encoding='utf-8')))
for command in commands:
    result = subprocess.run(command, cwd=ROOT, capture_output=True)
    stdout = result.stdout.decode('utf-8', errors='replace')
    stderr = result.stderr.decode('utf-8', errors='replace')
    records.append({'command': command, 'exit_code': result.returncode, 'stdout': stdout, 'stderr': stderr})
    print(stdout, end='')
    print(stderr, end='', file=sys.stderr)
    print('EXIT CODE:', result.returncode)
    if result.returncode:
        break
(OUT / 'commands.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
history.append(records)
history_path.write_text(json.dumps(history, indent=2), encoding='utf-8')
sys.exit(records[-1]['exit_code'])
