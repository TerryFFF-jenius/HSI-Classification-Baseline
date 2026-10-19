"""Run the one approved HongHu spatial-strength adaptation.

This wrapper fixes the model, strength, and budget so the command cannot
silently become a sweep or a second model run. It writes the exit code only
after the training process returns, leaving the training directory empty
before launch as required by diagnostic_train.py.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


MODEL_NAME = "gscvit_dssr_spatial"
SPATIAL_STATE_STRENGTH = 0.25


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--assets-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    assets = Path(args.assets_dir).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output}")
    output_empty_before_launch = not output.exists() or not any(output.iterdir())
    if not assets.is_dir():
        raise FileNotFoundError(f"asset directory not found: {assets}")
    train_script = root / "diagnostics" / "honghu_single_seed_20261018" / "diagnostic_train.py"
    if not train_script.is_file():
        raise FileNotFoundError(f"diagnostic training script not found: {train_script}")

    command = [
        sys.executable,
        str(train_script),
        "--project-root", str(root),
        "--assets-dir", str(assets),
        "--model-name", MODEL_NAME,
        "--output-dir", str(output),
        "--spatial-state-strength", str(SPATIAL_STATE_STRENGTH),
    ]
    started = datetime.now(timezone.utc)
    start_monotonic = time.monotonic()
    completed = subprocess.run(command, cwd=root, check=False)
    ended = datetime.now(timezone.utc)
    output.mkdir(parents=True, exist_ok=True)
    record = {
        "status": "completed" if completed.returncode == 0 else "failed",
        "started_at_utc": started.isoformat(),
        "ended_at_utc": ended.isoformat(),
        "elapsed_seconds": time.monotonic() - start_monotonic,
        "exit_code": completed.returncode,
        "command": command,
        "working_directory": str(root),
        "model_name": MODEL_NAME,
        "spatial_state_strength": SPATIAL_STATE_STRENGTH,
        "training_seed": 20261017,
        "split_seed": 20261018,
        "epochs": 200,
        "batch_size": 32,
        "assets_dir": str(assets),
        "assets_file_manifest_sha256": sha256_file(assets / "file_manifest.json"),
        "output_empty_before_launch": output_empty_before_launch,
        "training_script_sha256": sha256_file(train_script),
        "wrapper_sha256": sha256_file(Path(__file__).resolve()),
        "test_used": False,
    }
    (output / "execution_record.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
