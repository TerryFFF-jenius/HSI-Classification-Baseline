"""Train exactly one HongHu diagnostic model from shared prepared assets.

The script never loads the test split. It is delivered for later reviewed
execution and is intentionally separate from main_train.py.
"""

import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from models import build_model


MODEL_FLAGS = {
    "gscvit_tssr": False,
    "gscvit_dssr": False,
    "gscvit_spatial": True,
    "gscvit_dssr_spatial": True,
}


RUNTIME_CODE_FILES = (
    "compare/dynamic_router.py",
    "compare/gscvit.py",
    "models/__init__.py",
    "diagnostics/honghu_single_seed_20261018/diagnostic_train.py",
)

ASSET_FILES = (
    "split_indices.npz",
    "sample_index_map.npz",
    "pca_transform.npz",
    "pca_scene.npz",
    "validation_batch.npz",
    "split_manifest.json",
    "file_manifest.json",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect_asset_identity(assets):
    """Record every shared diagnostic asset without reading test labels."""
    records = {}
    for name in ASSET_FILES:
        path = assets / name
        record = {
            "path": str(path),
            "bytes": None,
            "sha256": None,
            "exists": path.is_file(),
        }
        if path.is_file():
            record["bytes"] = path.stat().st_size
            record["sha256"] = sha256_file(path)
        records[name] = record
    return records


def validate_asset_identity(assets):
    """Refuse to train when any shared asset differs from its manifest."""
    manifest_path = assets / "file_manifest.json"
    expected = json.loads(manifest_path.read_text(encoding="utf-8"))
    observed = collect_asset_identity(assets)
    mismatches = []
    for name in ASSET_FILES:
        record = observed[name]
        expected_record = expected.get(name)
        if name == "file_manifest.json":
            if not record["exists"]:
                mismatches.append(f"{name}: missing")
            continue
        if expected_record is None:
            mismatches.append(f"{name}: absent from file_manifest.json")
            continue
        if not record["exists"]:
            mismatches.append(f"{name}: missing")
        elif record["bytes"] != expected_record.get("bytes") or record["sha256"] != expected_record.get("sha256"):
            mismatches.append(
                f"{name}: expected bytes={expected_record.get('bytes')} sha256={expected_record.get('sha256')}; "
                f"observed bytes={record['bytes']} sha256={record['sha256']}"
            )
    if mismatches:
        raise RuntimeError("asset identity mismatch; training not started: " + " | ".join(mismatches))
    return observed


def _git_result(root, *args):
    """Run one read-only Git query and return its completed process."""
    return subprocess.run(
        ["git", "--no-optional-locks", "-C", str(root), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=10,
    )


def _record_git_issue(identity, message, reason):
    git_info = identity["git"]
    git_info["unavailable_reason"] = "; ".join(
        part for part in (git_info["unavailable_reason"], reason) if part
    )
    identity["errors"].append(f"{message}: {reason}")


def collect_runtime_code_identity(root):
    """Collect the code identity used by this run without requiring Git."""
    root = Path(root).resolve()
    identity = {
        "project_root": str(root),
        "working_directory": str(Path.cwd()),
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "collection_status": "complete",
        "git": {
            "available": False,
            "is_worktree": None,
            "worktree_root": None,
            "version": None,
            "head": None,
            "branch": None,
            "detached_head": None,
            "worktree_clean": None,
            "status_short": [],
            "unavailable_reason": None,
        },
        "files": {},
        "errors": [],
    }
    git_info = identity["git"]

    try:
        version = subprocess.run(
            ["git", "--version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=10,
        )
        if version.returncode == 0:
            git_info["available"] = True
            git_info["version"] = version.stdout.strip()
        else:
            reason = version.stderr.strip() or f"git --version exited with {version.returncode}"
            _record_git_issue(identity, "Git unavailable", reason)
    except (OSError, subprocess.SubprocessError) as exc:
        reason = f"{type(exc).__name__}: {exc}"
        _record_git_issue(identity, "Git unavailable", reason)

    if git_info["available"]:
        try:
            repository = _git_result(root, "rev-parse", "--show-toplevel")
            if repository.returncode != 0:
                git_info["is_worktree"] = False
                reason = repository.stderr.strip() or "path is not a Git worktree"
                _record_git_issue(identity, "Git worktree unavailable", reason)
            else:
                git_info["is_worktree"] = True
                git_info["worktree_root"] = repository.stdout.strip()

                head = _git_result(root, "rev-parse", "HEAD")
                if head.returncode == 0 and head.stdout.strip():
                    git_info["head"] = head.stdout.strip()
                else:
                    reason = head.stderr.strip() or f"git rev-parse HEAD exited with {head.returncode}"
                    _record_git_issue(identity, "Git HEAD unavailable", reason)

                branch = _git_result(root, "symbolic-ref", "--short", "-q", "HEAD")
                if branch.returncode == 0:
                    git_info["branch"] = branch.stdout.strip() or None
                    git_info["detached_head"] = False
                elif branch.returncode == 1:
                    git_info["detached_head"] = True
                else:
                    reason = branch.stderr.strip() or f"git symbolic-ref exited with {branch.returncode}"
                    _record_git_issue(identity, "Git branch unavailable", reason)

                status = _git_result(root, "status", "--porcelain=v1", "--untracked-files=all")
                if status.returncode == 0:
                    git_info["status_short"] = status.stdout.splitlines()
                    git_info["worktree_clean"] = not bool(git_info["status_short"])
                else:
                    reason = status.stderr.strip() or f"git status exited with {status.returncode}"
                    _record_git_issue(identity, "Git worktree status unavailable", reason)
        except (OSError, subprocess.SubprocessError) as exc:
            reason = f"{type(exc).__name__}: {exc}"
            _record_git_issue(identity, "Git identity collection failed", reason)

    # Called after build_model so lazy router imports resolve to the loaded files.
    runtime_sources = {
        "compare/dynamic_router.py": getattr(sys.modules.get("compare.dynamic_router"), "__file__", None),
        "compare/gscvit.py": getattr(sys.modules.get("compare.gscvit"), "__file__", None),
        "models/__init__.py": getattr(sys.modules.get("models"), "__file__", None),
        "diagnostics/honghu_single_seed_20261018/diagnostic_train.py": __file__,
    }
    for relative_path in RUNTIME_CODE_FILES:
        source_path = runtime_sources[relative_path]
        path = Path(source_path).resolve() if source_path else None
        record = {
            "relative_path": relative_path,
            "path": str(path) if path else None,
            "matches_project_root": path == (root / relative_path).resolve() if path else False,
            "exists": False,
            "sha256": None,
            "error": None,
        }
        try:
            if path is None:
                raise RuntimeError(f"module source path unavailable for {relative_path}")
            if not path.is_file():
                raise FileNotFoundError(f"required runtime file not found: {path}")
            record["exists"] = True
            record["sha256"] = sha256_file(path)
        except (OSError, RuntimeError, ValueError) as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            identity["errors"].append(f"SHA-256 unavailable for {relative_path}: {record['error']}")
        identity["files"][relative_path] = record
        if not record["matches_project_root"]:
            identity["errors"].append(f"Runtime source does not match project root: {relative_path}: {path}")

    hashes_complete = all(record["sha256"] for record in identity["files"].values())
    git_complete = (
        git_info["available"]
        and git_info["is_worktree"]
        and git_info["head"]
        and git_info["worktree_clean"] is not None
        and (git_info["branch"] is not None or git_info["detached_head"] is True)
    )
    sources_match = all(record["matches_project_root"] for record in identity["files"].values())
    if not hashes_complete or not sources_match or (git_info["available"] and git_info["is_worktree"] and not git_complete):
        identity["collection_status"] = "partial"
    elif not git_complete:
        identity["collection_status"] = "complete_without_git"
    return identity


def build_run_config(
    manifest,
    args,
    assets,
    spatial_enabled,
    seed,
    device,
    runtime_identity,
    history,
    best_checkpoint,
    run_status,
    output=None,
    elapsed_seconds=None,
    nonfinite_events=None,
    actual_command=None,
):
    run_config = {
        "diagnostic_id": manifest["diagnostic_id"],
        "dataset": "HongHu",
        "model_name": args.model_name,
        "spatial_state_enabled": spatial_enabled,
        "seed": seed,
        "split_seed": manifest.get("split_seed"),
        "device": str(device),
        "run_status": run_status,
        "assets_dir": str(assets),
        "assets_file_manifest_sha256": sha256_file(assets / "file_manifest.json"),
        "assets": collect_asset_identity(assets),
        "config": {
            "epochs": 200,
            "batch_size": 32,
            "optimizer": "adamw",
            "learning_rate": 3e-4,
            "weight_decay": 0.005,
            "scheduler": "poly",
            "lr_decay": 0.95,
            "train_ratio": 0.01,
            "val_ratio": 0.01,
            "spectral_groups": 8,
            "route_strength": 0.5,
            "route_temperature": 1.0,
            "router_variant": "full",
            "spatial_state_strength": args.spatial_state_strength,
        },
        "checkpoint_selection": "validation OA max; ties select later validation record; test sealed",
        "best_checkpoint": best_checkpoint,
        "history": history,
        "elapsed_seconds": elapsed_seconds,
        "nonfinite_events": nonfinite_events or [],
        "actual_command": actual_command or [sys.executable, *sys.argv],
        "code_root": str(Path(args.project_root).resolve()),
        "runtime_code_identity": runtime_identity,
        "runtime_environment": {
            "python_version": sys.version,
            "pytorch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
        },
    }
    if best_checkpoint and output is not None:
        checkpoint_path = Path(output) / best_checkpoint
        run_config["best_checkpoint_path"] = str(checkpoint_path.resolve())
        run_config["best_checkpoint_sha256"] = sha256_file(checkpoint_path) if checkpoint_path.is_file() else None
    else:
        run_config["best_checkpoint_path"] = None
        run_config["best_checkpoint_sha256"] = None
    return run_config


def write_run_config(output, run_config):
    (output / "train_config.json").write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")


def metric_triplet(logits, labels, classes=22):
    prediction = logits.argmax(dim=1).detach().cpu().numpy()
    truth = labels.detach().cpu().numpy()
    oa = float(np.mean(prediction == truth) * 100.0)
    per_class = []
    for class_id in range(classes):
        mask = truth == class_id
        per_class.append(float(np.mean(prediction[mask] == truth[mask]) * 100.0) if mask.any() else float("nan"))
    aa = float(np.nanmean(per_class) if per_class else float("nan"))
    confusion = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(confusion, (truth, prediction), 1)
    total = confusion.sum()
    po = np.trace(confusion) / total
    pe = np.dot(confusion.sum(axis=1), confusion.sum(axis=0)) / (total * total)
    kappa = float((po - pe) / (1.0 - pe) * 100.0) if pe < 1.0 else float("nan")
    return {"oa": oa, "aa": aa, "kappa": kappa, "per_class_aa": per_class}


def build_split_tensors(assets):
    pca_scene = np.load(assets / "pca_scene.npz")["pca_scene"]
    index_map = np.load(assets / "sample_index_map.npz")
    coordinates = index_map["coordinates"]
    labels = index_map["labels"].astype(np.int64)
    split = np.load(assets / "split_indices.npz")
    margin = 3
    padded = np.pad(pca_scene, ((margin, margin), (margin, margin), (0, 0)), mode="constant")

    def make(indices):
        patches = np.empty((len(indices), 7, 7, 15), dtype=np.float32)
        for output_index, sample_index in enumerate(indices):
            row, col = coordinates[int(sample_index)]
            patches[output_index] = padded[row : row + 7, col : col + 7]
        patches = patches.transpose(0, 3, 1, 2)[:, None]
        return torch.from_numpy(patches), torch.from_numpy(labels[indices])

    return make(split["train"]), make(split["validation"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--assets-dir", required=True)
    parser.add_argument("--model-name", choices=sorted(MODEL_FLAGS), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--spatial-state-strength",
        type=float,
        default=0.5,
        help="Spatial residual strength; default preserves the original 0.5 behavior.",
    )
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    assets = Path(args.assets_dir).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    validate_asset_identity(assets)
    manifest = json.loads((assets / "split_manifest.json").read_text(encoding="utf-8"))
    if manifest["diagnostic_id"] != "honghu_single_seed_20261018":
        raise ValueError("unexpected diagnostic asset manifest")
    if args.spatial_state_strength <= 0:
        raise ValueError("spatial-state-strength must be positive")

    seed = 20261017
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    spatial_enabled = MODEL_FLAGS[args.model_name]
    history = []
    best_checkpoint = None
    nonfinite_events = []
    start_time = time.monotonic()
    (train_x, train_y), (val_x, val_y) = build_split_tensors(assets)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(train_x, train_y), batch_size=32, shuffle=True, generator=generator, num_workers=0
    )
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=32, shuffle=False, num_workers=0)

    model = build_model(
        args.model_name,
        15,
        22,
        7,
        spectral_groups=8,
        route_strength=0.5,
        route_temperature=1.0,
        router_variant="full",
        spatial_state_enabled=spatial_enabled,
        spatial_state_strength=args.spatial_state_strength,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.005)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: 0.95 ** ((epoch - 1) / 10 + 1)
    )
    criterion = torch.nn.CrossEntropyLoss()
    best_oa = 0.0
    runtime_identity = collect_runtime_code_identity(root)
    run_config = build_run_config(
        manifest, args, assets, spatial_enabled, seed, device,
        runtime_identity, history, best_checkpoint, "started",
        output=output,
        elapsed_seconds=0.0,
        nonfinite_events=nonfinite_events,
    )
    invalid_sources = [
        relative_path
        for relative_path, record in runtime_identity["files"].items()
        if not record["sha256"] or not record["matches_project_root"]
    ]
    if invalid_sources:
        run_config["run_status"] = "not_started_runtime_identity_incomplete"
    write_run_config(output, run_config)
    if invalid_sources:
        raise RuntimeError(
            "runtime source identity invalid; training not started; see train_config.json: "
            + ", ".join(invalid_sources)
        )
    for epoch in range(1, 201):
        model.train()
        train_loss = 0.0
        train_count = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            if not torch.isfinite(logits).all():
                event = {"epoch": epoch, "stage": "train_logits", "message": "non-finite model output"}
                nonfinite_events.append(event)
                run_config["run_status"] = "failed_nonfinite"
                run_config["nonfinite_events"] = nonfinite_events
                run_config["elapsed_seconds"] = time.monotonic() - start_time
                write_run_config(output, run_config)
                raise FloatingPointError(json.dumps(event))
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                event = {"epoch": epoch, "stage": "train_loss", "message": "non-finite loss"}
                nonfinite_events.append(event)
                run_config["run_status"] = "failed_nonfinite"
                run_config["nonfinite_events"] = nonfinite_events
                run_config["elapsed_seconds"] = time.monotonic() - start_time
                write_run_config(output, run_config)
                raise FloatingPointError(json.dumps(event))
            loss.backward()
            if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                   for parameter in model.parameters()):
                event = {"epoch": epoch, "stage": "train_gradient", "message": "non-finite gradient"}
                nonfinite_events.append(event)
                run_config["run_status"] = "failed_nonfinite"
                run_config["nonfinite_events"] = nonfinite_events
                run_config["elapsed_seconds"] = time.monotonic() - start_time
                write_run_config(output, run_config)
                raise FloatingPointError(json.dumps(event))
            optimizer.step()
            train_loss += float(loss.detach()) * labels.numel()
            train_count += labels.numel()
        scheduler.step()
        record = {
            "epoch": epoch,
            "train_loss": train_loss / train_count,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        if epoch % 2 == 0:
            model.eval()
            logits_parts, label_parts = [], []
            with torch.no_grad():
                for inputs, labels in val_loader:
                    batch_logits = model(inputs.to(device))
                    if not torch.isfinite(batch_logits).all():
                        event = {"epoch": epoch, "stage": "validation_logits", "message": "non-finite model output"}
                        nonfinite_events.append(event)
                        run_config["run_status"] = "failed_nonfinite"
                        run_config["nonfinite_events"] = nonfinite_events
                        run_config["elapsed_seconds"] = time.monotonic() - start_time
                        write_run_config(output, run_config)
                        raise FloatingPointError(json.dumps(event))
                    logits_parts.append(batch_logits.cpu())
                    label_parts.append(labels)
            metrics = metric_triplet(torch.cat(logits_parts), torch.cat(label_parts))
            if not all(np.isfinite(value) for value in (metrics["oa"], metrics["aa"], metrics["kappa"])):
                event = {"epoch": epoch, "stage": "validation_metrics", "message": "non-finite validation metric"}
                nonfinite_events.append(event)
                run_config["run_status"] = "failed_nonfinite"
                run_config["nonfinite_events"] = nonfinite_events
                run_config["elapsed_seconds"] = time.monotonic() - start_time
                write_run_config(output, run_config)
                raise FloatingPointError(json.dumps(event))
            record["validation"] = metrics
            # Match main_train.py: OA maximum, and >= means later ties replace earlier ones.
            if metrics["oa"] >= best_oa:
                best_oa = metrics["oa"]
                checkpoint = output / f"checkpoint_epoch_{epoch:03d}.pth"
                torch.save(model.state_dict(), checkpoint)
                best_checkpoint = checkpoint.name
                record["checkpoint_saved"] = True
                record["checkpoint_sha256"] = sha256_file(checkpoint)
            else:
                record["checkpoint_saved"] = False
        history.append(record)
    run_config = build_run_config(
        manifest,
        args,
        assets,
        spatial_enabled,
        seed,
        device,
        runtime_identity,
        history,
        best_checkpoint,
        "completed",
        output=output,
        elapsed_seconds=time.monotonic() - start_time,
        nonfinite_events=nonfinite_events,
    )
    write_run_config(output, run_config)
    print(json.dumps({"output_dir": str(output), "best_checkpoint": best_checkpoint, "device": str(device)}, indent=2))


if __name__ == "__main__":
    main()
