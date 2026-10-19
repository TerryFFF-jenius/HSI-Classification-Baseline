"""Compare adaptation and baseline validation histories without test access."""

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED_BASELINE_SHA256 = "606ef81f050cda8171dd0effe05f77ef6a24c12289942f5e076f253cf0183304"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_run(path):
    config_path = path / "train_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("model_name") != "gscvit_dssr_spatial":
        raise ValueError(f"unexpected model in {config_path}")
    if config.get("config", {}).get("epochs") != 200:
        raise ValueError(f"unexpected epoch budget in {config_path}")
    if config.get("config", {}).get("batch_size") != 32:
        raise ValueError(f"unexpected batch size in {config_path}")
    return config


def validation_records(config):
    return [record for record in config.get("history", []) if "validation" in record]


def best_record(config):
    records = validation_records(config)
    if not records:
        return None
    return max(records, key=lambda record: (record["validation"]["oa"], record["epoch"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--adaptation-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    baseline_dir = Path(args.baseline_dir).resolve()
    adaptation_dir = Path(args.adaptation_dir).resolve()
    baseline = load_run(baseline_dir)
    adaptation = load_run(adaptation_dir)
    if adaptation.get("config", {}).get("spatial_state_strength") != 0.25:
        raise ValueError("adaptation run is not spatial_state_strength=0.25")
    if baseline.get("config", {}).get("spatial_state_strength") != 0.5:
        raise ValueError("baseline run is not spatial_state_strength=0.5")
    if adaptation.get("run_status") != "completed":
        raise ValueError("adaptation run is not completed")
    baseline_checkpoint = baseline.get("best_checkpoint")
    if not baseline_checkpoint:
        raise ValueError("baseline best checkpoint is missing")
    checkpoint_path = baseline_dir / baseline_checkpoint
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    observed = sha256_file(checkpoint_path)
    if observed != EXPECTED_BASELINE_SHA256:
        raise ValueError(f"baseline checkpoint SHA-256 mismatch: {observed}")
    adaptation_checkpoint = adaptation.get("best_checkpoint")
    if not adaptation_checkpoint or not (adaptation_dir / adaptation_checkpoint).is_file():
        raise FileNotFoundError("adaptation best checkpoint is missing")
    base_records = {record["epoch"]: record for record in validation_records(baseline)}
    adapt_records = {record["epoch"]: record for record in validation_records(adaptation)}
    shared_epochs = sorted(set(base_records) & set(adapt_records))
    curve = []
    for epoch in shared_epochs:
        base_metrics = base_records[epoch]["validation"]
        adapt_metrics = adapt_records[epoch]["validation"]
        curve.append({
            "epoch": epoch,
            "baseline_oa": base_metrics["oa"],
            "adaptation_oa": adapt_metrics["oa"],
            "delta_oa": adapt_metrics["oa"] - base_metrics["oa"],
            "baseline_aa": base_metrics["aa"],
            "adaptation_aa": adapt_metrics["aa"],
            "delta_aa": adapt_metrics["aa"] - base_metrics["aa"],
            "baseline_kappa": base_metrics["kappa"],
            "adaptation_kappa": adapt_metrics["kappa"],
            "delta_kappa": adapt_metrics["kappa"] - base_metrics["kappa"],
        })
    base_best = best_record(baseline)
    adapt_best = best_record(adaptation)
    result = {
        "comparison_scope": "validation only; test sealed and not read",
        "baseline_dir": str(baseline_dir),
        "adaptation_dir": str(adaptation_dir),
        "baseline_checkpoint": {
            "name": baseline_checkpoint,
            "expected_sha256": EXPECTED_BASELINE_SHA256,
            "observed_sha256": sha256_file(baseline_dir / baseline_checkpoint) if baseline_checkpoint and (baseline_dir / baseline_checkpoint).is_file() else None,
        },
        "adaptation_checkpoint": {
            "name": adaptation.get("best_checkpoint"),
            "sha256": sha256_file(adaptation_dir / adaptation["best_checkpoint"]) if adaptation.get("best_checkpoint") and (adaptation_dir / adaptation["best_checkpoint"]).is_file() else None,
        },
        "shared_validation_epochs": shared_epochs,
        "best_validation": {
            "baseline": base_best,
            "adaptation": adapt_best,
            "delta_oa": adapt_best["validation"]["oa"] - base_best["validation"]["oa"] if base_best and adapt_best else None,
            "delta_aa": adapt_best["validation"]["aa"] - base_best["validation"]["aa"] if base_best and adapt_best else None,
            "delta_kappa": adapt_best["validation"]["kappa"] - base_best["validation"]["kappa"] if base_best and adapt_best else None,
            "best_epoch_delta": adapt_best["epoch"] - base_best["epoch"] if base_best and adapt_best else None,
        },
        "curve": curve,
        "nonfinite_events": {
            "baseline": baseline.get("nonfinite_events", []),
            "adaptation": adaptation.get("nonfinite_events", []),
        },
    }
    output = Path(args.output).resolve()
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "shared_validation_epochs": len(shared_epochs)}, indent=2))


if __name__ == "__main__":
    main()
