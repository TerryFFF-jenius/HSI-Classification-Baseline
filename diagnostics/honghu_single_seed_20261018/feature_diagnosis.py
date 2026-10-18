"""Collect fixed validation-batch routing features for one reviewed checkpoint."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from models import build_model


MODEL_FLAGS = {
    "gscvit_tssr": False,
    "gscvit_dssr": False,
    "gscvit_spatial": True,
    "gscvit_dssr_spatial": True,
}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_stats(value):
    if value is None:
        return {"status": "missing"}
    value = value.detach().cpu()
    result = {
        "status": "available",
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "numel": int(value.numel()),
        "finite": bool(torch.isfinite(value).all().item()) if value.is_floating_point() or value.is_complex() else True,
    }
    if value.numel():
        numeric = value.float() if not (value.is_floating_point() or value.is_complex()) else value
        result.update({
            "min": float(numeric.min()), "max": float(numeric.max()),
            "mean": float(numeric.mean()), "std": float(numeric.std(unbiased=False)),
            "rms": float(torch.sqrt(torch.mean(numeric.square()))),
        })
    return result


def sample_rms(value):
    return torch.sqrt(torch.mean(value.square(), dim=tuple(range(1, value.dim()))))


def cosine_per_sample(left, right):
    left = left.flatten(1)
    right = right.flatten(1)
    return torch.nn.functional.cosine_similarity(left, right, dim=1, eps=1e-12)


def collect(model, model_name, inputs):
    captured = {}

    def adapter(route_feat):
        captured["route_feat"] = route_feat
        routed, alpha, beta = model.dynamic_router.route(route_feat)
        captured["tssr_output"] = routed
        captured["alpha"] = alpha
        captured["beta"] = beta
        if model_name == "gscvit_tssr":
            return routed
        if model_name == "gscvit_dssr":
            spectral = model.state_router(routed, alpha, beta)
            captured["spectral_state_output"] = spectral
            captured["spectral_permutation"] = model.state_router.last_permutation
            captured["spectral_inverse_permutation"] = model.state_router.last_inverse_permutation
            return spectral
        if model_name == "gscvit_spatial":
            spatial = model.spatial_state_router(routed, beta)
            captured["spatial_state_output"] = spatial
            captured["spatial_permutation"] = model.spatial_state_router.last_permutation
            captured["spatial_inverse_permutation"] = model.spatial_state_router.last_inverse_permutation
            return spatial
        spectral = model.state_router(routed, alpha, beta)
        captured["spectral_state_output"] = spectral
        captured["spectral_permutation"] = model.state_router.last_permutation
        captured["spectral_inverse_permutation"] = model.state_router.last_inverse_permutation
        spatial = model.spatial_state_router(spectral, beta)
        captured["spatial_state_output"] = spatial
        captured["spatial_permutation"] = model.spatial_state_router.last_permutation
        captured["spatial_inverse_permutation"] = model.spatial_state_router.last_inverse_permutation
        return spatial

    with torch.no_grad():
        features = model.net.forward_features(inputs, pre_gssa_adapter=adapter)
        captured["logits"] = model.net.mlp_head(features)
    return captured


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--model-name", choices=sorted(MODEL_FLAGS), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    assets = Path(args.assets_dir).resolve()
    run_dir = Path(args.run_dir).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    train_config = json.loads((run_dir / "train_config.json").read_text(encoding="utf-8"))
    if train_config["model_name"] != args.model_name:
        raise ValueError("run model name does not match requested model")
    checkpoint_name = train_config.get("best_checkpoint")
    if not checkpoint_name:
        raise ValueError("run has no selected checkpoint")
    checkpoint = run_dir / checkpoint_name
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    batch = np.load(assets / "validation_batch.npz")
    inputs = torch.from_numpy(batch["inputs"]).to(args.device)
    model = build_model(
        args.model_name, 15, 22, 7, spectral_groups=8, route_strength=0.5,
        route_temperature=1.0, router_variant="full",
        spatial_state_enabled=MODEL_FLAGS[args.model_name], spatial_state_strength=0.5,
    ).to(args.device)
    model.load_state_dict(torch.load(checkpoint, map_location=args.device), strict=True)
    model.eval()
    values = collect(model, args.model_name, inputs)

    stats = {name: tensor_stats(value) for name, value in values.items()}
    residuals = {}
    if values.get("spectral_state_output") is not None:
        residuals["Rs"] = values["spectral_state_output"] - values["tssr_output"]
    if values.get("spatial_state_output") is not None and values.get("spectral_state_output") is not None:
        residuals["Rp"] = values["spatial_state_output"] - values["spectral_state_output"]
    elif values.get("spatial_state_output") is not None:
        residuals["spatial_residual_from_tssr"] = values["spatial_state_output"] - values["tssr_output"]
    for name, value in residuals.items():
        stats[name] = tensor_stats(value)
    ratios = {}
    if "Rs" in residuals:
        ratios["Rs_over_tssr_input"] = (sample_rms(residuals["Rs"]) / (sample_rms(values["tssr_output"]) + 1e-12)).tolist()
    if "Rp" in residuals:
        ratios["Rp_over_spectral_output"] = (sample_rms(residuals["Rp"]) / (sample_rms(values["spectral_state_output"]) + 1e-12)).tolist()
    if "Rs" in residuals and "Rp" in residuals:
        ratios["Rp_over_Rs"] = (sample_rms(residuals["Rp"]) / (sample_rms(residuals["Rs"]) + 1e-12)).tolist()
        ratios["cosine_Rs_Rp"] = cosine_per_sample(residuals["Rs"], residuals["Rp"]).tolist()

    arrays = {name: value.detach().cpu().numpy() for name, value in values.items() if torch.is_tensor(value)}
    arrays.update({name: value.detach().cpu().numpy() for name, value in residuals.items()})
    np.savez_compressed(output / "feature_diagnosis.npz", **arrays)
    report = {
        "diagnostic_id": "honghu_single_seed_20261018",
        "model_name": args.model_name,
        "device": args.device,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "assets_file_manifest_sha256": sha256_file(assets / "file_manifest.json"),
        "validation_batch_sample_indices": batch["sample_indices"].tolist(),
        "stats": stats,
        "ratios": ratios,
        "missing_observations": {
            "Rs": "spectral state output is not present for gscvit_tssr" if "Rs" not in residuals else None,
            "Rp": "spectral state output is not present for gscvit_spatial" if "Rp" not in residuals and args.model_name == "gscvit_spatial" else None,
            "cosine_Rs_Rp": "requires both spectral and spatial state outputs" if "cosine_Rs_Rp" not in ratios else None,
            "spectral_permutation": "not applicable because this model has no spectral state route" if "spectral_permutation" not in values and args.model_name in {"gscvit_tssr", "gscvit_spatial"} else None,
            "spatial_permutation": "not applicable because this model has no spatial state route" if "spatial_permutation" not in values and args.model_name in {"gscvit_tssr", "gscvit_dssr"} else None,
        },
        "argsort_note": "permutation is discrete and non-differentiable; this script is inference-only",
    }
    (output / "feature_diagnosis.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output), "checkpoint_sha256": report["checkpoint_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
