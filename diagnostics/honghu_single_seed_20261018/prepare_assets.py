"""Prepare one new, shared HongHu diagnostic split and PCA asset.

This script intentionally mirrors data_loader.py: PCA is fitted on the full
scene before labeled-pixel indexing and splitting. It does not import models
or perform inference/training.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


DATA_IMAGE = Path("oridata/HongHu/WHU_Hi_HongHu.mat")
DATA_LABEL = Path("oridata/HongHu/WHU_Hi_HongHu_gt.mat")
PCA_COMPONENTS = 15
PCA_RANDOM_STATE = 20261017
SPLIT_SEED = 20261018
TRAIN_RATIO = 0.01
VAL_RATIO = 0.01
PATCH_SIZE = 7


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def git_snapshot(root):
    def run(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()

    status = run("status", "--short")
    return {
        "head": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "status_short": status.splitlines(),
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
    }


def make_patches(pca_scene, labels, coordinates):
    margin = PATCH_SIZE // 2
    padded = np.pad(
        pca_scene,
        ((margin, margin), (margin, margin), (0, 0)),
        mode="constant",
        constant_values=0,
    )
    patches = np.empty((len(coordinates), PATCH_SIZE, PATCH_SIZE, PCA_COMPONENTS), dtype=np.float32)
    for output_index, (row, col) in enumerate(coordinates):
        patches[output_index] = padded[
            row : row + PATCH_SIZE, col : col + PATCH_SIZE, :
        ]
    # Match data_loader.py's final model input layout: [B, 1, C, H, W].
    return patches.transpose(0, 3, 1, 2)[:, None, :, :, :].astype(np.float32, copy=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    image_path = root / DATA_IMAGE
    label_path = root / DATA_LABEL
    image = sio.loadmat(image_path)["WHU_Hi_HongHu"]
    labels = sio.loadmat(label_path)["WHU_Hi_HongHu_gt"]
    if image.shape[:2] != labels.shape or image.shape[2] != 270:
        raise ValueError(f"unexpected HongHu shapes: image={image.shape}, labels={labels.shape}")

    # Exact current baseline ordering: flatten labeled pixels in row-major order.
    coordinates = np.argwhere(labels.reshape(-1) > 0)
    rows, cols = np.unravel_index(coordinates[:, 0], labels.shape)
    coordinates = np.column_stack((rows, cols)).astype(np.int64)
    y = labels[rows, cols].astype(np.int64) - 1
    all_indices = np.arange(len(coordinates), dtype=np.int64)
    test_ratio = 1.0 - TRAIN_RATIO - VAL_RATIO
    train_val, test = train_test_split(
        all_indices,
        test_size=test_ratio,
        stratify=y,
        random_state=SPLIT_SEED,
    )
    validation_fraction = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    train, validation = train_test_split(
        train_val,
        test_size=validation_fraction,
        stratify=y[train_val],
        random_state=SPLIT_SEED,
    )
    split_indices = {
        "train": np.sort(train),
        "validation": np.sort(validation),
        "test": np.sort(test),
    }

    # Exact current baseline PCA semantics: fit_transform over the full scene.
    flat = image.reshape(-1, image.shape[2])
    pca = PCA(n_components=PCA_COMPONENTS, whiten=True, random_state=PCA_RANDOM_STATE)
    pca_scene = pca.fit_transform(flat).reshape(image.shape[0], image.shape[1], PCA_COMPONENTS).astype(np.float32)

    np.savez_compressed(
        output / "split_indices.npz",
        train=split_indices["train"],
        validation=split_indices["validation"],
        test=split_indices["test"],
    )
    np.savez_compressed(
        output / "sample_index_map.npz",
        coordinates=coordinates,
        labels=y,
    )
    np.savez_compressed(
        output / "pca_transform.npz",
        components=pca.components_.astype(np.float64),
        mean=pca.mean_.astype(np.float64),
        explained_variance=pca.explained_variance_.astype(np.float64),
        explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float64),
        singular_values=pca.singular_values_.astype(np.float64),
    )
    np.savez_compressed(output / "pca_scene.npz", pca_scene=pca_scene)

    validation_batch_indices = split_indices["validation"][: min(32, len(split_indices["validation"]))]
    validation_batch = make_patches(pca_scene, labels, coordinates[validation_batch_indices])
    np.savez_compressed(
        output / "validation_batch.npz",
        inputs=validation_batch,
        sample_indices=validation_batch_indices,
        labels=y[validation_batch_indices],
    )

    class_counts = {}
    for name, indices in split_indices.items():
        class_counts[name] = {
            str(int(class_id)): int(np.sum(y[indices] == class_id))
            for class_id in range(int(y.max()) + 1)
        }
    manifest = {
        "diagnostic_id": "honghu_single_seed_20261018",
        "dataset": "HongHu",
        "source": {
            "image_path": str(DATA_IMAGE),
            "label_path": str(DATA_LABEL),
            "image_sha256": sha256_file(image_path),
            "label_sha256": sha256_file(label_path),
            "image_shape": list(image.shape),
            "label_shape": list(labels.shape),
        },
        "index_space": "nonzero_label_pixels_row_major",
        "split_seed": SPLIT_SEED,
        "train_ratio": TRAIN_RATIO,
        "val_ratio": VAL_RATIO,
        "test_ratio": test_ratio,
        "counts": {name: int(len(indices)) for name, indices in split_indices.items()},
        "class_counts_zero_based": class_counts,
        "preprocessing": {
            "pca_components": PCA_COMPONENTS,
            "pca_whiten": True,
            "pca_random_state": PCA_RANDOM_STATE,
            "pca_fit_scope": "full_scene_before_patch_creation_and_split",
            "patch_size": PATCH_SIZE,
            "zero_padding": "padWithZeros, margin=patch_size//2",
            "remove_zero_labels": True,
            "transpose": "NHWC to NCHW with singleton channel [B,H,W,C] -> [B,1,C,H,W]",
        },
        "training_seed": 20261017,
        "git_snapshot": git_snapshot(root),
    }
    json_dump(output / "split_manifest.json", manifest)
    files = {}
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "file_manifest.json":
            files[path.name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    json_dump(output / "file_manifest.json", files)
    print(json.dumps({"output_dir": str(output), "files": files, "counts": manifest["counts"]}, indent=2))


if __name__ == "__main__":
    main()
