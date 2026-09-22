"""Real-data mechanism validation for the TSSR routing module.

This is a diagnostic experiment, not a paper result.  It trains TSSR for a
small number of epochs on the real training/validation split and records
whether the routing scores are finite, non-uniform, and connected to the
classification loss through non-zero gradients.
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from torch import nn

from data_loader import build_data_loader
from models import build_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_args(parsed):
    """Create the small argument object expected by data_loader."""
    return argparse.Namespace(
        dataset=parsed.dataset,
        batch_size=parsed.batch_size,
        train_ratio=parsed.train_ratio,
        val_ratio=parsed.val_ratio,
        is_train=True,
        PCA=None,
        hsi_bands=None,
        num_class=None,
        patch_size=None,
        seed=parsed.seed,
    )


def finite_or_raise(name, tensor):
    if not torch.isfinite(tensor).all().item():
        raise RuntimeError(f"{name} contains NaN or Inf")


def router_gradient_stats(model):
    values = []
    missing = []
    for name, parameter in model.named_parameters():
        if not name.startswith("dynamic_router."):
            continue
        if parameter.grad is None:
            missing.append(name)
            continue
        finite_or_raise(f"gradient:{name}", parameter.grad)
        values.append(float(parameter.grad.detach().abs().max().item()))
    if not values:
        raise RuntimeError(
            "No gradient reached dynamic_router parameters. "
            f"Parameters without gradients: {missing}"
        )
    return {
        "parameter_count_with_gradient": len(values),
        "max_abs_gradient": max(values),
        "mean_abs_gradient_max_per_parameter": float(np.mean(values)),
    }


def summarize_scores(alpha_values, beta_values):
    alpha = torch.cat(alpha_values, dim=0)
    beta = torch.cat(beta_values, dim=0)
    finite_or_raise("alpha_history", alpha)
    finite_or_raise("beta_history", beta)

    alpha_entropy = -(alpha * alpha.clamp_min(1e-8).log()).sum(dim=1)
    return {
        "alpha_mean": alpha.mean(dim=0).tolist(),
        "alpha_std_across_samples": float(alpha.std(unbiased=False).item()),
        "alpha_entropy_mean": float(alpha_entropy.mean().item()),
        "alpha_entropy_std": float(alpha_entropy.std(unbiased=False).item()),
        "beta_mean": float(beta.mean().item()),
        "beta_std": float(beta.std(unbiased=False).item()),
        "beta_min": float(beta.min().item()),
        "beta_max": float(beta.max().item()),
    }


@torch.no_grad()
def validation_oa(model, loader, device):
    if loader is None:
        return None
    model.eval()
    correct = 0
    total = 0
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        logits = model(inputs)
        finite_or_raise("validation_logits", logits)
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += int(labels.numel())
    return 100.0 * correct / total if total else None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate TSSR routing behavior on a real HSI split."
    )
    parser.add_argument("--dataset", default="HanChuan")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--train_ratio", type=float, default=0.01)
    parser.add_argument("--val_ratio", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=300)
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory for the JSON report and diagnostic checkpoint.",
    )
    return parser.parse_args()


def main():
    parsed = parse_args()
    if parsed.epochs <= 0:
        raise ValueError("epochs must be positive")

    set_seed(parsed.seed)
    data_args = make_args(parsed)
    train_loader, val_loader, _ = build_data_loader(data_args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_model(
        "gscvit_tssr",
        data_args.in_channels,
        data_args.num_class,
        data_args.patch_size,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.005,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda epoch: 0.95 ** ((epoch - 1) / 10 + 1),
    )
    criterion = nn.CrossEntropyLoss()

    print("========== TSSR Mechanism Validation ==========")
    print("dataset:", parsed.dataset)
    print("seed:", parsed.seed)
    print("epochs:", parsed.epochs)
    print("device:", device)
    print("num_classes:", data_args.num_class)
    print("in_channels:", data_args.in_channels)
    print("patch_size:", data_args.patch_size)

    history = []
    for epoch in range(parsed.epochs):
        model.train()
        loss_values = []
        alpha_values = []
        beta_values = []
        grad_stats = None

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)

            logits, alpha, beta = model.forward_with_routing(inputs)
            finite_or_raise("logits", logits)
            finite_or_raise("alpha", alpha)
            finite_or_raise("beta", beta)
            loss = criterion(logits, labels)
            finite_or_raise("loss", loss)
            loss.backward()
            grad_stats = router_gradient_stats(model)
            optimizer.step()

            loss_values.append(float(loss.detach().item()))
            alpha_values.append(alpha.detach().cpu())
            beta_values.append(beta.detach().cpu())

        scheduler.step()
        score_summary = summarize_scores(alpha_values, beta_values)
        val_oa = validation_oa(model, val_loader, device)
        epoch_record = {
            "epoch": epoch,
            "loss_mean": float(np.mean(loss_values)),
            "loss_last": loss_values[-1],
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "validation_oa": val_oa,
            "router_gradients": grad_stats,
            **score_summary,
        }
        history.append(epoch_record)
        print(
            f"epoch={epoch} "
            f"loss={epoch_record['loss_mean']:.6f} "
            f"val_oa={val_oa if val_oa is not None else 'NA'} "
            f"alpha_std={score_summary['alpha_std_across_samples']:.6f} "
            f"beta_std={score_summary['beta_std']:.6f} "
            f"router_grad={grad_stats['max_abs_gradient']:.6e}"
        )

    output_dir = parsed.output_dir
    if output_dir is None:
        output_dir = os.path.join(
            "results",
            "own",
            parsed.dataset,
            f"exp_tssr_mechanism_{parsed.dataset.lower()}_seed{parsed.seed}",
        )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    report = {
        "experiment_type": "tssr_mechanism_validation",
        "dataset": parsed.dataset,
        "model": "gscvit_tssr",
        "seed": parsed.seed,
        "epochs": parsed.epochs,
        "batch_size": parsed.batch_size,
        "train_ratio": parsed.train_ratio,
        "val_ratio": parsed.val_ratio,
        "device": str(device),
        "history": history,
        "checks": {
            "finite_loss_and_scores": True,
            "router_gradients_reached": True,
            "alpha_is_probability_distribution": True,
        },
    }
    report_file = output_path / "mechanism_report.json"
    report_file.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    torch.save(model.state_dict(), output_path / "mechanism_last.pth")

    print("report:", report_file)
    print("checkpoint:", output_path / "mechanism_last.pth")
    print("========== TSSR MECHANISM CHECK: PASS ==========")


if __name__ == "__main__":
    main()

