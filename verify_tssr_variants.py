"""Forward/backward smoke checks for the TSSR ablation variants.

This script intentionally uses synthetic tensors.  It verifies the routing
contract before any dataset training is started:

* ``full`` learns both spectral and spatial scores;
* ``no_spectral`` uses a uniform spectral distribution;
* ``no_spatial`` uses the neutral spatial gain one.
"""

import argparse
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

from models import build_model


VARIANTS = ("full", "no_spectral", "no_spatial")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_channels", type=int, default=15)
    parser.add_argument("--num_classes", type=int, default=16)
    parser.add_argument("--patch_size", type=int, default=7)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def check_variant(variant, args, device):
    torch.manual_seed(123)
    model = build_model(
        "gscvit_tssr",
        args.in_channels,
        args.num_classes,
        args.patch_size,
        spectral_groups=8,
        route_strength=0.5,
        route_temperature=1.0,
        router_variant=variant,
    ).to(device)
    model.train()

    inputs = torch.randn(
        args.batch_size,
        1,
        args.in_channels,
        args.patch_size,
        args.patch_size,
        device=device,
    )
    logits, alpha, beta = model.forward_with_routing(inputs)
    loss = logits.square().mean()
    loss.backward()

    if logits.shape != (args.batch_size, args.num_classes):
        raise AssertionError(
            f"{variant}: unexpected logits shape {tuple(logits.shape)}"
        )
    if not torch.isfinite(logits).all():
        raise AssertionError(f"{variant}: logits contain non-finite values")
    if not torch.isfinite(alpha).all() or not torch.isfinite(beta).all():
        raise AssertionError(f"{variant}: routing scores contain non-finite values")

    alpha_sum = alpha.sum(dim=1)
    if not torch.allclose(alpha_sum, torch.ones_like(alpha_sum), atol=1e-5):
        raise AssertionError(f"{variant}: alpha is not a probability distribution")

    router = model.dynamic_router
    if variant == "no_spectral":
        expected = torch.full_like(alpha, 1.0 / router.num_groups)
        if not torch.allclose(alpha, expected, atol=1e-6):
            raise AssertionError(f"{variant}: alpha is not uniform")
    elif variant == "no_spatial":
        if not torch.allclose(beta, torch.ones_like(beta), atol=1e-6):
            raise AssertionError(f"{variant}: beta is not neutral")

    print(
        f"[PASS] variant={variant} "
        f"logits={tuple(logits.shape)} "
        f"alpha={tuple(alpha.shape)} "
        f"beta={tuple(beta.shape)}"
    )


def main():
    args = parse_args()
    device = torch.device(args.device)
    for variant in VARIANTS:
        check_variant(variant, args, device)


if __name__ == "__main__":
    main()
