"""Tensor-level verification for the GSC-ViT TSSR model.

Run from the repository root with::

    python verify_tssr.py

The OpenMP compatibility variable must be set before importing torch on some
Windows/Anaconda installations.  It is an unsafe workaround for duplicate
OpenMP runtimes, but this script only performs a small local verification.
"""

import os

# Some Windows Anaconda environments load two copies of Intel OpenMP while
# importing torch.  Set this before importing torch so the test can start.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

from models import build_model


def assert_finite(name, tensor):
    """Fail with a useful message if a tensor contains NaN or Inf."""
    assert torch.isfinite(tensor).all().item(), f"{name} contains NaN/Inf"


def main():
    torch.manual_seed(300)

    batch_size = 2
    in_channels = 15
    num_classes = 16
    patch_size = 7

    print("========== TSSR Tensor Verification ==========")
    print("torch:", torch.__version__)
    print("device: cpu")

    model = build_model(
        model_name="gscvit_tssr",
        in_channels=in_channels,
        num_classes=num_classes,
        patch_size=patch_size,
    )
    model.eval()

    x = torch.randn(
        batch_size,
        1,
        in_channels,
        patch_size,
        patch_size,
    )
    assert_finite("input", x)
    print("Input shape:", tuple(x.shape))

    # -------------------------------------------------------------
    # 1. Normal classification forward
    # -------------------------------------------------------------
    with torch.no_grad():
        logits = model(x)

    print("1. Logits shape:", tuple(logits.shape))
    assert logits.shape == (batch_size, num_classes)
    assert_finite("logits", logits)

    # -------------------------------------------------------------
    # 2. Forward with routing diagnostics
    # -------------------------------------------------------------
    with torch.no_grad():
        routed_logits, alpha, beta = model.forward_with_routing(x)

    print("2. Routed logits shape:", tuple(routed_logits.shape))
    print("3. Alpha shape:", tuple(alpha.shape))
    print("4. Beta shape:", tuple(beta.shape))

    assert routed_logits.shape == (batch_size, num_classes)
    assert alpha.shape == (batch_size, 8)
    assert beta.shape == (batch_size, 1, 9, 9)
    assert_finite("routed_logits", routed_logits)
    assert_finite("alpha", alpha)
    assert_finite("beta", beta)

    # alpha is a softmax distribution over spectral groups.
    alpha_sum_error = (alpha.sum(dim=1) - 1.0).abs().max().item()
    print("5. Max alpha sum error:", alpha_sum_error)
    assert alpha_sum_error < 1e-5

    # The spatial gate is scaled to the interval (0, 2).
    print(
        "6. Beta range:",
        float(beta.min().item()),
        "to",
        float(beta.max().item()),
    )
    assert beta.min().item() >= 0.0
    assert beta.max().item() <= 2.0

    # -------------------------------------------------------------
    # 3. Confirm the adapter receives [B, 64, 9, 9] and preserves shape
    # -------------------------------------------------------------
    with torch.no_grad():
        final_feat, pre_gssa_feat = model.forward_features(
            x,
            pre_gssa_adapter=model.dynamic_router,
            return_pre_gssa=True,
        )

    print("7. Pre-GSSA feature shape:", tuple(pre_gssa_feat.shape))
    print("8. Routed feature shape:", tuple(final_feat.shape))
    assert pre_gssa_feat.shape == (batch_size, 64, 9, 9)
    assert final_feat.shape == (batch_size, 64, 9, 9)
    assert_finite("pre_gssa_feat", pre_gssa_feat)
    assert_finite("routed final_feat", final_feat)

    # -------------------------------------------------------------
    # 4. Training-mode backward pass and gradient checks
    # -------------------------------------------------------------
    train_model = build_model(
        model_name="gscvit_tssr",
        in_channels=in_channels,
        num_classes=num_classes,
        patch_size=patch_size,
    )
    train_model.train()

    train_x = torch.randn_like(x)
    train_logits, train_alpha, train_beta = train_model.forward_with_routing(train_x)
    loss = train_logits.square().mean() + 0.01 * train_alpha.square().mean()
    loss.backward()

    nonfinite_grads = []
    router_gradients = []
    for name, parameter in train_model.named_parameters():
        if parameter.grad is None:
            continue
        if not torch.isfinite(parameter.grad).all().item():
            nonfinite_grads.append(name)
        if name.startswith("dynamic_router."):
            router_gradients.append(name)

    print("9. Backward loss:", float(loss.item()))
    print("10. Router parameters with gradients:", len(router_gradients))
    assert len(nonfinite_grads) == 0, (
        "Non-finite gradients found in: " + str(nonfinite_grads)
    )
    assert len(router_gradients) > 0, "No gradient reached dynamic_router"

    print()
    print("========== TSSR TENSOR CHECK: PASS ==========")


if __name__ == "__main__":
    main()
