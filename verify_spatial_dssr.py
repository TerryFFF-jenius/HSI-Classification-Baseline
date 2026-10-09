"""Phase-six spatial state routing smoke verification.

This script uses synthetic tensors only.  It checks the beta-ordered spatial
token contract, inverse layout restoration, both phase-six model paths,
backward gradients, and compatibility with the existing model names.  It does
not train a dataset model or write experiment results.

Run from the repository root with::

    python verify_spatial_dssr.py
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

from compare.dynamic_router import SpatialStateRouter
from models import build_model


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def finite(name, tensor):
    require(torch.is_tensor(tensor), f"{name} is not a tensor")
    if tensor.is_floating_point() or tensor.is_complex():
        require(torch.isfinite(tensor).all().item(), f"{name} contains NaN/Inf")


def assert_inverse(permutation, inverse, count, label):
    expected = torch.arange(count, device=permutation.device).expand_as(permutation)
    require(permutation.shape == inverse.shape, f"{label}: permutation shape mismatch")
    require(
        torch.equal(torch.gather(permutation, 1, inverse), expected),
        f"{label}: inverse permutation does not restore indices",
    )


def check_spatial_router_contract():
    """Check shapes, deterministic ties, ordering, restoration, and gradients."""
    torch.manual_seed(61)
    batch, channels, height, width = 2, 64, 9, 9
    route_feat = torch.randn(batch, channels, height, width, requires_grad=True)
    beta = torch.randn(batch, 1, height, width, requires_grad=True)
    router = SpatialStateRouter(channels=channels, state_strength=0.5)
    state_feat = router(route_feat, beta)

    require(router.last_space_tokens.shape == (batch, 81, 64), "space_tokens shape")
    require(router.last_permutation.shape == (batch, 81), "permutation shape")
    require(router.last_inverse_permutation.shape == (batch, 81), "inverse shape")
    require(state_feat.shape == (batch, 64, 9, 9), "state_feat shape")
    for name, value in (
        ("space_tokens", router.last_space_tokens),
        ("permutation", router.last_permutation),
        ("inverse_permutation", router.last_inverse_permutation),
        ("state_feat", state_feat),
    ):
        finite(name, value)
    assert_inverse(router.last_permutation, router.last_inverse_permutation, 81, "random")

    # A controlled score map makes the ordering and restored row-major layout
    # observable.  state_strength=0 leaves tokens unchanged after restore.
    controlled = torch.arange(81.0).reshape(1, 1, 9, 9)
    controlled_beta = torch.arange(81.0).reshape(1, 1, 9, 9)
    identity_router = SpatialStateRouter(channels=1, state_strength=0.0)
    restored = identity_router(controlled, controlled_beta)
    expected_order = torch.arange(80, -1, -1).reshape(1, 81)
    require(
        torch.equal(identity_router.last_permutation, expected_order),
        "descending beta scores did not produce descending permutation",
    )
    assert_inverse(
        identity_router.last_permutation,
        identity_router.last_inverse_permutation,
        81,
        "controlled",
    )
    require(torch.equal(restored, controlled), "inverse restore changed spatial layout")

    # The implementation also accepts beta without a singleton channel.
    beta_3d_router = SpatialStateRouter(channels=1, state_strength=0.0)
    beta_3d_router(controlled, controlled_beta[:, 0])
    require(
        torch.equal(beta_3d_router.last_permutation, expected_order),
        "3-D beta map did not produce the expected ordering",
    )
    print("[PASS] beta shape compatibility: (B, 1, H, W) and (B, H, W)")
    restored_3d = identity_router(controlled, controlled_beta[:, 0])
    require(torch.equal(restored_3d, controlled), "3-D beta compatibility failed")

    # Equal scores must be reproducible.  Stable argsort keeps row-major order.
    tie_router = SpatialStateRouter(channels=1, state_strength=0.0)
    tie_router(torch.zeros(1, 1, 9, 9), torch.ones(1, 1, 9, 9))
    require(
        torch.equal(tie_router.last_permutation, torch.arange(81).reshape(1, 81)),
        "tied beta scores are not deterministic",
    )

    loss = state_feat.square().mean()
    loss.backward()
    finite("route_feat.grad", route_feat.grad)
    finite("beta.grad", beta.grad)
    parameter_gradients = [
        parameter.grad
        for parameter in router.parameters()
        if parameter.grad is not None
    ]
    require(parameter_gradients, "spatial state parameters received no gradients")
    for gradient in parameter_gradients:
        finite("spatial parameter gradient", gradient)
    print("[PASS] SpatialStateRouter shapes, ordering, ties, inverse restore, backward")


def copy_common_weights(source, target):
    """Copy shared backbone/router weights while ignoring phase-specific modules."""
    source_state = source.state_dict()
    target_state = target.state_dict()
    common = {
        name: value
        for name, value in source_state.items()
        if name in target_state and target_state[name].shape == value.shape
    }
    target_state.update(common)
    target.load_state_dict(target_state)


def check_model_path(model_name, expected_items):
    torch.manual_seed(62)
    batch, in_channels, classes, patch = 1, 15, 16, 7
    model = build_model(
        model_name,
        in_channels,
        classes,
        patch,
        spectral_groups=8,
        route_strength=0.5,
        route_temperature=1.0,
        router_variant="full",
        spatial_state_enabled=True,
    )
    model.train()
    inputs = torch.randn(batch, 1, in_channels, patch, patch)
    outputs = model.forward_with_routing(inputs)
    require(len(outputs) == expected_items, f"{model_name}: routing tuple length")
    logits, alpha, beta = outputs[:3]
    finite(f"{model_name}.logits", logits)
    finite(f"{model_name}.alpha", alpha)
    finite(f"{model_name}.beta", beta)
    require(logits.shape == (batch, classes), f"{model_name}: logits shape")
    require(alpha.shape == (batch, 8), f"{model_name}: alpha shape")
    require(beta.shape == (batch, 1, 9, 9), f"{model_name}: beta shape")
    spatial_perm = outputs[-2]
    spatial_inverse = outputs[-1]
    require(spatial_perm.shape == (batch, 81), f"{model_name}: spatial permutation shape")
    require(spatial_inverse.shape == (batch, 81), f"{model_name}: spatial inverse shape")
    assert_inverse(spatial_perm, spatial_inverse, 81, model_name)
    finite(f"{model_name}.spatial_permutation", spatial_perm)
    finite(f"{model_name}.spatial_inverse", spatial_inverse)

    loss = logits.square().mean()
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    require(gradients, f"{model_name}: no gradients")
    for gradient in gradients:
        finite(f"{model_name} gradient", gradient)
    print(
        f"[PASS] {model_name}: logits={tuple(logits.shape)} "
        f"alpha={tuple(alpha.shape)} beta={tuple(beta.shape)} "
        f"spatial_perm={tuple(spatial_perm.shape)} backward"
    )


def check_disabled_degenerations():
    """Verify disabling the new spatial module reproduces prior model paths."""
    torch.manual_seed(63)
    x = torch.randn(1, 1, 15, 7, 7)

    spatial = build_model("gscvit_spatial", 15, 16, 7, spatial_state_enabled=False)
    tssr = build_model("gscvit_tssr", 15, 16, 7)
    copy_common_weights(spatial, tssr)
    spatial.eval()
    tssr.eval()
    with torch.no_grad():
        spatial_logits = spatial(x)
        tssr_logits = tssr(x)
    require(torch.allclose(spatial_logits, tssr_logits, atol=1e-6, rtol=1e-6),
            "disabled spatial route does not reproduce TSSR")
    print("[PASS] gscvit_spatial with spatial route disabled reproduces gscvit_tssr")

    serial = build_model("gscvit_dssr_spatial", 15, 16, 7, spatial_state_enabled=False)
    dssr = build_model("gscvit_dssr", 15, 16, 7)
    copy_common_weights(serial, dssr)
    # The serial wrapper exposes a compatibility alias for the same spectral
    # state module that the historical DSSR wrapper calls ``state_router``.
    dssr.state_router.load_state_dict(serial.spectral_state_router.state_dict())
    serial.eval()
    dssr.eval()
    with torch.no_grad():
        serial_logits = serial(x)
        dssr_logits = dssr(x)
    require(torch.allclose(serial_logits, dssr_logits, atol=1e-6, rtol=1e-6),
            "disabled spatial route does not reproduce DSSR")
    print("[PASS] gscvit_dssr_spatial with spatial route disabled reproduces gscvit_dssr")


def check_compatibility_models():
    x = torch.randn(1, 1, 15, 7, 7)
    for name in ("gscvit", "gscvit_tssr", "gscvit_dssr"):
        model = build_model(name, 15, 16, 7)
        model.eval()
        with torch.no_grad():
            logits = model(x)
        require(logits.shape == (1, 16), f"{name}: compatibility logits shape")
        finite(f"{name}.logits", logits)
        print(f"[PASS] compatibility model={name} logits={(1, 16)}")
    for variant in ("no_spectral", "no_spatial"):
        model = build_model("gscvit_spatial", 15, 16, 7,
                            router_variant=variant,
                            spatial_state_enabled=True)
        model.eval()
        with torch.no_grad():
            logits, alpha, beta, permutation, inverse = model.forward_with_routing(x)
        require(logits.shape == (1, 16), f"{variant}: logits shape")
        require(permutation.shape == inverse.shape == (1, 81), f"{variant}: spatial route shape")
        assert_inverse(permutation, inverse, 81, variant)
        finite(f"{variant}.logits", logits)
        print(f"[PASS] compatibility router_variant={variant}")


def main():
    print("========== Phase-six spatial state routing verification ==========")
    print("device: cpu; synthetic tensors only; no AutoDL training")
    check_spatial_router_contract()
    check_model_path("gscvit_spatial", expected_items=5)
    check_model_path("gscvit_dssr_spatial", expected_items=7)
    check_disabled_degenerations()
    check_compatibility_models()
    print("========== SPATIAL STATE ROUTING CHECK: PASS ==========")


if __name__ == "__main__":
    main()
