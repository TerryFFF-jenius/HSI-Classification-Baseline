import torch
from torch import nn

from compare.gscvit import GSCViTWrapper
from models import build_model


def max_abs_diff(a, b):
    return float((a - b).abs().max().item())


class FeatureProbe(nn.Module):
    """检查未来路由模块收到的特征形状。"""

    def __init__(self):
        super().__init__()
        self.seen_shape = None

    def forward(self, x):
        self.seen_shape = tuple(x.shape)
        return x


def main():
    torch.manual_seed(0)

    batch_size = 2
    in_channels = 15
    num_classes = 16
    patch_size = 7

    print("========== Phase 1 Test ==========")
    print("Input shape: (2, 1, 15, 7, 7)")

    # 直接实例化 GSC-ViT Wrapper
    model = GSCViTWrapper(
        in_channels=in_channels,
        num_classes=num_classes,
        patch_size=patch_size,
    )

    # 使用 eval 避免 BatchNorm 和 Dropout 影响测试结果
    model.eval()

    # 模拟当前数据管道输出的 5D 输入
    x_5d = torch.randn(
        batch_size,
        1,
        in_channels,
        patch_size,
        patch_size,
    )

    # 同一批数据的 4D 形式
    x_4d = x_5d.squeeze(1)

    # -------------------------------------------------
    # 测试 1：原始 model(x) 分类接口
    # -------------------------------------------------
    with torch.no_grad():
        logits_5d = model(x_5d)

    print("1. 5D logits shape:", tuple(logits_5d.shape))

    assert logits_5d.shape == (
        batch_size,
        num_classes,
    ), f"Unexpected logits shape: {tuple(logits_5d.shape)}"

    # -------------------------------------------------
    # 测试 2：4D 和 5D 输入都可以工作
    # -------------------------------------------------
    with torch.no_grad():
        logits_4d = model(x_4d)

    print("2. 4D logits shape:", tuple(logits_4d.shape))

    assert logits_4d.shape == (
        batch_size,
        num_classes,
    ), f"Unexpected 4D logits shape: {tuple(logits_4d.shape)}"

    input_format_diff = max_abs_diff(logits_5d, logits_4d)

    print(
        "3. 4D/5D logits max difference:",
        input_format_diff,
    )

    assert input_format_diff < 1e-6, (
        "4D and 5D inputs produced different logits. "
        "Check input conversion or resize logic."
    )

    # -------------------------------------------------
    # 测试 3：获取最终特征和 pre-GSSA 特征
    # -------------------------------------------------
    with torch.no_grad():
        final_feat, pre_gssa_feat = model.forward_features(
            x_5d,
            return_pre_gssa=True,
        )

    print("4. Final feature shape:", tuple(final_feat.shape))
    print("5. Pre-GSSA feature shape:", tuple(pre_gssa_feat.shape))

    assert final_feat.shape == (
        batch_size,
        64,
        9,
        9,
    ), f"Unexpected final feature shape: {tuple(final_feat.shape)}"

    assert pre_gssa_feat.shape == (
        batch_size,
        64,
        9,
        9,
    ), f"Unexpected pre-GSSA feature shape: {tuple(pre_gssa_feat.shape)}"

    # -------------------------------------------------
    # 测试 4：确认后续 adapter 确实接收到最后 GSC 输出
    # -------------------------------------------------
    probe = FeatureProbe()

    with torch.no_grad():
        adapted_final_feat, adapted_pre_gssa_feat = model.forward_features(
            x_5d,
            pre_gssa_adapter=probe,
            return_pre_gssa=True,
        )

    print("6. Adapter received shape:", probe.seen_shape)

    assert probe.seen_shape == (
        batch_size,
        64,
        9,
        9,
    ), (
        "The adapter did not receive the expected final GSC feature shape: "
        f"{probe.seen_shape}"
    )

    assert adapted_final_feat.shape == (
        batch_size,
        64,
        9,
        9,
    )

    assert adapted_pre_gssa_feat.shape == (
        batch_size,
        64,
        9,
        9,
    )

    # Identity adapter 不应改变输出
    identity_diff = max_abs_diff(final_feat, adapted_final_feat)

    print(
        "7. Identity adapter max difference:",
        identity_diff,
    )

    assert identity_diff < 1e-6, (
        "Identity adapter changed the feature output unexpectedly."
    )

    # -------------------------------------------------
    # 测试 5：反向传播是否正常
    # -------------------------------------------------
    train_model = GSCViTWrapper(
        in_channels=in_channels,
        num_classes=num_classes,
        patch_size=patch_size,
    )

    train_model.train()

    train_input = torch.randn(
        batch_size,
        1,
        in_channels,
        patch_size,
        patch_size,
    )

    train_logits = train_model(train_input)
    loss = train_logits.mean()
    loss.backward()

    nonfinite_grads = []

    for name, parameter in train_model.named_parameters():
        if parameter.grad is not None:
            if not torch.isfinite(parameter.grad).all():
                nonfinite_grads.append(name)

    print("8. Backward propagation: PASS")

    assert len(nonfinite_grads) == 0, (
        "Non-finite gradients found in: "
        + str(nonfinite_grads)
    )

    # -------------------------------------------------
    # 测试 6：通过项目正式 build_model 接口构建
    # -------------------------------------------------
    registered_model = build_model(
        model_name="gscvit",
        in_channels=in_channels,
        num_classes=num_classes,
        patch_size=patch_size,
    )

    registered_model.eval()

    with torch.no_grad():
        registered_logits = registered_model(x_5d)

    print(
        "9. Registry model logits shape:",
        tuple(registered_logits.shape),
    )

    assert registered_logits.shape == (
        batch_size,
        num_classes,
    )

    print()
    print("========== PHASE 1 CHECK: PASS ==========")


if __name__ == "__main__":
    main()