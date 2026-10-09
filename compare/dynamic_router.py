"""Task-adaptive routing and state propagation for GSC-ViT.

The soft spectral-spatial route implements innovation 1.  The
alpha-driven bidirectional spectral state route implements the spectral-only
part of innovation 2.  ``SpatialStateRouter`` extends the same state
propagation idea to beta-ordered spatial tokens while restoring the original
2-D layout before GSSA.
"""

import torch
from torch import nn


class SpectralRelevanceEstimator(nn.Module):
    """Estimate sample-specific relevance for latent spectral groups."""

    def __init__(self, channels, num_groups, hidden_dim=None, temperature=1.0):
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if num_groups <= 0:
            raise ValueError("num_groups must be positive")
        if channels % num_groups != 0:
            raise ValueError(
                "channels must be divisible by num_groups; "
                f"got channels={channels}, num_groups={num_groups}"
            )
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        self.channels = channels
        self.num_groups = num_groups
        self.channels_per_group = channels // num_groups
        self.temperature = float(temperature)
        hidden_dim = hidden_dim or max(8, num_groups * 2)

        # One scalar descriptor per group is enough for the first soft-routing
        # version.  The second linear layer lets the groups compete with one
        # another before the softmax normalization.
        self.predictor = nn.Sequential(
            nn.Linear(num_groups, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_groups),
        )

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError(
                "SpectralRelevanceEstimator expects (B, C, H, W), "
                f"got {tuple(x.shape)}"
            )
        batch, channels, _, _ = x.shape
        if channels != self.channels:
            raise ValueError(
                "SpectralRelevanceEstimator received an unexpected channel "
                f"count: expected {self.channels}, got {channels}"
            )

        grouped = x.reshape(
            batch,
            self.num_groups,
            self.channels_per_group,
            x.shape[-2],
            x.shape[-1],
        )
        group_descriptor = grouped.mean(dim=(2, 3, 4))
        logits = self.predictor(group_descriptor)
        return torch.softmax(logits / self.temperature, dim=1)


class SpatialRelevanceEstimator(nn.Module):
    """Estimate sample-specific relevance for spatial tokens."""

    def __init__(self, channels, hidden_dim=None):
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        hidden_dim = hidden_dim or max(8, channels // 4)
        self.channels = channels
        self.predictor = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=1, bias=True),
        )

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError(
                "SpatialRelevanceEstimator expects (B, C, H, W), "
                f"got {tuple(x.shape)}"
            )
        if x.shape[1] != self.channels:
            raise ValueError(
                "SpatialRelevanceEstimator received an unexpected channel "
                f"count: expected {self.channels}, got {x.shape[1]}"
            )

        # Scale the sigmoid output around one.  This keeps the residual route
        # numerically well behaved while still allowing spatial suppression
        # and enhancement after the centered residual transform.
        return 2.0 * torch.sigmoid(self.predictor(x))


class SpectralSpatialRouter(nn.Module):
    """Task-adaptive spectral-spatial residual soft routing.

    Parameters
    ----------
    channels : int
        Number of feature channels at the insertion point.
    num_groups : int
        Number of latent spectral groups.  ``channels`` must be divisible by
        this value.
    route_strength : float, default=0.5
        Residual routing strength.  Zero is equivalent to an identity route.
    temperature : float, default=1.0
        Softmax temperature for spectral relevance estimation.
    router_variant : {"full", "no_spectral", "no_spatial"}, default="full"
        Ablation mode. ``no_spectral`` replaces the learned spectral scores
        with a uniform distribution, while ``no_spatial`` replaces the
        learned spatial scores with the neutral gain one.
    """

    def __init__(
        self,
        channels,
        num_groups=8,
        route_strength=0.5,
        temperature=1.0,
        router_variant="full",
    ):
        super().__init__()
        if route_strength < 0 or route_strength > 1:
            raise ValueError("route_strength must be in [0, 1]")
        valid_variants = {"full", "no_spectral", "no_spatial"}
        if router_variant not in valid_variants:
            raise ValueError(
                "router_variant must be one of "
                f"{sorted(valid_variants)}, got {router_variant!r}"
            )

        self.channels = channels
        self.num_groups = num_groups
        self.route_strength = float(route_strength)
        self.router_variant = router_variant
        self.spectral_estimator = SpectralRelevanceEstimator(
            channels=channels,
            num_groups=num_groups,
            temperature=temperature,
        )
        self.spatial_estimator = SpatialRelevanceEstimator(channels=channels)

        # These are populated during forward for visualization/debugging.  We
        # detach them so retaining a score for logging does not retain the
        # complete training computation graph.
        self.last_alpha = None
        self.last_beta = None

    def route(self, x):
        if x.dim() != 4:
            raise ValueError(
                "SpectralSpatialRouter expects (B, C, H, W), "
                f"got {tuple(x.shape)}"
            )
        if x.shape[1] != self.channels:
            raise ValueError(
                "SpectralSpatialRouter received an unexpected channel count: "
                f"expected {self.channels}, got {x.shape[1]}"
            )

        if self.router_variant == "no_spectral":
            # Keep the same alpha shape and probability semantics while
            # removing the learned spectral branch from this ablation.
            alpha = torch.full(
                (x.shape[0], self.num_groups),
                1.0 / self.num_groups,
                dtype=x.dtype,
                device=x.device,
            )
        else:
            alpha = self.spectral_estimator(x)

        if self.router_variant == "no_spatial":
            # beta=1 is the neutral spatial gain.  The spatial estimator is
            # therefore absent from this ablation's forward computation.
            beta = torch.ones(
                (x.shape[0], 1, x.shape[-2], x.shape[-1]),
                dtype=x.dtype,
                device=x.device,
            )
        else:
            beta = self.spatial_estimator(x)

        # alpha is a probability distribution over groups.  Multiplying by G
        # turns it into a gain whose neutral value is approximately one.  The
        # same centering is used for beta so the residual route can suppress
        # low-relevance responses as well as enhance high-relevance ones.
        spectral_gain = alpha * self.num_groups
        spectral_gain = spectral_gain.repeat_interleave(
            self.spectral_estimator.channels_per_group, dim=1
        ).unsqueeze(-1).unsqueeze(-1)
        centered_gain = spectral_gain * beta

        routed = x * (
            1.0 + self.route_strength * (centered_gain - 1.0)
        )

        self.last_alpha = alpha.detach()
        self.last_beta = beta.detach()
        return routed, alpha, beta

    def forward(self, x):
        routed, _, _ = self.route(x)
        return routed


class SpectralStateRouter(nn.Module):
    """Alpha-driven bidirectional spectral state propagation before GSSA.

    ``alpha`` defines the sample-specific spectral-group order.  Forward and
    reverse recurrent passes are fused in that order, then ``inverse_perm``
    restores the original channel layout for the downstream GSSA.
    """

    def __init__(self, channels, num_groups=8, state_dim=None, enabled=True):
        super().__init__()
        if channels <= 0 or num_groups <= 0 or channels % num_groups != 0:
            raise ValueError("channels must be positive and divisible by num_groups")
        self.channels = channels
        self.num_groups = num_groups
        self.channels_per_group = channels // num_groups
        self.enabled = bool(enabled)
        state_dim = state_dim or self.channels_per_group
        self.to_state = nn.Linear(self.channels_per_group, state_dim)
        self.forward_update = nn.GRUCell(self.channels_per_group, state_dim)
        self.backward_update = nn.GRUCell(self.channels_per_group, state_dim)
        self.from_state = nn.Linear(state_dim, self.channels_per_group)
        self.mix = nn.Parameter(torch.tensor(0.5))
        self.last_permutation = None
        self.last_inverse_permutation = None
        self.last_beta = None

    def forward(self, route_feat, alpha, beta=None):
        if route_feat.dim() != 4:
            raise ValueError("route_feat must have shape (B, C, H, W)")
        if route_feat.shape[1] != self.channels:
            raise ValueError(f"expected {self.channels} channels, got {route_feat.shape[1]}")
        if alpha.shape != (route_feat.shape[0], self.num_groups):
            raise ValueError("alpha must have shape (B, num_groups)")
        if beta is not None:
            expected_beta = (route_feat.shape[0], 1, route_feat.shape[-2], route_feat.shape[-1])
            if tuple(beta.shape) != expected_beta:
                raise ValueError(f"beta must have shape {expected_beta}")
            self.last_beta = beta.detach()
        else:
            self.last_beta = None
        if not self.enabled:
            self.last_permutation = None
            self.last_inverse_permutation = None
            return route_feat

        b, _, h, w = route_feat.shape
        groups = route_feat.reshape(b, self.num_groups, self.channels_per_group, h, w)
        permutation = torch.argsort(alpha, dim=1, descending=True)
        inverse = torch.argsort(permutation, dim=1)
        gather = permutation[:, :, None, None, None].expand(-1, -1, self.channels_per_group, h, w)
        ordered = groups.gather(1, gather)
        # Make each sequence one spatial location with all ordered spectral
        # groups: (B, H, W, G, C) -> (B*H*W, G, C).
        tokens = ordered.permute(0, 3, 4, 1, 2).reshape(
            b * h * w, self.num_groups, self.channels_per_group
        )
        if beta is None:
            spatial_gain = torch.ones(
                b * h * w, 1, dtype=route_feat.dtype, device=route_feat.device
            )
        else:
            # beta is reused as the TSSR spatial confidence.  It modulates
            # state residual magnitude while alpha alone defines spectral
            # propagation order; no spatial state sequence is introduced.
            spatial_gain = beta[:, 0].reshape(b * h * w, 1)
        forward_states = []
        state = self.to_state(tokens[:, 0])
        for index in range(self.num_groups):
            if index > 0:
                state = self.forward_update(tokens[:, index], state)
            forward_states.append(state)

        backward_states = [None] * self.num_groups
        state = self.to_state(tokens[:, -1])
        for index in range(self.num_groups - 1, -1, -1):
            if index < self.num_groups - 1:
                state = self.backward_update(tokens[:, index], state)
            backward_states[index] = state

        outputs = []
        for index in range(self.num_groups):
            fused_state = forward_states[index] + backward_states[index]
            state_residual = self.from_state(fused_state) * spatial_gain
            outputs.append(tokens[:, index] + torch.tanh(self.mix) * state_residual)
        propagated = torch.stack(outputs, dim=1).reshape(b, h, w, self.num_groups, self.channels_per_group)
        propagated = propagated.permute(0, 3, 4, 1, 2)
        restore = inverse[:, :, None, None, None].expand(-1, -1, self.channels_per_group, h, w)
        restored = propagated.gather(1, restore).reshape(b, self.channels, h, w)
        self.last_permutation = permutation.detach()
        self.last_inverse_permutation = inverse.detach()
        return restored


class SpatialStateRouter(nn.Module):
    """Beta-ordered spatial state propagation before the original GSSA.

    The router uses the spatial confidence map produced by TSSR to order the
    ``H*W`` tokens independently for every sample.  Propagation is performed
    on the ordered sequence with the same pure-PyTorch bidirectional GRU
    executor used by :class:`SpectralStateRouter`.  The inverse permutation is
    applied before returning so the downstream GSSA always receives the
    original spatial layout.

    Parameters
    ----------
    channels : int
        Number of feature channels in the routed feature map.
    state_dim : int, optional
        Recurrent state width.  Defaults to ``channels``.
    state_strength : float, default=0.5
        Residual strength of the state propagation.  This is a spatial-state
        parameter and does not change the meaning of TSSR's ``route_strength``.
    enabled : bool, default=True
        When false, return the input feature map unchanged.
    """

    def __init__(self, channels, state_dim=None, state_strength=0.5, enabled=True):
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if state_strength < 0 or state_strength > 1:
            raise ValueError("state_strength must be in [0, 1]")
        self.channels = int(channels)
        self.state_dim = int(state_dim or channels)
        self.state_strength = float(state_strength)
        self.enabled = bool(enabled)

        self.to_state = nn.Linear(self.channels, self.state_dim)
        self.forward_update = nn.GRUCell(self.channels, self.state_dim)
        self.backward_update = nn.GRUCell(self.channels, self.state_dim)
        self.from_state = nn.Linear(self.state_dim, self.channels)
        self.mix = nn.Parameter(torch.tensor(0.5))

        self.last_permutation = None
        self.last_inverse_permutation = None
        self.last_beta = None
        self.last_space_tokens = None

    @staticmethod
    def _stable_argsort(scores):
        """Sort scores reproducibly, including when values are tied."""
        try:
            return torch.argsort(scores, dim=1, descending=True, stable=True)
        except TypeError:  # Older PyTorch versions do not expose ``stable``.
            return torch.argsort(scores, dim=1, descending=True)

    def _normalize_beta(self, beta, batch, height, width):
        if beta.dim() == 4:
            expected = (batch, 1, height, width)
            if tuple(beta.shape) != expected:
                raise ValueError(f"beta must have shape {expected} or (B, H, W)")
            return beta[:, 0]
        if beta.dim() == 3:
            expected = (batch, height, width)
            if tuple(beta.shape) != expected:
                raise ValueError(f"beta must have shape (B, H, W), got {tuple(beta.shape)}")
            return beta
        raise ValueError("beta must have shape (B, 1, H, W) or (B, H, W)")

    def forward(self, route_feat, beta):
        if route_feat.dim() != 4:
            raise ValueError("route_feat must have shape (B, C, H, W)")
        batch, channels, height, width = route_feat.shape
        if channels != self.channels:
            raise ValueError(
                f"expected {self.channels} channels, got {channels}"
            )
        beta_map = self._normalize_beta(beta, batch, height, width)
        self.last_beta = beta.detach()

        if not self.enabled:
            self.last_permutation = None
            self.last_inverse_permutation = None
            self.last_space_tokens = None
            return route_feat

        num_tokens = height * width
        # (B, C, H, W) -> (B, H*W, C), preserving row-major spatial indices.
        space_tokens = route_feat.permute(0, 2, 3, 1).reshape(
            batch, num_tokens, channels
        )
        spatial_score = beta_map.reshape(batch, num_tokens)
        permutation = self._stable_argsort(spatial_score)
        inverse = torch.argsort(permutation, dim=1)
        gather_index = permutation.unsqueeze(-1).expand(-1, -1, channels)
        ordered = space_tokens.gather(1, gather_index)
        ordered_gain = spatial_score.gather(1, permutation).unsqueeze(-1)

        # One sequence per sample.  The two directions reuse the validated
        # GRUCell state executor from the spectral state route.
        forward_states = []
        state = self.to_state(ordered[:, 0])
        for index in range(num_tokens):
            if index > 0:
                state = self.forward_update(ordered[:, index], state)
            forward_states.append(state)

        backward_states = [None] * num_tokens
        state = self.to_state(ordered[:, -1])
        for index in range(num_tokens - 1, -1, -1):
            if index < num_tokens - 1:
                state = self.backward_update(ordered[:, index], state)
            backward_states[index] = state

        propagated = []
        for index in range(num_tokens):
            fused_state = forward_states[index] + backward_states[index]
            residual = self.from_state(fused_state)
            residual = residual * ordered_gain
            propagated.append(
                ordered[:, index]
                + self.state_strength * torch.tanh(self.mix) * residual
            )
        propagated = torch.stack(propagated, dim=1)

        restore_index = inverse.unsqueeze(-1).expand(-1, -1, channels)
        restored_tokens = propagated.gather(1, restore_index)
        restored = restored_tokens.reshape(batch, height, width, channels).permute(
            0, 3, 1, 2
        )

        self.last_permutation = permutation.detach()
        self.last_inverse_permutation = inverse.detach()
        self.last_space_tokens = space_tokens.detach()
        return restored
