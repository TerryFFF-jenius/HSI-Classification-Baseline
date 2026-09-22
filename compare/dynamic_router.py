"""Task-adaptive spectral-spatial soft routing for GSC-ViT.

This module implements the first innovation only.  It estimates relevance for
latent spectral groups and spatial tokens, then applies a residual soft
reweighting.  It deliberately does not perform hard Top-K pruning or state
propagation; those belong to later experimental stages.
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
    """

    def __init__(
        self,
        channels,
        num_groups=8,
        route_strength=0.5,
        temperature=1.0,
    ):
        super().__init__()
        if route_strength < 0 or route_strength > 1:
            raise ValueError("route_strength must be in [0, 1]")

        self.channels = channels
        self.num_groups = num_groups
        self.route_strength = float(route_strength)
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

        alpha = self.spectral_estimator(x)
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

