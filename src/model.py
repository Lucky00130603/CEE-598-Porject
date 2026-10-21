"""Neural-network models for temporal precipitation downscaling."""

from __future__ import annotations

import torch
from torch import nn


class PrecipitationDownscalingANN(nn.Module):
    """Single-hidden-layer feedforward ANN used for one downscaling step."""

    def __init__(self, n_features: int = 4, hidden_size: int = 10) -> None:
        super().__init__()
        if n_features < 1:
            raise ValueError("n_features must be positive")
        if hidden_size < 1:
            raise ValueError("hidden_size must be positive")

        self.network = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return standardized precipitation predictions."""
        return self.network(x)
