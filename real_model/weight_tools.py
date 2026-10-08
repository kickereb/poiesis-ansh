"""Low-rank weight directions and output-gradient utilities."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from model_utils import image_array


@dataclass
class LowRankDirection:
    """A deterministic low-rank direction for one linear weight matrix."""

    module: nn.Linear
    delta: torch.Tensor
    rank: int
    relative_norm: float
    seed: int

    @classmethod
    def random(
        cls,
        module: nn.Linear,
        rank: int,
        relative_norm: float,
        seed: int,
    ) -> "LowRankDirection":
        generator = torch.Generator(device="cpu").manual_seed(seed)
        left = torch.randn(
            module.out_features, rank, generator=generator, dtype=torch.float64
        )
        right = torch.randn(
            rank, module.in_features, generator=generator, dtype=torch.float64
        )
        delta = left @ right
        weight_norm = float(module.weight.detach().float().norm().cpu())
        delta = delta * (relative_norm * weight_norm / float(delta.norm()))
        return cls(module, delta, rank, relative_norm, seed)

    @classmethod
    def from_delta(
        cls,
        module: nn.Linear,
        delta: torch.Tensor,
        rank: int,
        relative_norm: float,
        seed: int,
    ) -> "LowRankDirection":
        delta = delta.detach().to(device="cpu", dtype=torch.float64)
        weight_norm = float(module.weight.detach().float().norm().cpu())
        delta = delta * (relative_norm * weight_norm / float(delta.norm()))
        return cls(module, delta, rank, relative_norm, seed)

    @contextmanager
    def applied(self, strength: float) -> Iterator[None]:
        original = self.module.weight.detach().clone()
        change = self.delta.to(
            device=self.module.weight.device,
            dtype=self.module.weight.dtype,
        )
        with torch.no_grad():
            self.module.weight.add_(strength * change)
        try:
            yield
        finally:
            with torch.no_grad():
                self.module.weight.copy_(original)


@dataclass
class TrainableLoRAAdapter:
    """A trainable rank-limited update for one linear layer."""

    module: nn.Linear
    rank: int
    seed: int
    A: nn.Parameter = field(init=False)
    B: nn.Parameter = field(init=False)
    _handle: object = field(default=None, init=False)

    def __post_init__(self) -> None:
        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        initial = torch.randn(
            self.rank,
            self.module.in_features,
            generator=generator,
            dtype=torch.float32,
        ) / np.sqrt(self.module.in_features)
        self.A = nn.Parameter(
            initial.to(
                device=self.module.weight.device,
                dtype=self.module.weight.dtype,
            )
        )
        self.B = nn.Parameter(
            torch.zeros(
                self.module.out_features,
                self.rank,
                device=self.module.weight.device,
                dtype=self.module.weight.dtype,
            )
        )

    def _hook(self, _module, inputs, output):
        hidden = F.linear(inputs[0], self.A)
        return output + F.linear(hidden, self.B)

    def parameters(self) -> list[nn.Parameter]:
        """Return the two trainable factors."""
        return [self.A, self.B]

    def delta(self) -> torch.Tensor:
        """Return the current low-rank weight change on CPU."""
        return (self.B @ self.A).detach().to(device="cpu").to(torch.float64)

    def __enter__(self) -> "TrainableLoRAAdapter":
        self._handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, *_args) -> None:
        self._handle.remove()
        self._handle = None


def cross_attention_value_modules(unet: nn.Module) -> list[tuple[str, nn.Linear]]:
    """List U-Net cross-attention value projections in model order."""
    modules = []
    for name, module in unet.named_modules():
        if name.endswith("attn2.to_v") and isinstance(module, nn.Linear):
            modules.append((name, module))
    if not modules:
        raise ValueError("no cross-attention value projections were found")
    return modules


def selected_value_modules(unet: nn.Module) -> list[tuple[str, nn.Linear]]:
    """Select early, middle, and late cross-attention value projections."""
    modules = cross_attention_value_modules(unet)
    indices = sorted({0, len(modules) // 2, len(modules) - 1})
    return [modules[index] for index in indices]


def directional_output_gradient(
    negative: Image.Image,
    positive: Image.Image,
    epsilon: float,
) -> np.ndarray:
    """Calculate one central finite-difference image response."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    return (image_array(positive) - image_array(negative)) / (2.0 * epsilon)


def gradient_heatmap(gradient: np.ndarray) -> Image.Image:
    """Render directional-gradient magnitude as a heatmap."""
    magnitude = np.linalg.norm(gradient, axis=2)
    scale = float(np.quantile(magnitude, 0.995))
    normalised = np.clip(magnitude / max(scale, 1e-12), 0.0, 1.0)
    red = np.uint8(np.round(255.0 * normalised))
    green = np.uint8(np.round(180.0 * np.sqrt(normalised)))
    blue = np.uint8(np.round(40.0 * (1.0 - normalised)))
    return Image.fromarray(np.stack([red, green, blue], axis=2), mode="RGB")
