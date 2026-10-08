"""Activation recording and patching for classifier-free diffusion runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn


def replace_first_tensor(output: Any, replacement: torch.Tensor) -> Any:
    """Replace the primary tensor while preserving the output container."""
    if isinstance(output, torch.Tensor):
        return replacement
    if isinstance(output, tuple):
        return (replacement, *output[1:])
    if isinstance(output, list):
        return [replacement, *output[1:]]
    raise TypeError(f"unsupported hooked output type: {type(output)!r}")


def first_tensor(output: Any) -> torch.Tensor:
    """Return the primary tensor from a hooked module output."""
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(f"unsupported hooked output type: {type(output)!r}")


@dataclass
class ActivationRecorder:
    """Record one module output for every denoising step."""

    module: nn.Module
    activations: dict[int, torch.Tensor] = field(default_factory=dict)
    call_index: int = 0
    _handle: Any = None

    def _hook(self, _module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
        tensor = first_tensor(output)
        self.activations[self.call_index] = tensor.detach().to("cpu")
        self.call_index += 1
        return output

    def __enter__(self) -> "ActivationRecorder":
        self.activations.clear()
        self.call_index = 0
        self._handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, *_args: Any) -> None:
        self._handle.remove()
        self._handle = None


@dataclass
class ActivationPatcher:
    """Patch selected conditional-branch activations from a donor run."""

    module: nn.Module
    donor: dict[int, torch.Tensor]
    steps: set[int]
    strength: float = 1.0
    conditional_only: bool = True
    call_index: int = 0
    patched_calls: int = 0
    _handle: Any = None

    def _hook(self, _module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
        step = self.call_index
        self.call_index += 1
        if step not in self.steps:
            return output
        current = first_tensor(output)
        donor = self.donor[step].to(device=current.device, dtype=current.dtype)
        patched = current.clone()
        if self.conditional_only:
            if current.shape[0] % 2:
                raise ValueError("classifier-free batch must have an even size")
            start = current.shape[0] // 2
        else:
            start = 0
        patched[start:] = (
            (1.0 - self.strength) * current[start:]
            + self.strength * donor[start:]
        )
        self.patched_calls += 1
        return replace_first_tensor(output, patched)

    def __enter__(self) -> "ActivationPatcher":
        missing = self.steps - set(self.donor)
        if missing:
            raise ValueError(f"donor activations are missing steps: {sorted(missing)}")
        self.call_index = 0
        self.patched_calls = 0
        self._handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, *_args: Any) -> None:
        self._handle.remove()
        self._handle = None
