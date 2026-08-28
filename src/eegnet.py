"""Minimal EEGNet-style model scaffold for GAMEEMO windows.

The model expects window tensors shaped ``batch x channels x samples`` and
internally converts them to the ``batch x 1 x channels x samples`` layout used
by the original EEGNet convolution pattern.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class EEGNetConfig:
    n_channels: int
    n_samples: int
    n_classes: int = 4
    temporal_filters: int = 8
    depth_multiplier: int = 2
    separable_filters: int = 16
    temporal_kernel: int = 64
    separable_kernel: int = 16
    dropout: float = 0.5


def _require_torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for EEGNet. Install torch to run deep-learning experiments."
        ) from exc
    return torch, nn


def _conv_output_length(length: int, kernel_size: int, padding: int = 0, stride: int = 1) -> int:
    return ((length + 2 * padding - kernel_size) // stride) + 1


def eegnet_flattened_size(config: EEGNetConfig) -> int:
    """Return flattened classifier input size for the configured sample length."""
    temporal_length = _conv_output_length(
        config.n_samples,
        config.temporal_kernel,
        padding=config.temporal_kernel // 2,
    )
    pooled_once = temporal_length // 4
    separable_length = _conv_output_length(
        pooled_once,
        config.separable_kernel,
        padding=config.separable_kernel // 2,
    )
    pooled_twice = separable_length // 8
    return config.separable_filters * pooled_twice


def build_eegnet(config: EEGNetConfig):
    """Build an EEGNet-style classifier for windows shaped batch x channels x samples."""
    _, nn = _require_torch()

    if config.n_channels <= 0:
        raise ValueError(f"n_channels must be positive, got {config.n_channels}")
    if config.n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {config.n_samples}")
    if config.n_classes <= 1:
        raise ValueError(f"n_classes must be greater than 1, got {config.n_classes}")

    return EEGNet(config, nn)


class EEGNet:
    """Thin factory-backed wrapper around a PyTorch module.

    The real ``nn.Module`` base class is attached at runtime so importing this
    module does not require PyTorch unless the model is actually constructed.
    """

    def __new__(cls, config: EEGNetConfig, nn_module):
        class _EEGNet(nn_module.Module):
            def __init__(self, model_config: EEGNetConfig) -> None:
                super().__init__()
                self.config = model_config
                depthwise_filters = model_config.temporal_filters * model_config.depth_multiplier
                flattened_size = eegnet_flattened_size(model_config)
                if flattened_size <= 0:
                    raise ValueError(
                        "EEGNet flattened size is non-positive; increase n_samples or reduce pooling/kernel sizes."
                    )

                self.features = nn_module.Sequential(
                    nn_module.Conv2d(
                        1,
                        model_config.temporal_filters,
                        kernel_size=(1, model_config.temporal_kernel),
                        padding=(0, model_config.temporal_kernel // 2),
                        bias=False,
                    ),
                    nn_module.BatchNorm2d(model_config.temporal_filters),
                    nn_module.Conv2d(
                        model_config.temporal_filters,
                        depthwise_filters,
                        kernel_size=(model_config.n_channels, 1),
                        groups=model_config.temporal_filters,
                        bias=False,
                    ),
                    nn_module.BatchNorm2d(depthwise_filters),
                    nn_module.ELU(),
                    nn_module.AvgPool2d(kernel_size=(1, 4)),
                    nn_module.Dropout(model_config.dropout),
                    nn_module.Conv2d(
                        depthwise_filters,
                        depthwise_filters,
                        kernel_size=(1, model_config.separable_kernel),
                        padding=(0, model_config.separable_kernel // 2),
                        groups=depthwise_filters,
                        bias=False,
                    ),
                    nn_module.Conv2d(depthwise_filters, model_config.separable_filters, kernel_size=1, bias=False),
                    nn_module.BatchNorm2d(model_config.separable_filters),
                    nn_module.ELU(),
                    nn_module.AvgPool2d(kernel_size=(1, 8)),
                    nn_module.Dropout(model_config.dropout),
                    nn_module.Flatten(),
                )
                self.classifier = nn_module.Linear(flattened_size, model_config.n_classes)

            def forward(self, windows: "torch.Tensor") -> "torch.Tensor":
                if windows.ndim != 3:
                    raise ValueError(f"Expected batch x channels x samples, got {tuple(windows.shape)}")
                if windows.shape[1] != self.config.n_channels:
                    raise ValueError(f"Expected {self.config.n_channels} channels, got {windows.shape[1]}")
                if windows.shape[2] != self.config.n_samples:
                    raise ValueError(f"Expected {self.config.n_samples} samples, got {windows.shape[2]}")

                x = windows.unsqueeze(1)
                return self.classifier(self.features(x))

        return _EEGNet(config)
