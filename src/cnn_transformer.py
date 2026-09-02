"""Small CNN-Transformer hybrid for GAMEEMO EEG windows.

The model uses an EEGNet-style temporal convolution and depthwise spatial
filter before a compact Transformer encoder. This gives the Transformer local
EEG structure instead of asking attention to learn directly from raw samples.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class CNNTransformerConfig:
    n_channels: int
    n_samples: int
    n_classes: int = 4
    temporal_filters: int = 8
    depth_multiplier: int = 2
    temporal_kernel: int = 64
    pool_size: int = 4
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.1


def _require_torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for the CNN-Transformer hybrid. Install torch to run deep-learning experiments."
        ) from exc
    return torch, nn


def _conv_output_length(length: int, kernel_size: int, padding: int = 0, stride: int = 1) -> int:
    return ((length + 2 * padding - kernel_size) // stride) + 1


def _pooled_token_count(config: CNNTransformerConfig) -> int:
    temporal_length = _conv_output_length(
        config.n_samples,
        config.temporal_kernel,
        padding=config.temporal_kernel // 2,
    )
    return temporal_length // config.pool_size


def validate_cnn_transformer_config(config: CNNTransformerConfig) -> None:
    if config.n_channels <= 0:
        raise ValueError(f"n_channels must be positive, got {config.n_channels}")
    if config.n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {config.n_samples}")
    if config.n_classes <= 1:
        raise ValueError(f"n_classes must be greater than 1, got {config.n_classes}")
    if config.temporal_filters <= 0:
        raise ValueError(f"temporal_filters must be positive, got {config.temporal_filters}")
    if config.depth_multiplier <= 0:
        raise ValueError(f"depth_multiplier must be positive, got {config.depth_multiplier}")
    if config.temporal_kernel <= 0:
        raise ValueError(f"temporal_kernel must be positive, got {config.temporal_kernel}")
    if config.pool_size <= 0:
        raise ValueError(f"pool_size must be positive, got {config.pool_size}")
    if _pooled_token_count(config) <= 0:
        raise ValueError("CNN front end produced no time tokens; increase n_samples or reduce pooling/kernel size.")
    if config.d_model <= 0:
        raise ValueError(f"d_model must be positive, got {config.d_model}")
    if config.n_heads <= 0:
        raise ValueError(f"n_heads must be positive, got {config.n_heads}")
    if config.d_model % config.n_heads != 0:
        raise ValueError(f"d_model={config.d_model} must be divisible by n_heads={config.n_heads}")
    if config.n_layers <= 0:
        raise ValueError(f"n_layers must be positive, got {config.n_layers}")
    if config.dim_feedforward <= 0:
        raise ValueError(f"dim_feedforward must be positive, got {config.dim_feedforward}")
    if not 0 <= config.dropout < 1:
        raise ValueError(f"dropout must be in [0, 1), got {config.dropout}")


def build_cnn_transformer(config: CNNTransformerConfig):
    """Build a CNN-Transformer classifier for windows shaped batch x channels x samples."""
    torch, nn = _require_torch()
    validate_cnn_transformer_config(config)
    return CNNTransformer(config, torch, nn)


class CNNTransformer:
    """Factory wrapper so importing this module does not require PyTorch."""

    def __new__(cls, config: CNNTransformerConfig, torch_module, nn_module):
        class _CNNTransformer(nn_module.Module):
            def __init__(self, model_config: CNNTransformerConfig) -> None:
                super().__init__()
                self.config = model_config
                depthwise_filters = model_config.temporal_filters * model_config.depth_multiplier
                token_count = _pooled_token_count(model_config)

                self.cnn_frontend = nn_module.Sequential(
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
                    nn_module.AvgPool2d(kernel_size=(1, model_config.pool_size)),
                    nn_module.Dropout(model_config.dropout),
                )
                self.token_projection = nn_module.Linear(depthwise_filters, model_config.d_model)
                self.cls_token = nn_module.Parameter(torch_module.zeros(1, 1, model_config.d_model))
                self.position_embedding = nn_module.Parameter(torch_module.zeros(1, token_count, model_config.d_model))
                encoder_layer = nn_module.TransformerEncoderLayer(
                    d_model=model_config.d_model,
                    nhead=model_config.n_heads,
                    dim_feedforward=model_config.dim_feedforward,
                    dropout=model_config.dropout,
                    activation="gelu",
                    batch_first=True,
                )
                self.encoder = nn_module.TransformerEncoder(encoder_layer, num_layers=model_config.n_layers)
                self.norm = nn_module.LayerNorm(model_config.d_model)
                self.classifier = nn_module.Linear(model_config.d_model, model_config.n_classes)

                nn_module.init.trunc_normal_(self.cls_token, std=0.02)
                nn_module.init.trunc_normal_(self.position_embedding, std=0.02)

            def forward(self, windows: "torch.Tensor") -> "torch.Tensor":
                if windows.ndim != 3:
                    raise ValueError(f"Expected batch x channels x samples, got {tuple(windows.shape)}")
                if windows.shape[1] != self.config.n_channels:
                    raise ValueError(f"Expected {self.config.n_channels} channels, got {windows.shape[1]}")
                if windows.shape[2] != self.config.n_samples:
                    raise ValueError(f"Expected {self.config.n_samples} samples, got {windows.shape[2]}")

                features = self.cnn_frontend(windows.unsqueeze(1)).squeeze(2)
                tokens = features.transpose(1, 2)
                token_embeddings = self.token_projection(tokens) + self.position_embedding[:, : tokens.shape[1]]
                cls_tokens = self.cls_token.expand(windows.shape[0], -1, -1)
                encoded = self.encoder(torch_module.cat([cls_tokens, token_embeddings], dim=1))
                cls_state = self.norm(encoded[:, 0])
                return self.classifier(cls_state)

        return _CNNTransformer(config)
