"""Minimal raw-window Transformer scaffold for GAMEEMO EEG windows.

The model expects tensors shaped ``batch x channels x samples``. It treats each
EEG channel as one token by default, projects the token into a Transformer
embedding, prepends a learned CLS token, and classifies from that CLS state.
The alternate temporal-patch mode splits the time axis into fixed patches where
each token sees all EEG channels for that patch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class EEGTransformerConfig:
    n_channels: int
    n_samples: int
    n_classes: int = 4
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.1
    input_mode: str = "channel"
    patch_samples: int = 32


def _require_torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for the raw-window Transformer. Install torch to run deep-learning experiments."
        ) from exc
    return torch, nn


def validate_transformer_config(config: EEGTransformerConfig) -> None:
    if config.n_channels <= 0:
        raise ValueError(f"n_channels must be positive, got {config.n_channels}")
    if config.n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {config.n_samples}")
    if config.n_classes <= 1:
        raise ValueError(f"n_classes must be greater than 1, got {config.n_classes}")
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
    if config.input_mode not in {"channel", "temporal_patch"}:
        raise ValueError(f"input_mode must be 'channel' or 'temporal_patch', got {config.input_mode}")
    if config.patch_samples <= 0:
        raise ValueError(f"patch_samples must be positive, got {config.patch_samples}")
    if config.input_mode == "temporal_patch" and config.n_samples % config.patch_samples != 0:
        raise ValueError(
            f"n_samples={config.n_samples} must be divisible by patch_samples={config.patch_samples}"
        )


def build_eeg_transformer(config: EEGTransformerConfig):
    """Build a Transformer for windows shaped batch x channels x samples."""
    torch, nn = _require_torch()
    validate_transformer_config(config)
    return EEGTransformer(config, torch, nn)


class EEGTransformer:
    """Thin factory-backed wrapper around a PyTorch module.

    Importing this module does not require PyTorch; the dependency is only
    needed when constructing the actual model.
    """

    def __new__(cls, config: EEGTransformerConfig, torch_module, nn_module):
        class _EEGTransformer(nn_module.Module):
            def __init__(self, model_config: EEGTransformerConfig) -> None:
                super().__init__()
                self.config = model_config
                if model_config.input_mode == "channel":
                    self.token_projection = nn_module.Linear(model_config.n_samples, model_config.d_model)
                    token_count = model_config.n_channels
                else:
                    self.token_projection = nn_module.Linear(
                        model_config.n_channels * model_config.patch_samples,
                        model_config.d_model,
                    )
                    token_count = model_config.n_samples // model_config.patch_samples
                self.cls_token = nn_module.Parameter(torch_module.zeros(1, 1, model_config.d_model))
                self.token_embedding = nn_module.Parameter(torch_module.zeros(1, token_count, model_config.d_model))
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
                nn_module.init.trunc_normal_(self.token_embedding, std=0.02)

            def _make_tokens(self, windows: "torch.Tensor") -> "torch.Tensor":
                if self.config.input_mode == "channel":
                    return self.token_projection(windows) + self.token_embedding

                batch_size, n_channels, n_samples = windows.shape
                n_patches = n_samples // self.config.patch_samples
                patches = windows.reshape(batch_size, n_channels, n_patches, self.config.patch_samples)
                patches = patches.permute(0, 2, 1, 3).reshape(batch_size, n_patches, -1)
                return self.token_projection(patches) + self.token_embedding

            def forward(self, windows: "torch.Tensor") -> "torch.Tensor":
                if windows.ndim != 3:
                    raise ValueError(f"Expected batch x channels x samples, got {tuple(windows.shape)}")
                if windows.shape[1] != self.config.n_channels:
                    raise ValueError(f"Expected {self.config.n_channels} channels, got {windows.shape[1]}")
                if windows.shape[2] != self.config.n_samples:
                    raise ValueError(f"Expected {self.config.n_samples} samples, got {windows.shape[2]}")

                token_embeddings = self._make_tokens(windows)
                cls_tokens = self.cls_token.expand(windows.shape[0], -1, -1)
                tokens = torch_module.cat([cls_tokens, token_embeddings], dim=1)
                encoded = self.encoder(tokens)
                cls_state = self.norm(encoded[:, 0])
                return self.classifier(cls_state)

        return _EEGTransformer(config)
