from __future__ import annotations

from dataclasses import dataclass, replace
from typing import ClassVar, Literal

import torch

from ..core.graph import Kind


@dataclass(frozen=True)
class CacheBlock:
    """Post-position-encoding K/V in [batch, kv_heads, position, head_dim]."""

    value_kind: ClassVar[Kind] = Kind.KV
    k: torch.Tensor
    v: torch.Tensor
    model_revision: str
    layer: int
    positions: tuple[int, ...]
    text_hash: str
    positional_encoding: str

    def validate(self):
        if self.k.ndim != 4 or self.k.shape != self.v.shape:
            raise ValueError("cache requires matching BHSD tensors")
        if self.k.shape[-2] != len(self.positions):
            raise ValueError("position IDs do not match tensor length")
        if any(b <= a for a, b in zip(self.positions, self.positions[1:])):
            raise ValueError("cache positions must be strictly increasing")
        if self.k.dtype != self.v.dtype or self.k.device != self.v.device:
            raise ValueError("K/V dtype/device mismatch")
        if not self.k.is_floating_point() or not torch.isfinite(self.k).all() or not torch.isfinite(self.v).all():
            raise ValueError("cache must contain finite floating point values")
        if self.positional_encoding not in ("sinusoidal", "rope_post_rotation"):
            raise ValueError("unsupported or unspecified positional encoding")
        if self.layer < 0 or not self.model_revision or not self.text_hash:
            raise ValueError("cache provenance is required")


@dataclass(frozen=True)
class Intervention:
    mode: Literal["k", "v", "kv"] = "kv"
    alpha: float = 1
    heads: tuple[int, ...] | None = None
    rank: int | None = None
    relative_norm_cap: float = 2

    def validate(self, block: CacheBlock):
        if self.mode not in ("k", "v", "kv") or not 0 <= self.alpha <= 1:
            raise ValueError("invalid intervention mode or strength")
        if self.relative_norm_cap <= 0:
            raise ValueError("norm cap must be positive")
        if self.rank is not None and not 1 <= self.rank <= min(block.k.shape[-2:]):
            raise ValueError("rank outside cache domain")
        if self.heads is not None and (not self.heads or len(set(self.heads)) != len(self.heads)
                                      or any(h < 0 or h >= block.k.shape[1] for h in self.heads)):
            raise ValueError("invalid KV head selection")


def patch_cache(source: CacheBlock, donor: CacheBlock, edit: Intervention) -> CacheBlock:
    source.validate(); donor.validate(); edit.validate(source)
    for key in ("model_revision", "layer", "positions", "text_hash", "positional_encoding"):
        if getattr(source, key) != getattr(donor, key):
            raise ValueError(f"incompatible cache {key}")
    if source.k.shape != donor.k.shape or source.k.dtype != donor.k.dtype or source.k.device != donor.k.device:
        raise ValueError("incompatible cache shape/dtype/device")

    def mix(a, b, enabled):
        out = a.clone()
        if not enabled or edit.alpha == 0:
            return out
        heads = list(range(a.shape[1])) if edit.heads is None else list(edit.heads)
        delta = b[:, heads] - a[:, heads]
        if edit.rank is not None:
            u, s, vh = torch.linalg.svd(delta.float(), full_matrices=False)
            delta = ((u[..., :edit.rank] * s[..., None, :edit.rank]) @ vh[..., :edit.rank, :]).to(a.dtype)
        allowed = edit.relative_norm_cap * a[:, heads].norm().clamp(min=1e-8)
        scale = (allowed / delta.norm().clamp(min=1e-8)).clamp(max=1)
        out[:, heads] += edit.alpha * scale * delta
        return out

    return replace(source, k=mix(source.k, donor.k, "k" in edit.mode),
                   v=mix(source.v, donor.v, "v" in edit.mode))


def prefix_window_mask(query_positions: torch.Tensor, key_positions: torch.Tensor,
                       prefix: int, window: int) -> torch.Tensor:
    """True means visible. Keep initial prefix OR recent window, always causal."""
    if prefix < 0 or window < 1 or query_positions.ndim != 1 or key_positions.ndim != 1:
        raise ValueError("invalid prefix/window/positions")
    q, k = query_positions[:, None], key_positions[None, :]
    return (k <= q) & ((k < prefix) | (k >= q - window + 1))
