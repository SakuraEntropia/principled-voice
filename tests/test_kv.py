from dataclasses import replace

import pytest
import torch

from principled_voice.research.kv import CacheBlock, Intervention, patch_cache, prefix_window_mask


def block(seed=1):
    g = torch.Generator().manual_seed(seed)
    return CacheBlock(torch.randn(1, 3, 8, 4, generator=g), torch.randn(1, 3, 8, 4, generator=g),
                      "revision-a", 2, tuple(range(8)), "same-text", "rope_post_rotation")


def test_noop_is_exact_and_non_destructive():
    a, b = block(1), block(2)
    old = a.k.clone()
    out = patch_cache(a, b, Intervention(alpha=0))
    assert torch.equal(out.k, a.k) and out.k.data_ptr() != a.k.data_ptr()
    assert torch.equal(a.k, old)


def test_k_only_v_only_and_head_selection():
    a, b = block(1), block(2)
    out = patch_cache(a, b, Intervention(mode="k", heads=(1,), relative_norm_cap=100))
    assert torch.equal(out.v, a.v)
    assert torch.equal(out.k[:, 0], a.k[:, 0])
    assert torch.allclose(out.k[:, 1], b.k[:, 1])
    v = patch_cache(a, b, Intervention(mode="v", relative_norm_cap=100))
    assert torch.equal(v.k, a.k) and torch.allclose(v.v, b.v, atol=1e-6)


@pytest.mark.parametrize("kwargs", [{"model_revision": "other"}, {"layer": 3},
                                      {"positions": tuple(range(1, 9))},
                                      {"text_hash": "other-text"},
                                      {"positional_encoding": "sinusoidal"}])
def test_incompatible_caches_rejected(kwargs):
    with pytest.raises(ValueError, match="incompatible"):
        patch_cache(block(), replace(block(2), **kwargs), Intervention())


def test_low_rank_and_strength_bound():
    a, b = block(1), block(2)
    out = patch_cache(a, b, Intervention(mode="k", rank=1, relative_norm_cap=0.1))
    assert torch.linalg.matrix_rank(out.k[0, 0] - a.k[0, 0], atol=1e-5) <= 1
    assert (out.k - a.k).norm() <= a.k.norm() * 0.10001
    with pytest.raises(ValueError):
        patch_cache(a, b, Intervention(alpha=2))


def test_prefix_window_mask_is_union_and_causal():
    mask = prefix_window_mask(torch.tensor([5, 9]), torch.arange(12), prefix=2, window=3)
    assert mask[0].nonzero().flatten().tolist() == [0, 1, 3, 4, 5]
    assert mask[1].nonzero().flatten().tolist() == [0, 1, 7, 8, 9]
