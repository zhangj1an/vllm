# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the EarlyTom token-compression primitives.

Run directly: .venv/bin/python tests/models/multimodal/test_earlytom_units.py
"""

import torch

from vllm.multimodal.earlytom import (
    EarlyTomConfig,
    EarlyTomVideoAux,
    ema_segments,
    local_window_select,
    merge_attention_frame,
    outer_compress,
    temporal_merge,
)

torch.manual_seed(0)


def test_ema_segments():
    # Identical frames -> single segment capped by max_window.
    frames = torch.nn.functional.normalize(
        torch.ones(10, 4, 8) + 0 * torch.randn(10, 4, 8), dim=-1
    )
    segs = ema_segments(frames, tau=0.6, ema=0.9, max_window=6)
    # Repo semantics: a segment closes at i-1 once it spans max_window
    # frames, so each segment holds at most max_window - 1 frames.
    assert segs[0] == (0, 4) and segs[-1][1] == 9, segs
    # Alternating dissimilar frames -> every frame its own segment.
    a = torch.nn.functional.normalize(torch.randn(1, 4, 8), dim=-1)
    b = -a
    frames = torch.cat([a, b] * 5, dim=0)
    segs = ema_segments(frames, tau=0.6, ema=0.9, max_window=6)
    assert len(segs) == 10, segs
    print("ema_segments OK")


def test_temporal_merge():
    cfg = EarlyTomConfig(tau=0.6, max_window=6)
    # Slowly drifting video: adjacent middle frames are more similar to each
    # other than the segment boundary frames are -> middles get mixed.
    base = torch.randn(16, 8)
    drift = torch.randn(16, 8)
    feat = torch.stack([base + 0.08 * i * drift for i in range(12)])
    origin = torch.arange(12)
    merged, mo, segs = temporal_merge(feat, origin, cfg, refine=True)
    assert merged.size(0) == mo.numel() < 12
    assert segs is not None and segs[-1][1] == merged.size(0) - 1
    assert (mo.sort().values == mo).all(), "origins must stay ordered"
    print(f"temporal_merge OK ({12} -> {merged.size(0)} frames)")


def test_local_window_select():
    for s, n in [(64, 16), (64, 64), (10, 3), (7, 7), (100, 1)]:
        attn = torch.rand(s)
        idx = local_window_select(attn, n, local_k=1)
        assert idx.numel() == n, (s, n, idx.numel())
        assert idx.unique().numel() == n
        assert (idx.sort().values == idx).all()
    print("local_window_select OK")


def test_merge_attention_frame():
    cfg = EarlyTomConfig(dominant_drop=0.5, knn_k=5)
    for s, n in [(64, 16), (64, 63), (32, 32), (16, 5)]:
        feat = torch.randn(s, 8)
        attn = torch.rand(s)
        idx, feats = merge_attention_frame(feat, attn, n, cfg)
        assert idx.numel() == n and feats.shape == (n, 8), (s, n)
        assert idx.unique().numel() == n
    # D=0 -> pure global top-k.
    cfg0 = EarlyTomConfig(dominant_drop=0.0)
    feat, attn = torch.randn(64, 8), torch.rand(64)
    idx, feats = merge_attention_frame(feat, attn, 16, cfg0)
    assert set(idx.tolist()) == set(attn.topk(16).indices.tolist())
    assert torch.equal(feats, feat[idx])
    print("merge_attention_frame OK")


def test_outer_compress_exact_counts():
    cfg = EarlyTomConfig(dominant_drop=0.25)
    S = 30  # tokens per frame
    for t_orig, t_kept, n_keep in [
        (16, 7, 120),  # normal budget
        (16, 7, 30),  # tiny budget (== EVS minimum, tokens_per_frame)
        (16, 2, 200),  # over-merged: budget > surviving tokens (deficit)
        (1, 1, 30),  # single frame
    ]:
        emb = torch.randn(t_kept * S, 8)
        origin = torch.linspace(0, t_orig - 1, t_kept).round().long().unique()
        t_k = origin.numel()
        emb = torch.randn(t_k * S, 8)
        aux = EarlyTomVideoAux(
            frame_origin=origin,
            segments=[(0, t_k - 1)],
            saliency=torch.rand(t_k * S),
        )
        out, mask = outer_compress(emb, aux, S, t_orig, n_keep, cfg)
        assert out.shape == (n_keep, 8), (t_orig, t_k, n_keep, out.shape)
        assert mask.shape == (t_orig * S,)
        assert int(mask.sum()) == n_keep
    print("outer_compress OK")


if __name__ == "__main__":
    test_ema_segments()
    test_temporal_merge()
    test_local_window_select()
    test_merge_attention_frame()
    test_outer_compress_exact_counts()
    print("\nAll EarlyTom unit tests passed.")
