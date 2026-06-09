# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EarlyTom: training-free early token compression for video LLMs.

Port of EarlyTom (https://github.com/viridisGreen/EarlyTom, arXiv:2605.30010)
from LLaVA-OneVision/SigLIP to vLLM's Qwen2.5-VL.

EarlyTom compresses visual tokens in two places:

1. *In-encoder temporal frame merging* (this file: :func:`temporal_merge`):
   at selected vision-encoder layers, frames are grouped into segments by
   EMA-smoothed cosine similarity; similar adjacent middle frames within a
   segment are mixed together (token-aligned weighted average), shrinking
   the temporal axis before the remaining encoder layers run.

2. *Outer spatial compression* (this file: :func:`outer_compress`): after the
   encoder, each segment's boundary frames ("dynamic") keep their most
   salient tokens via global attention top-k plus DPC-KNN clustering of the
   remainder, while middle frames ("static") keep local-window attention
   top-k tokens to preserve spatial coverage.

The original "inner" LLM-layer compression (FastV-style merge+prune at LLM
layer k) is intentionally NOT ported: vLLM's paged KV cache and scheduler
assume a fixed prompt length, so sequence shrinking inside the LLM forward is
not representable. Outer compression is the main contribution and is fully
ported.

Integration rides on the EVS (Efficient Video Sampling) infrastructure: the
processor-side placeholder count comes from
``compute_retained_tokens_count(tokens_per_frame, T, q)`` with
``q = --video-pruning-rate``, and this module guarantees that exactly that
many embeddings are produced for every video, regardless of how many frames
the temporal stage merged.

Enable with ``VLLM_EARLYTOM=1`` together with ``--video-pruning-rate``.
Hyperparameters (defaults follow the EarlyTom repo where applicable):

- ``VLLM_EARLYTOM_T``            segmentation similarity threshold tau (0.6)
- ``VLLM_EARLYTOM_EMA``          EMA factor for similarity smoothing (0.9)
- ``VLLM_EARLYTOM_M``            max frames per temporal segment (6)
- ``VLLM_EARLYTOM_MODE``         'mixing' | 'naive_mixing' (mixing)
- ``VLLM_EARLYTOM_PRUNE_LAYERS`` vision layers applying temporal merging
                                 ("15,23,27" for the 32-layer Qwen2.5-VL ViT;
                                 the repo used "10,21,23" on 26-layer SigLIP)
- ``VLLM_EARLYTOM_D``            fraction of each frame budget routed to
                                 DPC-KNN contextual merging instead of
                                 global top-k (0.0 = pure top-k, repo default)
- ``VLLM_EARLYTOM_BETA``         center/cluster-mean mixing weight (0.6)
- ``VLLM_EARLYTOM_NO_BETA``      1 = plain mean instead of Beta-weighted
- ``VLLM_EARLYTOM_K``            k for DPC-KNN local density (7)
- ``VLLM_EARLYTOM_LOCAL_K``      tokens kept per local window in static
                                 frames (1)
"""

import math
import os
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from vllm.distributed import parallel_state
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import MMEncoderAttention

logger = init_logger(__name__)


@dataclass
class EarlyTomConfig:
    tau: float = 0.6
    ema: float = 0.9
    max_window: int = 6
    merge_mode: str = "mixing"
    prune_layers: tuple[int, ...] = (15, 23, 27)
    dominant_drop: float = 0.0  # "D" in the paper
    beta: float = 0.6
    no_beta: bool = False
    knn_k: int = 7
    local_k: int = 1

    @classmethod
    def from_env(cls) -> "EarlyTomConfig | None":
        """Build a config from VLLM_EARLYTOM* env vars.

        Returns None when EarlyTom is disabled.
        """
        if os.environ.get("VLLM_EARLYTOM", "0") not in ("1", "true", "True"):
            return None
        cfg = cls(
            tau=float(os.environ.get("VLLM_EARLYTOM_T", 0.6)),
            ema=float(os.environ.get("VLLM_EARLYTOM_EMA", 0.9)),
            max_window=int(os.environ.get("VLLM_EARLYTOM_M", 6)),
            merge_mode=os.environ.get("VLLM_EARLYTOM_MODE", "mixing"),
            prune_layers=tuple(
                int(x)
                for x in os.environ.get("VLLM_EARLYTOM_PRUNE_LAYERS", "15,23,27").split(
                    ","
                )
            ),
            dominant_drop=float(os.environ.get("VLLM_EARLYTOM_D", 0.0)),
            beta=float(os.environ.get("VLLM_EARLYTOM_BETA", 0.6)),
            no_beta=os.environ.get("VLLM_EARLYTOM_NO_BETA", "0") == "1",
            knn_k=int(os.environ.get("VLLM_EARLYTOM_K", 7)),
            local_k=int(os.environ.get("VLLM_EARLYTOM_LOCAL_K", 1)),
        )
        if cfg.merge_mode not in ("mixing", "naive_mixing"):
            raise ValueError(f"Invalid VLLM_EARLYTOM_MODE: {cfg.merge_mode}")
        return cfg


# ---------------------------------------------------------------------------
# Temporal stage (in-encoder frame merging)
# ---------------------------------------------------------------------------


def ema_segments(
    frames_normed: torch.Tensor, tau: float, ema: float, max_window: int
) -> list[tuple[int, int]]:
    """Segment frames by EMA-smoothed cosine similarity to the segment head.

    Mirrors ``SigLipEncoder.compression`` from the EarlyTom repo: a new
    segment starts when smoothed similarity to the segment's first frame
    drops below ``tau`` or the segment reaches ``max_window`` frames.

    Args:
        frames_normed: [T, S, D] L2-normalized frame features.

    Returns:
        List of inclusive (start, end) frame index pairs.
    """
    num_frames = frames_normed.size(0)
    if num_frames <= 1:
        return [(0, num_frames - 1)]

    # Per-step similarity of each frame to every other frame is not needed;
    # only to the running reference frame. Batch the dot products per
    # reference to keep GPU syncs to one per segment, not one per frame.
    segments: list[tuple[int, int]] = []
    start = 0
    sim = 1.0
    ref = frames_normed[0]
    sims_to_ref = (frames_normed * ref).sum(-1).mean(-1).tolist()  # [T]
    i = 1
    while i < num_frames:
        cur_sim = sims_to_ref[i]
        sim = ema * cur_sim + (1 - ema) * sim
        if sim < tau or (i - start + 1) >= max_window:
            segments.append((start, i - 1))
            start = i
            ref = frames_normed[i]
            sims_to_ref = (frames_normed * ref).sum(-1).mean(-1).tolist()
            sim = 1.0
        i += 1
    segments.append((start, num_frames - 1))
    return segments


def _fine_merge(
    feat: torch.Tensor,
    origin: torch.Tensor,
    tau: float,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge similar adjacent middle frames of a segment.

    Mirrors ``SigLipEncoder.fine_segment``: adjacent frame pairs whose
    token-level similarity exceeds ``tau`` (and beats the next pair) are
    mixed with a similarity-proportional weight.

    Args:
        feat: [L, S, D] middle-frame features.
        origin: [L] original frame index of each frame.

    Returns:
        (merged feat [L', S, D], merged origin [L'])
    """
    num_frames = feat.size(0)
    if num_frames < 2:
        return feat, origin

    if mode == "naive_mixing":
        return feat.mean(dim=0, keepdim=True), origin[:1]

    feat_norm = F.normalize(feat.float(), dim=-1)
    sims = (
        F.cosine_similarity(feat_norm[:-1], feat_norm[1:], dim=-1).mean(dim=-1).tolist()
    )  # [L-1]

    out_feat: list[torch.Tensor] = []
    out_origin: list[int] = []
    origin_list = origin.tolist()
    i = 0
    while i < num_frames - 1:
        cur_sim = sims[i]
        if i + 1 < num_frames - 1:
            next_sim = sims[i + 1]
            should_merge = cur_sim > tau and cur_sim > next_sim
        else:
            next_sim = 0.0
            should_merge = cur_sim > tau
        if should_merge:
            weight = cur_sim / (cur_sim + next_sim) if next_sim != 0 else 1.0
            out_feat.append(weight * feat[i] + (1 - weight) * feat[i + 1])
            out_origin.append(origin_list[i])
            i += 2
        else:
            out_feat.append(feat[i])
            out_origin.append(origin_list[i])
            i += 1
    if i == num_frames - 1:
        out_feat.append(feat[i])
        out_origin.append(origin_list[i])
    return (
        torch.stack(out_feat, dim=0),
        torch.tensor(out_origin, dtype=origin.dtype, device=origin.device),
    )


def temporal_merge(
    feat: torch.Tensor,
    origin: torch.Tensor,
    cfg: EarlyTomConfig,
    refine: bool,
) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, int]] | None]:
    """One EarlyTom temporal compression pass over the frame axis.

    Mirrors ``SigLipEncoder.compression`` + ``coarse_segment``: segment the
    frames, keep each segment's boundary frames, and fine-merge the middles.

    Args:
        feat: [T, S, D] frame features (any dtype).
        origin: [T] original frame index of each (possibly merged) frame.
        refine: when True (last prune layer) re-segment the merged frames
            and return the refined segments for the outer stage.

    Returns:
        (merged feat [T', S, D], origins [T'],
         segments over the new frame indices or None)
    """
    num_frames = feat.size(0)
    if num_frames <= 1:
        return feat, origin, [(0, num_frames - 1)] if refine else None

    normed = F.normalize(feat.float(), dim=-1)
    segments = ema_segments(normed, cfg.tau, cfg.ema, cfg.max_window)

    out_feat: list[torch.Tensor] = []
    out_origin: list[torch.Tensor] = []
    for s, e in segments:
        length = e - s + 1
        if length <= 2:
            out_feat.append(feat[s : e + 1])
            out_origin.append(origin[s : e + 1])
        else:
            # Threshold for middle-frame mixing: similarity between the
            # segment's boundary frames (per coarse_segment in the repo).
            pair_tau = float((normed[s] * normed[e]).sum(-1).mean())
            mid_feat, mid_origin = _fine_merge(
                feat[s + 1 : e], origin[s + 1 : e], pair_tau, cfg.merge_mode
            )
            out_feat.append(
                torch.cat([feat[s : s + 1], mid_feat, feat[e : e + 1]], dim=0)
            )
            out_origin.append(
                torch.cat([origin[s : s + 1], mid_origin, origin[e : e + 1]])
            )

    merged = torch.cat(out_feat, dim=0)
    merged_origin = torch.cat(out_origin, dim=0)

    new_segments: list[tuple[int, int]] | None = None
    if refine:
        merged_normed = F.normalize(merged.float(), dim=-1)
        new_segments = ema_segments(merged_normed, cfg.tau, cfg.ema, cfg.max_window)
    return merged, merged_origin, new_segments


# ---------------------------------------------------------------------------
# Outer stage (post-encoder spatial compression)
# ---------------------------------------------------------------------------


def dpc_knn_cluster(
    x: torch.Tensor, cluster_num: int, k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """DPC-KNN cluster-center selection (single frame).

    Port of ``cluster_dpc_knn`` from the EarlyTom repo, minus the random
    tie-break noise (replaced with a deterministic ramp so that all TP ranks
    select identical tokens).

    Args:
        x: [N, D] token features.

    Returns:
        (center indices [cluster_num], dist matrix [N, N])
    """
    with torch.no_grad():
        x = x.float()
        n_tokens, embed_dim = x.shape
        dist_matrix = torch.cdist(x, x) / (embed_dim**0.5)  # [N, N]

        dist_nearest, _ = torch.topk(
            dist_matrix, min(k, n_tokens), dim=-1, largest=False
        )
        density = (-(dist_nearest**2).mean(dim=-1)).exp()  # [N]
        # Deterministic tie-break (repo uses torch.rand * 1e-6).
        density = (
            density
            + torch.arange(n_tokens, device=x.device, dtype=density.dtype) * 1e-8
        )

        mask = (density[None, :] > density[:, None]).to(x.dtype)
        dist_max = dist_matrix.max()
        dist, _ = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)

        score = dist * density
        _, index_center = score.topk(cluster_num, dim=-1)
        return index_center, dist_matrix


def _merge_clusters(
    feat: torch.Tensor,
    center_idx: torch.Tensor,
    dist_matrix: torch.Tensor,
    beta: float,
    no_beta: bool,
) -> torch.Tensor:
    """Merge non-center tokens into their nearest cluster center.

    Port of ``merge_tokens_by_clustering`` (batch size 1, vectorized).

    Args:
        feat: [N, D] token features.
        center_idx: [C] cluster center token indices (sorted).
        dist_matrix: [N, N] pairwise distances from dpc_knn_cluster.

    Returns:
        [C, D] merged cluster tokens, aligned with center_idx order.
    """
    n_tokens = feat.size(0)
    num_clusters = center_idx.numel()
    device = feat.device

    is_center = torch.zeros(n_tokens, dtype=torch.bool, device=device)
    is_center[center_idx] = True
    non_center_idx = (~is_center).nonzero(as_tuple=True)[0]

    centers = feat[center_idx]  # [C, D]
    if non_center_idx.numel() == 0:
        return centers

    # Assign each non-center token to its nearest center.
    assign = dist_matrix[non_center_idx][:, center_idx].argmin(dim=-1)  # [M]

    member_sum = torch.zeros_like(centers, dtype=torch.float32)
    member_sum.index_add_(0, assign, feat[non_center_idx].float())
    member_cnt = torch.zeros(
        num_clusters, device=device, dtype=torch.float32
    ).index_add_(0, assign, torch.ones_like(assign, dtype=torch.float32))

    has_members = member_cnt > 0
    member_mean = member_sum / member_cnt.clamp(min=1).unsqueeze(-1)

    if no_beta:
        merged = (member_sum + centers.float()) / (member_cnt + 1).unsqueeze(-1)
    else:
        merged = beta * centers.float() + (1 - beta) * member_mean
    merged = torch.where(has_members.unsqueeze(-1), merged, centers.float())
    return merged.to(feat.dtype)


def merge_attention_frame(
    feat: torch.Tensor,
    attn: torch.Tensor,
    n_keep: int,
    cfg: EarlyTomConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compress one dynamic frame: global attention top-k + DPC-KNN merge.

    Port of ``merge_tokens_by_attention_frame``: the ``1 - D`` share of the
    budget goes to *dominant* tokens (global top-k by encoder attention
    saliency); the remaining ``D`` share to *contextual* tokens obtained by
    DPC-KNN clustering of the non-dominant tokens and merging each cluster.

    Args:
        feat: [S, D] frame tokens.
        attn: [S] per-token saliency.
        n_keep: exact number of tokens to keep.

    Returns:
        (kept token positions [n_keep] sorted asc, features [n_keep, D])
    """
    n_tokens = feat.size(0)
    n_keep = min(n_keep, n_tokens)
    n_dominant = min(n_tokens, round(n_keep * (1 - cfg.dominant_drop)))
    n_contextual = n_keep - n_dominant

    device = feat.device
    if n_dominant > 0:
        dom_idx = attn.topk(n_dominant).indices
    else:
        dom_idx = torch.empty(0, dtype=torch.long, device=device)

    if n_contextual > 0:
        rest_mask = torch.ones(n_tokens, dtype=torch.bool, device=device)
        rest_mask[dom_idx] = False
        rest_idx = rest_mask.nonzero(as_tuple=True)[0]
        rest_feat = feat[rest_idx]
        center_local, dist_matrix = dpc_knn_cluster(
            rest_feat, n_contextual, k=min(cfg.knn_k, rest_idx.numel())
        )
        center_local = center_local.sort().values
        ctx_feat = _merge_clusters(
            rest_feat, center_local, dist_matrix, cfg.beta, cfg.no_beta
        )
        ctx_idx = rest_idx[center_local]
    else:
        ctx_idx = torch.empty(0, dtype=torch.long, device=device)
        ctx_feat = feat.new_zeros((0, feat.size(-1)))

    all_idx = torch.cat([dom_idx, ctx_idx])
    all_feat = torch.cat([feat[dom_idx], ctx_feat], dim=0)
    order = all_idx.argsort()
    return all_idx[order], all_feat[order]


def local_window_select(attn: torch.Tensor, n_keep: int, local_k: int) -> torch.Tensor:
    """Compress one static frame: local-window attention top-k.

    Port of ``merge_tokens_by_local_window`` with exact-count semantics:
    the frame is split into ``ceil(n_keep / local_k)`` contiguous windows
    and the most salient ``local_k`` tokens of each window are kept, which
    preserves spatial coverage of low-motion frames.

    Args:
        attn: [S] per-token saliency.
        n_keep: exact number of tokens to keep (<= S).

    Returns:
        kept token positions [n_keep], sorted ascending.
    """
    n_tokens = attn.size(0)
    n_keep = min(n_keep, n_tokens)
    if n_keep == 0:
        return torch.empty(0, dtype=torch.long, device=attn.device)

    local_k = max(1, min(local_k, n_keep))
    num_windows = math.ceil(n_keep / local_k)
    if n_tokens // num_windows < local_k:
        local_k = max(1, n_tokens // num_windows)
        num_windows = math.ceil(n_keep / local_k)

    base = n_tokens // num_windows
    rem = n_tokens % num_windows
    sizes = [base + 1 if i < rem else base for i in range(num_windows)]

    picked: list[torch.Tensor] = []
    offset = 0
    for size in sizes:
        window = attn[offset : offset + size]
        k = min(local_k, size)
        idx = window.topk(k).indices + offset
        picked.append(idx)
        offset += size
    idx = torch.cat(picked)

    if idx.numel() > n_keep:
        # Uniform subsample to the exact budget (repo behavior), then the
        # final sort restores spatial order.
        step = idx.numel() / n_keep
        sel = torch.round(torch.arange(0, idx.numel(), step, device=idx.device)).long()[
            :n_keep
        ]
        idx = idx[sel]
    elif idx.numel() < n_keep:
        # Top up with the best unused tokens instead of duplicating
        # (duplicates would collide in the retention mask).
        unused = torch.ones(n_tokens, dtype=torch.bool, device=attn.device)
        unused[idx] = False
        unused_idx = unused.nonzero(as_tuple=True)[0]
        extra = attn[unused_idx].topk(n_keep - idx.numel()).indices
        idx = torch.cat([idx, unused_idx[extra]])

    return idx.sort().values


def _allocate_budget(num_frames: int, per_frame_cap: int, total: int) -> list[int]:
    """Split ``total`` tokens across frames as evenly as possible.

    Uses largest-remainder rounding so the quotas always sum to ``total``
    (each capped at ``per_frame_cap``).
    """
    total = min(total, num_frames * per_frame_cap)
    base = total // num_frames
    rem = total - base * num_frames
    quotas = [base + (1 if i < rem else 0) for i in range(num_frames)]

    # Redistribute any excess over the cap (only possible if base == cap).
    overflow = sum(max(0, q - per_frame_cap) for q in quotas)
    quotas = [min(q, per_frame_cap) for q in quotas]
    i = 0
    while overflow > 0 and i < num_frames:
        room = per_frame_cap - quotas[i]
        take = min(room, overflow)
        quotas[i] += take
        overflow -= take
        i += 1
    return quotas


@dataclass
class EarlyTomVideoAux:
    """Per-video auxiliary outputs of the EarlyTom-instrumented encoder."""

    # [T'] original frame (grid_t) index of each surviving merged frame
    frame_origin: torch.Tensor
    # final temporal segments over surviving frame indices
    segments: list[tuple[int, int]] = field(default_factory=list)
    # [T' * S] per-token saliency in grid order (S = merger tokens / frame)
    saliency: torch.Tensor | None = None


def outer_compress(
    emb: torch.Tensor,
    aux: EarlyTomVideoAux,
    tokens_per_frame: int,
    num_orig_frames: int,
    n_keep: int,
    cfg: EarlyTomConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """EarlyTom outer spatial compression for one video.

    Args:
        emb: [T' * S, D] merger-output embeddings (grid order).
        aux: encoder aux info (frame origins, segments, saliency).
        tokens_per_frame: S, LLM tokens per frame after the 2x2 merger.
        num_orig_frames: T, original temporal grid size.
        n_keep: exact number of output embeddings required (EVS count).

    Returns:
        (embeddings [n_keep, D],
         retention mask [T * S] bool over the original grid; exactly n_keep
         True entries, aligned with the row order of the embeddings)
    """
    device = emb.device
    n_frames = aux.frame_origin.numel()
    feat = emb.view(n_frames, tokens_per_frame, -1)
    saliency = aux.saliency.view(n_frames, tokens_per_frame)
    origins = aux.frame_origin.tolist()

    avail = n_frames * tokens_per_frame
    grid_size = num_orig_frames * tokens_per_frame

    # Classify each surviving frame: segment boundaries are "dynamic"
    # (attention top-k + DPC-KNN), middles are "static" (local window).
    is_dynamic = torch.zeros(n_frames, dtype=torch.bool)
    for s, e in aux.segments:
        is_dynamic[s] = True
        is_dynamic[e] = True

    quotas = _allocate_budget(n_frames, tokens_per_frame, min(n_keep, avail))

    out_feat: list[torch.Tensor] = []
    flat_idx: list[torch.Tensor] = []
    for f in range(n_frames):
        n_f = quotas[f]
        if n_f == 0:
            continue
        if is_dynamic[f]:
            idx, feats = merge_attention_frame(feat[f], saliency[f], n_f, cfg)
        else:
            idx = local_window_select(saliency[f], n_f, cfg.local_k)
            feats = feat[f][idx]
        out_feat.append(feats)
        flat_idx.append(idx + origins[f] * tokens_per_frame)

    emb_out = torch.cat(out_feat, dim=0)
    positions = torch.cat(flat_idx)

    # Over-merged below the budget: replicate surviving frames into the
    # original-grid slots of their merged-away neighbours so the embedding
    # count still matches the processor's placeholder count exactly.
    deficit = n_keep - emb_out.size(0)
    if deficit > 0:
        origin_set = set(origins)
        donors: list[tuple[int, int]] = []  # (orig_frame, surviving frame)
        for t in range(num_orig_frames):
            if t not in origin_set:
                nearest = min(range(n_frames), key=lambda f: abs(origins[f] - t))
                donors.append((t, nearest))
        extra_feat: list[torch.Tensor] = []
        extra_idx: list[torch.Tensor] = []
        for orig_t, f in donors:
            if deficit <= 0:
                break
            take = min(deficit, tokens_per_frame)
            idx = saliency[f].topk(take).indices.sort().values
            extra_feat.append(feat[f][idx])
            extra_idx.append(idx + orig_t * tokens_per_frame)
            deficit -= take
        if deficit > 0:
            raise RuntimeError(
                "EarlyTom could not satisfy the token budget: "
                f"need {n_keep}, grid {grid_size}"
            )
        emb_out = torch.cat([emb_out] + extra_feat, dim=0)
        positions = torch.cat([positions] + extra_idx)

    # Final embeddings must be in original-grid order to match the
    # retention-mask order used for mrope positions.
    order = positions.argsort()
    emb_out = emb_out[order]
    positions = positions[order]

    mask = torch.zeros(grid_size, dtype=torch.bool, device=device)
    mask[positions] = True
    if int(mask.sum()) != n_keep:
        raise RuntimeError(
            "EarlyTom produced duplicate grid positions: "
            f"{int(mask.sum())} unique vs {n_keep} required"
        )
    return emb_out, mask


# ---------------------------------------------------------------------------
# Encoder instrumentation (frame merging mid-ViT + saliency capture)
# ---------------------------------------------------------------------------


@dataclass
class _VideoItem:
    grid_t: int  # original temporal grid size
    t: int  # current (post-merging) number of frames
    f: int  # patch tokens per frame (h * w)
    g: int  # merger tokens per frame (f / spatial_merge_unit)
    origin: torch.Tensor  # [t] original frame index per surviving frame
    # single-frame window-attention pattern (group-level permutation and
    # token-level cumulative window boundaries), identical for every frame
    window_index_1f: torch.Tensor
    cu_window_1f: np.ndarray
    segments: list[tuple[int, int]] = field(default_factory=list)
    saliency: torch.Tensor | None = None  # [t * f] window order, token level


class EarlyTomEncoderState:
    """Tracks EarlyTom state through one Qwen2.5-VL vision tower forward.

    The Qwen2.5-VL ViT flattens all videos into one token sequence in
    *window order*; each frame is a contiguous block of ``f`` tokens with an
    identical internal spatial permutation, full attention runs per frame,
    and window attention runs within (intra-frame) windows. Merging whole
    frames therefore only requires weighted-averaging aligned frame blocks
    and rebuilding the per-frame attention metadata, rope tables (which
    repeat every frame), and the merger's reverse permutation.
    """

    def __init__(self, cfg: EarlyTomConfig, vit, grid_thw_list):
        self.cfg = cfg
        self.vit = vit
        last_fullatt = max(vit.fullatt_block_indexes)
        self.saliency_layer = last_fullatt
        if max(cfg.prune_layers) >= last_fullatt:
            raise ValueError(
                f"EarlyTom prune layers {cfg.prune_layers} must lie before "
                f"the last full-attention layer ({last_fullatt}) so that "
                "saliency is computed on the final merged frames"
            )
        self.items: list[_VideoItem] = []
        for t, h, w in grid_thw_list:
            f = h * w
            w1f, cu1f = vit.get_window_index_thw(1, h, w)
            self.items.append(
                _VideoItem(
                    grid_t=t,
                    t=t,
                    f=f,
                    g=f // vit.spatial_merge_unit,
                    origin=torch.arange(t),
                    window_index_1f=w1f,
                    cu_window_1f=cu1f.numpy().astype(np.int64),
                )
            )

    def merge_after_layer(
        self,
        hidden: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        last: bool,
    ):
        """Apply temporal frame merging after a prune layer.

        Args:
            hidden: [s, 1, C] window-ordered hidden states.
            rope_cos/rope_sin: [s, rope_dim] current rope tables.
            last: whether this is the final prune layer (triggers the
                segment refinement that defines the outer-stage segments).

        Returns:
            (hidden', rope_cos', rope_sin', attn metadata dict)
        """
        new_hidden: list[torch.Tensor] = []
        new_cos: list[torch.Tensor] = []
        new_sin: list[torch.Tensor] = []
        offset = 0
        for item in self.items:
            n_tok = item.t * item.f
            block = hidden[offset : offset + n_tok, 0]
            feat = block.view(item.t, item.f, -1)
            merged, origin, segments = temporal_merge(
                feat, item.origin.to(feat.device), self.cfg, refine=last
            )
            t_new = merged.size(0)
            # Rope rows repeat every frame; keep the first t_new frames.
            new_cos.append(rope_cos[offset : offset + t_new * item.f])
            new_sin.append(rope_sin[offset : offset + t_new * item.f])
            new_hidden.append(merged.reshape(t_new * item.f, 1, -1))
            item.t = t_new
            item.origin = origin.cpu()
            if last:
                item.segments = segments or [(0, t_new - 1)]
            offset += n_tok

        hidden = torch.cat(new_hidden, dim=0)
        rope_cos = torch.cat(new_cos, dim=0)
        rope_sin = torch.cat(new_sin, dim=0)
        return hidden, rope_cos, rope_sin, self._build_attn_metadata(hidden.device)

    def _build_attn_metadata(self, device: torch.device) -> dict:
        """Rebuild cu_seqlens & friends for the current frame counts."""
        lens_full: list[int] = []
        cu_window: list[int] = [0]
        offset = 0
        for item in self.items:
            lens_full.extend([item.f] * item.t)
            for j in range(item.t):
                cu_window.extend((item.cu_window_1f + j * item.f + offset).tolist())
            offset += item.t * item.f
        cu_full_np = np.concatenate([[0], np.cumsum(lens_full)]).astype(np.int32)
        cu_window_np = np.asarray(cu_window, dtype=np.int32)

        vit = self.vit
        backend = vit.attn_backend
        meta = {
            "max_seqlen_full": torch.tensor(
                MMEncoderAttention.compute_max_seqlen(backend, cu_full_np),
                dtype=torch.int32,
            ),
            "max_seqlen_window": torch.tensor(
                MMEncoderAttention.compute_max_seqlen(backend, cu_window_np),
                dtype=torch.int32,
            ),
            "sequence_lengths_full": MMEncoderAttention.maybe_compute_seq_lens(
                backend, cu_full_np, device
            ),
            "sequence_lengths_window": (
                MMEncoderAttention.maybe_compute_seq_lens(backend, cu_window_np, device)
            ),
            "cu_seqlens": MMEncoderAttention.maybe_recompute_cu_seqlens(
                backend,
                cu_full_np,
                vit.hidden_size,
                vit.tp_size,
                device,
                fp8_padded_hidden_size=vit.fp8_padded_hidden_size,
            ),
            "cu_window_seqlens": MMEncoderAttention.maybe_recompute_cu_seqlens(
                backend,
                cu_window_np,
                vit.hidden_size,
                vit.tp_size,
                device,
                fp8_padded_hidden_size=vit.fp8_padded_hidden_size,
            ),
        }
        return meta

    @torch.no_grad()
    def capture_saliency(
        self,
        blk,
        hidden: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> None:
        """Compute per-token attention saliency at one vision layer.

        Recomputes the layer's attention probabilities from its input
        (norm1 -> qkv -> rope -> softmax(QK^T)), per frame, and averages
        them over heads and query positions — the vLLM equivalent of the
        head/query-averaged encoder attention EarlyTom reads out of SigLIP.
        Backend-agnostic: works with flash attention since the probabilities
        are recomputed explicitly (one frame at a time to bound memory).
        """
        attn = blk.attn
        n_heads = attn.num_attention_heads_per_partition
        head_dim = attn.hidden_size_per_attention_head
        scale = head_dim**-0.5

        xn = blk.norm1(hidden)
        qkv_out, _ = attn.qkv(xn)
        seq_len, bsz, _ = qkv_out.shape
        qkv = qkv_out.view(seq_len, bsz, 3, n_heads, head_dim).permute(
            1, 0, 2, 3, 4
        )  # [b, s, 3, H, d]

        if rope_cos is not None and rope_sin is not None:
            qk = qkv[:, :, :2]  # [b, s, 2, H, d]
            qk = (
                qk.permute(2, 0, 1, 3, 4)
                .reshape(2 * bsz, seq_len, n_heads, head_dim)
                .contiguous()
            )
            qk = attn.apply_rotary_emb(qk, rope_cos, rope_sin)
            qk = qk.view(2, bsz, seq_len, n_heads, head_dim)
            q, k = qk.unbind(dim=0)
        else:
            q, k = qkv[:, :, 0], qkv[:, :, 1]

        q = q[0].float()  # [s, H, d]
        k = k[0].float()

        # Chunk the query axis so the [H, chunk, f] probability tensor stays
        # bounded regardless of frame size (the profiling dummy can be one
        # enormous frame).
        chunk_bytes = 256 * 1024**2

        offset = 0
        for item in self.items:
            sal_frames: list[torch.Tensor] = []
            for _ in range(item.t):
                q_f = q[offset : offset + item.f].permute(1, 0, 2)  # [H,f,d]
                k_f = k[offset : offset + item.f].permute(1, 0, 2)
                k_t = k_f.transpose(1, 2)
                chunk = max(1, min(item.f, chunk_bytes // (n_heads * item.f * 4)))
                sal = q.new_zeros(item.f)
                for s0 in range(0, item.f, chunk):
                    probs = torch.softmax(
                        torch.bmm(q_f[:, s0 : s0 + chunk], k_t) * scale,
                        dim=-1,
                    )
                    # Accumulate sum over (local) heads and queries -> [f].
                    sal += probs.sum(dim=(0, 1))
                # Mean over query positions.
                sal_frames.append(sal / item.f)
                offset += item.f
            item.saliency = torch.cat(sal_frames)

        # With TP > 1 the vision attention heads are sharded; reduce so all
        # ranks select identical tokens downstream.
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        if tp_size > 1:
            full = torch.cat([item.saliency for item in self.items])
            full = parallel_state.get_tp_group().all_reduce(full)
            full /= n_heads * tp_size
            offset = 0
            for item in self.items:
                item.saliency = full[offset : offset + item.t * item.f]
                offset += item.t * item.f
        else:
            for item in self.items:
                item.saliency = item.saliency / n_heads

    def final_reverse_indices(self, device: torch.device) -> torch.Tensor:
        """Merger-output (group-level) window->grid reverse permutation."""
        parts: list[torch.Tensor] = []
        offset = 0
        for item in self.items:
            wi = torch.cat([item.window_index_1f + j * item.g for j in range(item.t)])
            inv = torch.empty_like(wi)
            inv[wi] = torch.arange(wi.numel(), dtype=wi.dtype)
            parts.append(inv + offset)
            offset += item.t * item.g
        return torch.cat(parts).to(device)

    def build_aux(self) -> list[EarlyTomVideoAux]:
        """Per-video aux outputs with saliency pooled to merger tokens and
        reordered to grid order (matching the merger output rows)."""
        unit = self.vit.spatial_merge_unit
        aux_list: list[EarlyTomVideoAux] = []
        for item in self.items:
            assert item.saliency is not None
            group_sal = item.saliency.view(-1, unit).mean(dim=-1)  # window
            wi = torch.cat([item.window_index_1f + j * item.g for j in range(item.t)])
            inv = torch.empty_like(wi)
            inv[wi] = torch.arange(wi.numel(), dtype=wi.dtype)
            grid_sal = group_sal[inv.to(group_sal.device)]
            aux_list.append(
                EarlyTomVideoAux(
                    frame_origin=item.origin,
                    segments=item.segments or [(0, item.t - 1)],
                    saliency=grid_sal,
                )
            )
        return aux_list
