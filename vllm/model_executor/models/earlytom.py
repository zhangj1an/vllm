# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EarlyTom encoder instrumentation for the Qwen2.5-VL vision tower.

The model-agnostic EarlyTom algorithm (temporal segmentation, frame mixing,
DPC-KNN, attention top-k, exact-budget outer compression) lives in
:mod:`vllm.multimodal.earlytom`, next to EVS. This module holds the part
that is specific to the Qwen2.5-VL ViT: merging frames mid-encoder inside
its flattened, window-shuffled token sequence (with rebuild of cu_seqlens,
window metadata and rope tables) and recomputing attention saliency at the
last full-attention layer.
"""

from dataclasses import dataclass, field

import numpy as np
import torch

from vllm.distributed import parallel_state
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import MMEncoderAttention
from vllm.multimodal.earlytom import (
    EarlyTomConfig,
    EarlyTomVideoAux,
    temporal_merge,
)

logger = init_logger(__name__)

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
