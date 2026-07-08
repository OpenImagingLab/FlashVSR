import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import os
import time
from typing import Tuple, Optional, List
from einops import rearrange
from .utils import hash_state_dict_keys
from ..nvtx_utils import nvtx_range
# Phase 2B-2: FP8 GEMM infrastructure (FLASHVSR_FP8_GEMM, default OFF).
from . import fp8_gemm as _fp8g

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    from sageattention import sageattn
    SAGE_ATTN_AVAILABLE = True
except ModuleNotFoundError:
    SAGE_ATTN_AVAILABLE = False

from block_sparse_attn import block_sparse_attn_func
from PIL import Image
import numpy as np

# ---------------------------------------------------------------------------
# Hopper adaptive attention backend.
#
# Measured on GH200 @768x1408 (seq=25344, 12 heads, dim=128): the block-sparse
# self-attention runs at block-mask density ~0.606 and takes ~7.3 ms, while
# cuDNN's fused dense SDPA computes the FULL attention in ~6.5 ms (605 TFLOP/s).
# i.e. at high density the sparse kernel is slower than dense AND drops context.
#
# When the block mask is dense enough (density >= threshold) we route to cuDNN
# fused dense attention instead of the sparse kernel: faster and uses full
# context. Below the threshold the sparse kernel wins, so we keep it.
#
# Knob: FLASHVSR_ATTN_BACKEND = sparse | triton | triton2 | auto | dense
#   sparse -> always block_sparse (DEFAULT = original behaviour, no quality change)
#   triton -> Hopper WGMMA block-sparse kernel: exact same mask, output matches
#             block_sparse very closely (~49.7 dB PSNR end-to-end; the only
#             difference is WGMMA vs HMMA accumulation order). ~1.2x faster than
#             block_sparse at the kernel level (~+9% end-to-end). sm_90 only;
#             silently falls back to block_sparse elsewhere / on error.
#   triton2 -> Phase-3 warp-specialized (producer/consumer + pingpong) Gluon
#             kernel: exact same mask/CSR, ~1.5x the v1 kernel at the steady
#             shape. sm_90 only; falls back v2 -> v1 triton -> block_sparse.
#   auto   -> density-adaptive: dense if density>=FLASHVSR_ATTN_DENSE_THRESH
#   dense  -> always cuDNN fused dense
# FLASHVSR_ATTN_DENSE_THRESH default 0.5 (the measured crossover point).
#
# NOTE: routing to dense changes the output (it uses FULL attention instead of
# the trained locality-constrained sparse pattern, which measurably lowers PSNR),
# and at the default density the E2E gain is negligible, so the default stays
# 'sparse'. The knob exists for future aggressive-sparsity experiments (lower
# topk -> lower density -> sparse wins big, see docs).
# ---------------------------------------------------------------------------
_ATTN_BACKEND = os.environ.get("FLASHVSR_ATTN_BACKEND", "sparse").lower()
_ATTN_DENSE_THRESH = float(os.environ.get("FLASHVSR_ATTN_DENSE_THRESH", "0.5"))

# ---------------------------------------------------------------------------
# Norm / elementwise fusion (Phase 3-B).
#
# The DiT block is dominated (~17% of denoise GPU time) by memory-bound
# elementwise kernels: RMSNorm (q/k, with an fp32 up/down cast), LayerNorm,
# modulate (x*(1+scale)+shift) and the gate (x + gate*residual). torch.compile
# fuses each chain into a single kernel; in isolation the fused RMSNorm is
# several times faster than the eager path (the exact ratio depends on the
# tensor shape), at near-identical precision (cos ~1.0).
# These functions contain no attention / custom kernels, so compiling them does
# not interact with the block_sparse path or the streaming cache. End-to-end the
# fusion is measured at ~49 dB PSNR vs the unfused path (see test_fuse_norm.py).
#
# Knob FLASHVSR_FUSE_NORM = 0 | 1 (default off; opt-in, parity-gated).
# ---------------------------------------------------------------------------
_FUSE_NORM = os.environ.get("FLASHVSR_FUSE_NORM", "0") != "0"

# ---------------------------------------------------------------------------
# Lossless step-invariant caches (Phase B).
#
# Several per-block elementwise results are recomputed on every denoise step but
# depend only on fixed inputs (parameters + the once-computed timestep modulation
# t_mod), NOT on x / q / k. Caching them is bit-identical (max|diff|==0).
#
# B1  FLASHVSR_CACHE_MOD       = 0 | 1  -> cache (modulation + t_mod).chunk(...)
#                                          per DiTBlock / Head.
# B2  FLASHVSR_CACHE_MASK_BIAS = 0 | 1  -> cache the local_attn_mask additive bias
#                                          (0/-inf) in generate_draft_block_mask.
# Both default OFF (opt-in), pure elementwise, silent fallback. They do not touch
# attention / the streaming cache, so they compose with every other knob.
# ---------------------------------------------------------------------------
_CACHE_MOD = os.environ.get("FLASHVSR_CACHE_MOD", "0") != "0"
_CACHE_MASK_BIAS = os.environ.get("FLASHVSR_CACHE_MASK_BIAS", "0") != "0"

# ---------------------------------------------------------------------------
# Phase 2A-4: mask-generation allocation/sync cleanup (FLASHVSR_MASKGEN_LEAN,
# default OFF). Exact-semantics only — the produced boolean mask is identical:
#   (a) threshold via torch.kthvalue(n-k) instead of topk(k+1).values[:,-1]
#       — the same order statistic (ties included), computed by one
#       radix-select kernel without materializing the (rows, k+1) values +
#       int64 indices tensors (topk here selects ~45% of 17k elements/row);
#   (b) drop the no-op `.repeat(1,1,1,1)` copy of the boolean mask (B==1 is
#       asserted in generate_draft_block_mask);
#   (c) sparse backend only: cache the cu_seqlens_q/k + head_mask_type int32
#       tensors keyed on (seqlen, seqlen_kv, heads, device) — their per-call
#       `torch.tensor(..., device=...)` construction is a hidden H2D sync
#       (ANALYSIS §3, sparse-backend idle).
# ---------------------------------------------------------------------------
_MASKGEN_LEAN = os.environ.get("FLASHVSR_MASKGEN_LEAN", "0") != "0"

# (c): persistent per-shape int32 tensors for the block_sparse_attn call.
_SPARSE_SEQLENS_CACHE = {}


def _maybe_compile(fn):
    if not _FUSE_NORM:
        return fn
    try:
        return torch.compile(fn, dynamic=True)
    except Exception:
        return fn

try:
    from torch.nn.attention import sdpa_kernel as _sdpa_kernel, SDPBackend as _SDPBackend
    _CUDNN_SDPA_OK = True
except Exception:
    _CUDNN_SDPA_OK = False

# Triton WGMMA block-sparse attention kernel (optional; sm_90 fast path).
try:
    from .triton_block_sparse_attn import triton_block_sparse_attention as _TRITON_BSA
except Exception:
    _TRITON_BSA = None

# ---------------------------------------------------------------------------
# Phase 2A-3: strided attention IO (FLASHVSR_ATTN_STRIDED_IO, default OFF).
#
# The triton-backend glue below performs three (S,n,d)->(n,S,d) .contiguous()
# transposes for q/k/v plus a transpose of the output — ~11.6 ms/chunk of
# SM-bound strided copies (ANALYSIS §1.2/§2.3). The strided-IO variant keeps
# tensors in the glue's natural token-major (S, n, d) layout end to end:
# TMA descriptors address per-head tiles inside the 2D (S, n*d) view, and the
# kernel stores the output directly in (S, n, d). Same tile schedule and
# accumulation order -> gated on max|diff| kernel-level + E2E PSNR >= 49 dB.
# On any failure the glue falls back to the contiguous triton path (NOT to
# the sparse backend), preserving the existing fallback ladder.
# ---------------------------------------------------------------------------
_ATTN_STRIDED_IO = os.environ.get("FLASHVSR_ATTN_STRIDED_IO", "0") != "0"
try:
    from .triton_block_sparse_attn import triton_block_sparse_attention_snd as _TRITON_BSA_SND
except Exception:
    _TRITON_BSA_SND = None

# ---------------------------------------------------------------------------
# Phase 3: warp-specialized block-sparse attention v2
# (FLASHVSR_ATTN_BACKEND=triton2, sm_90 + Gluon only).
#
# FA3-style producer/consumer rewrite of the block-sparse kernel: a TMA
# producer warp group streams the CSR-selected K/V blocks into a ring buffer
# while two 64-row consumer warpgroups pingpong softmax against WGMMA
# (12 warps/SM vs v1's barrier-locked 8). Exact same boolean mask, same CSR,
# same (S, n, d) glue interface as the strided-IO path. Fallback ladder on
# ANY failure: v2 -> v1 triton (strided -> contiguous) -> block_sparse_attn.
# ---------------------------------------------------------------------------
try:
    from .triton_block_sparse_attn_v2 import (
        triton_block_sparse_attention_v2 as _TRITON_BSA_V2)
except Exception:
    _TRITON_BSA_V2 = None


def _is_hopper_dev(device):
    try:
        return device.type == "cuda" and torch.cuda.get_device_capability(device) == (9, 0)
    except Exception:
        return False


def _dense_sdpa(q_bnsd):
    """cuDNN fused dense attention on (B, heads, S, D) layout."""
    if _CUDNN_SDPA_OK:
        with _sdpa_kernel(_SDPBackend.CUDNN_ATTENTION):
            return F.scaled_dot_product_attention(*q_bnsd)
    return F.scaled_dot_product_attention(*q_bnsd)


# ----------------------------
# Local / window masks
# ----------------------------
@torch.no_grad()
def build_local_block_mask_shifted_vec(block_h: int,
                                       block_w: int,
                                       win_h: int = 6,
                                       win_w: int = 6,
                                       include_self: bool = True,
                                       device=None) -> torch.Tensor:
    device = device or torch.device("cpu")
    H, W = block_h, block_w
    r = torch.arange(H, device=device)
    c = torch.arange(W, device=device)
    YY, XX = torch.meshgrid(r, c, indexing="ij")
    r_all = YY.reshape(-1)
    c_all = XX.reshape(-1)
    r_half = win_h // 2
    c_half = win_w // 2
    start_r = torch.clamp(r_all - r_half, 0, H - win_h)
    end_r   = start_r + win_h - 1
    start_c = torch.clamp(c_all - c_half, 0, W - win_w)
    end_c   = start_c + win_w - 1
    in_row = (r_all[None, :] >= start_r[:, None]) & (r_all[None, :] <= end_r[:, None])
    in_col = (c_all[None, :] >= start_c[:, None]) & (c_all[None, :] <= end_c[:, None])
    mask = in_row & in_col
    if not include_self:
        mask.fill_diagonal_(False)
    return mask

@torch.no_grad()
def build_local_block_mask_shifted_vec_normal_slide(block_h: int,
                                                   block_w: int,
                                                   win_h: int = 6,
                                                   win_w: int = 6,
                                                   include_self: bool = True,
                                                   device=None) -> torch.Tensor:
    device = device or torch.device("cpu")
    H, W = block_h, block_w
    r = torch.arange(H, device=device)
    c = torch.arange(W, device=device)
    YY, XX = torch.meshgrid(r, c, indexing="ij")
    r_all = YY.reshape(-1)
    c_all = XX.reshape(-1)
    r_half = win_h // 2
    c_half = win_w // 2
    start_r = r_all - r_half
    end_r   = start_r + win_h - 1
    start_c = c_all - c_half
    end_c   = start_c + win_w - 1
    in_row = (r_all[None, :] >= start_r[:, None]) & (r_all[None, :] <= end_r[:, None])
    in_col = (c_all[None, :] >= start_c[:, None]) & (c_all[None, :] <= end_c[:, None])
    mask = in_row & in_col
    if not include_self:
        mask.fill_diagonal_(False)
    return mask


class WindowPartition3D:
    """Partition / reverse-partition helpers for 5-D tensors (B,F,H,W,C)."""
    @staticmethod
    def partition(x: torch.Tensor, win: Tuple[int, int, int]):
        B, F, H, W, C = x.shape
        wf, wh, ww = win
        assert F % wf == 0 and H % wh == 0 and W % ww == 0, "Dims must divide by window size."
        x = x.view(B, F // wf, wf, H // wh, wh, W // ww, ww, C)
        x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
        return x.view(-1, wf * wh * ww, C)

    @staticmethod
    def reverse(windows: torch.Tensor, win: Tuple[int, int, int], orig: Tuple[int, int, int]):
        F, H, W = orig
        wf, wh, ww = win
        nf, nh, nw = F // wf, H // wh, W // ww
        B = windows.size(0) // (nf * nh * nw)
        x = windows.view(B, nf, nh, nw, wf, wh, ww, -1)
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
        return x.view(B, F, H, W, -1)


# ---------------------------------------------------------------------------
# Phase 2A-2: KV cache arena (FLASHVSR_KV_RINGBUF, default OFF).
#
# The streaming self-attention KV cache is maintained today as
#   k_w = torch.cat([pre_cache_k, k_w_new], dim=0)   # copies the WHOLE window
#   cache_k = k_w[one_len:]                          # trim = slice view
# i.e. every chunk copies pre+new (~264 window-blocks @768) per tensor per
# block at HBM bandwidth (kv_cat = 3.4 ms/chunk, ANALYSIS §1.2) even though
# only `one_len` (~66) windows are new.
#
# The arena replaces this with a preallocated per-block buffer of
# (kv_len + 1 + SPARE) temporal slots × one_len windows:
#   - new windows are partition-written directly into the tail (this replaces
#     the .contiguous() materialization inside WindowPartition3D.partition, so
#     it adds no traffic),
#   - the live KV window is the contiguous slice buf[start : start+length] —
#     same values, same temporal order, same contiguity as the cat result,
#   - trimming the oldest slot advances `start` (no copy),
#   - when the tail reaches capacity the live window is compacted to offset 0
#     (amortized once every SPARE chunks; overlap-safe).
# Net: ~264 → ~66+198/(SPARE+1) window-copies per chunk per tensor, at the
# cost of SPARE extra slots of retained memory per tensor.
#
# Values and ordering are bit-identical to the cat path (copies only, no
# arithmetic), so this is gated by max|diff|==0 across a full multi-chunk
# clip (exercises rotation + compaction). Works with every attention backend
# (the consumed view is contiguous, exactly like the cat result).
# ---------------------------------------------------------------------------
_KV_RINGBUF = os.environ.get("FLASHVSR_KV_RINGBUF", "0") != "0"
_KV_RINGBUF_SPARE = max(1, int(os.environ.get("FLASHVSR_KV_RINGBUF_SPARE", "2")))


class _KVArena:
    """Sliding-window KV arena; flows through the pre_cache_k/v slots."""
    __slots__ = ("buf", "start", "length", "one_len")

    def __init__(self, kv_len, one_len, block_s, dim, dtype, device):
        cap = (kv_len + 1 + _KV_RINGBUF_SPARE) * one_len
        self.buf = torch.empty(cap, block_s, dim, dtype=dtype, device=device)
        self.start = 0
        self.length = 0
        self.one_len = one_len

    def append_partition(self, x, win):
        """Window-partition `x` (B=1, f, h, w, C) directly into the arena tail
        and return the live contiguous view buf[start : start+length].

        The write performs exactly the permute-copy that
        WindowPartition3D.partition's .contiguous() would perform, with the
        arena slice as destination -> bit-identical values/order."""
        B, F_, H_, W_, C = x.shape
        wf, wh, ww = win
        n_new = (F_ // wf) * (H_ // wh) * (W_ // ww)
        cap = self.buf.shape[0]
        if self.start + self.length + n_new > cap:
            # Compact the live window to offset 0 (kv_cat residual, amortized).
            with nvtx_range("kv_cat"):
                if self.start < self.length:
                    # overlapping ranges: stage through a temp (copy_ on
                    # overlapping storage is undefined). Only possible with
                    # very tight SPARE settings.
                    tmp = self.buf[self.start:self.start + self.length].clone()
                    self.buf[:self.length].copy_(tmp)
                else:
                    self.buf[:self.length].copy_(
                        self.buf[self.start:self.start + self.length])
                self.start = 0
        dst = self.buf[self.start + self.length:
                       self.start + self.length + n_new]
        src = x.view(B, F_ // wf, wf, H_ // wh, wh, W_ // ww, ww, C) \
               .permute(0, 1, 3, 5, 2, 4, 6, 7)
        dst.view(B, F_ // wf, H_ // wh, W_ // ww, wf, wh, ww, C).copy_(src)
        self.length += n_new
        return self.buf[self.start:self.start + self.length]

    def trim(self, n_windows):
        """Drop the oldest n_windows (pointer advance, no data movement)."""
        self.start += n_windows
        self.length -= n_windows


# B2: process-wide cache for the geometry-only additive attention bias.
_MASK_BIAS_CACHE = {}


def _build_mask_bias(local_attn_mask, repeat_head, repeat_len, repeat_num):
    """Build the (repeat_head, S, S) 0/-inf additive bias from the boolean local
    block mask. Exact re-expression of the original inline code (lines below) so
    cached and uncached paths are bit-identical."""
    m = local_attn_mask.unsqueeze(1).unsqueeze(0).repeat(repeat_len, 1, repeat_num, 1)
    m = rearrange(m, 'x a y b -> (x a) (y b)')
    m = m.unsqueeze(0).repeat(repeat_head, 1, 1)
    m = m.to(torch.float32)
    m = m.masked_fill(m == False, -float('inf'))
    m = m.masked_fill(m == True, 0)
    return m


@torch.no_grad()
def generate_draft_block_mask(batch_size, nheads, seqlen,
                              q_w, k_w, topk=10, local_attn_mask=None):
    assert batch_size == 1, "Only batch_size=1 supported for now"
    assert local_attn_mask is not None, "local_attn_mask must be provided"
    avgpool_q = torch.mean(q_w, dim=1) 
    avgpool_k = torch.mean(k_w, dim=1)
    avgpool_q = rearrange(avgpool_q, 's (h d) -> s h d', h=nheads)
    avgpool_k = rearrange(avgpool_k, 's (h d) -> s h d', h=nheads)
    q_heads = avgpool_q.permute(1, 0, 2)
    k_heads = avgpool_k.permute(1, 0, 2)
    D = avgpool_q.shape[-1]
    scores = torch.einsum("hld,hmd->hlm", q_heads, k_heads) / math.sqrt(D)

    repeat_head = scores.shape[0]
    repeat_len = scores.shape[1] // local_attn_mask.shape[0]
    repeat_num = scores.shape[2] // local_attn_mask.shape[1]

    # B2 lossless cache: the additive bias (0 / -inf) depends only on the geometry
    # (local_attn_mask + repeat factors), not on q/k, so it is identical across
    # every block and every step. Recomputing it (repeat + 2x masked_fill + cast)
    # each call is pure overhead. Cache it keyed on those shape-only inputs.
    bias = None
    if _CACHE_MASK_BIAS:
        key = (id(local_attn_mask), repeat_head, repeat_len, repeat_num,
               local_attn_mask.device, scores.shape[1], scores.shape[2])
        bias = _MASK_BIAS_CACHE.get(key)
        if bias is None:
            bias = _build_mask_bias(local_attn_mask, repeat_head, repeat_len, repeat_num)
            _MASK_BIAS_CACHE[key] = bias
    if bias is None:
        bias = _build_mask_bias(local_attn_mask, repeat_head, repeat_len, repeat_num)
    scores = scores + bias

    attn_map = torch.softmax(scores, dim=-1)
    attn_map = rearrange(attn_map, 'h (it s1) s2 -> (h it) s1 s2', it=seqlen)
    loop_num, s1, s2 = attn_map.shape
    flat = attn_map.reshape(loop_num, -1)
    n = flat.shape[1]
    apply_topk = min(flat.shape[1]-1, topk)
    if _MASKGEN_LEAN:
        # 2A-4(a): (apply_topk+1)-th largest == (n-apply_topk)-th smallest.
        # Identical exact value (order statistic, ties and all); single
        # radix-select kernel, no (rows, k+1) values/indices materialization.
        thresholds = torch.kthvalue(flat, n - apply_topk, dim=1).values
    else:
        thresholds = torch.topk(flat, k=apply_topk + 1, dim=1, largest=True).values[:, -1]
    thresholds = thresholds.unsqueeze(1)
    mask_new = (flat > thresholds).reshape(loop_num, s1, s2)
    mask_new = rearrange(mask_new, '(h it) s1 s2 -> h (it s1) s2', it=seqlen)  # keep shape note
    # 修正：上行变量名统一
    # mask_new = rearrange(attn_map, 'h (it s1) s2 -> h (it s1) s2', it=seqlen) * 0 + mask_new
    if _MASKGEN_LEAN and batch_size == 1:
        # 2A-4(b): batch_size==1 (asserted above) -> repeat(1,1,1,1) is a
        # full copy with no semantic effect; a view is enough.
        mask = mask_new.unsqueeze(0)
    else:
        mask = mask_new.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    return mask


@torch.no_grad()
def generate_causal_block_mask(batch_size, nheads, seqlen, local_num, window_size, device='cuda', train_img=False):
    i = torch.arange(seqlen, device=device).view(-1, 1)
    j = torch.arange(seqlen, device=device).view(1, -1)
    causal_mask = (j <= i) & (j >= i - local_num + 1)
    causal_mask[0,1] = True
    causal_mask[:2,2] = True
    if train_img:
        causal_mask[-1, :-1] = False
    causal_mask = causal_mask.unsqueeze(1).unsqueeze(-1).repeat(1, window_size, 1, window_size)
    causal_mask = rearrange(causal_mask, 'a n1 b n2 -> (a n1) (b n2)')
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).repeat(batch_size, nheads, 1, 1)
    return causal_mask


# ----------------------------
# Attention kernels
# ----------------------------
def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, compatibility_mode=False, attention_mask=None, return_KV=False):
    if attention_mask is not None:
        seqlen = q.shape[1]
        seqlen_kv = k.shape[1]
        q = rearrange(q, "b s (n d) -> (b s) n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> (b s) n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> (b s) n d", n=num_heads)
        base_blockmask = attention_mask

        # Adaptive backend: route dense when the sparse mask is too dense to pay off.
        use_dense = False
        if _ATTN_BACKEND == "dense":
            use_dense = True
        elif _ATTN_BACKEND == "auto" and _is_hopper_dev(q.device):
            try:
                density = base_blockmask.float().mean().item()
            except Exception:
                density = 1.0
            use_dense = density >= _ATTN_DENSE_THRESH

        if use_dense:
            try:
                # (S, n, d) -> (1, n, S, d) for fused dense SDPA, then back.
                qd = q.unsqueeze(0).transpose(1, 2)
                kd = k.unsqueeze(0).transpose(1, 2)
                vd = v.unsqueeze(0).transpose(1, 2)
                xd = _dense_sdpa((qd, kd, vd))             # (1, n, S, d)
                x = xd.transpose(1, 2)                      # (1, S, n, d)
                return rearrange(x, "b s n d -> b s (n d)", n=num_heads)
            except Exception:
                torch.cuda.empty_cache()  # fall back to sparse

        # Triton WGMMA block-sparse backend (opt-in, Hopper-guarded, exact mask).
        if _ATTN_BACKEND in ("triton", "triton2") and _is_hopper_dev(q.device) and _TRITON_BSA is not None:
            try:
                # block mask -> (H, Nqb, Nkvb) bool
                bm = base_blockmask
                if bm.dim() == 4:
                    bm = bm[0]
                bm = bm.bool()
                # Phase 3: warp-specialized v2 kernel. Falls back to the v1
                # triton paths below on any error (then to block_sparse).
                if _ATTN_BACKEND == "triton2" and _TRITON_BSA_V2 is not None:
                    try:
                        xh = _TRITON_BSA_V2(q, k, v, bm)    # (S, n, d)
                        return xh.reshape(1, xh.shape[0], -1)
                    except Exception:
                        torch.cuda.empty_cache()  # fall back to v1 triton
                # 2A-3: strided IO — q/k/v stay (S, n, d), output comes back
                # (S, n, d); zero transpose/contiguous copies in the glue.
                if _ATTN_STRIDED_IO and _TRITON_BSA_SND is not None:
                    try:
                        xh = _TRITON_BSA_SND(q, k, v, bm)   # (S, n, d)
                        return xh.reshape(1, xh.shape[0], -1)
                    except Exception:
                        torch.cuda.empty_cache()  # fall back to contiguous triton
                # contiguous path: q/k/v (S,n,d) -> (n,S,d)
                qh = q.transpose(0, 1).contiguous()
                kh = k.transpose(0, 1).contiguous()
                vh = v.transpose(0, 1).contiguous()
                xh = _TRITON_BSA(qh, kh, vh, bm)            # (n, S, d)
                x = xh.transpose(0, 1).contiguous().unsqueeze(0)  # (1, S, n, d)
                return rearrange(x, "b s n d -> b s (n d)", n=num_heads)
            except Exception:
                torch.cuda.empty_cache()  # fall back to sparse

        if _MASKGEN_LEAN:
            # 2A-4(c): these int32 tensors depend only on shapes; building
            # them per call is a hidden H2D sync on the sparse path.
            skey = (seqlen, seqlen_kv, num_heads, q.device.index)
            ent = _SPARSE_SEQLENS_CACHE.get(skey)
            if ent is None:
                ent = (torch.tensor([0, seqlen], device=q.device, dtype=torch.int32),
                       torch.tensor([0, seqlen_kv], device=q.device, dtype=torch.int32),
                       torch.tensor([1]*num_heads, device=q.device, dtype=torch.int32))
                _SPARSE_SEQLENS_CACHE[skey] = ent
            cu_seqlens_q, cu_seqlens_k, head_mask_type = ent
        else:
            cu_seqlens_q = torch.tensor([0, seqlen], device=q.device, dtype=torch.int32)
            cu_seqlens_k = torch.tensor([0, seqlen_kv], device=q.device, dtype=torch.int32)
            head_mask_type = torch.tensor([1]*num_heads, device=q.device, dtype=torch.int32)
        streaming_info = None
        max_seqlen_q_ = seqlen
        max_seqlen_k_ = seqlen_kv
        p_dropout = 0.0
        x = block_sparse_attn_func(
            q, k, v,
            cu_seqlens_q, cu_seqlens_k,
            head_mask_type,
            streaming_info,
            base_blockmask,
            max_seqlen_q_, max_seqlen_k_,
            p_dropout,
            deterministic=False,
            softmax_scale=None,
            is_causal=False,
            exact_streaming=False,
            return_attn_probs=False,
        ).unsqueeze(0)
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif compatibility_mode:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_3_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn_interface.flash_attn_func(q, k, v)
        if isinstance(x, tuple):
            x = x[0]
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_2_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn.flash_attn_func(q, k, v)
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif SAGE_ATTN_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = sageattn(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    else:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    return x


def _modulate_impl(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return (x * (1 + scale) + shift)


modulate = _maybe_compile(_modulate_impl)


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


# ---------------------------------------------------------------------------
# Phase 2A-1b: fused RoPE apply (FLASHVSR_FUSE_ROPE, default OFF).
#
# The eager rope_apply materializes an fp64 copy of x (~104 MB @768), a
# complex128 product (~104 MB) and a bf16 down-cast per call — 60 calls per
# steady chunk ≈ 10 ms of pure memory traffic (ANALYSIS §1.2 "rope apply").
# The fused path computes the *same* fp64 complex multiply in real arithmetic
# ((a+bi)(c+di) = (ac−bd)+(ad+bc)i — exactly what the eager complex kernel
# does) inside one torch.compile-generated kernel: reads bf16 x + fp64 freqs,
# writes bf16 out, no fp64 intermediates hit DRAM. freqs.real / freqs.imag are
# strided fp64 views of the complex tensor (no copy). Numerics: identical
# operations in fp64; any FMA-contraction difference is far below the bf16
# output quantum — gated at PSNR ≥ 49 dB vs OFF (measured max|diff| reported
# in PHASE_BENCH_LOG.md).
# Compiled lazily on first use so the flag can be toggled at runtime; any
# compile/runtime failure falls back to the eager path silently.
# ---------------------------------------------------------------------------
_FUSE_ROPE = os.environ.get("FLASHVSR_FUSE_ROPE", "0") != "0"

_rope_fused_fn = None


def _rope_apply_fused_impl(x, f_real, f_imag, num_heads):
    B, S, D = x.shape
    xv = x.reshape(B, S, num_heads, -1, 2)
    xr = xv[..., 0].to(torch.float64)
    xi = xv[..., 1].to(torch.float64)
    # f_real/f_imag: (S, 1, dc) fp64 views -> broadcast over (B, S, n, dc)
    o_r = xr * f_real - xi * f_imag
    o_i = xr * f_imag + xi * f_real
    out = torch.stack((o_r, o_i), dim=-1)      # (B, S, n, dc, 2)
    return out.flatten(2).to(x.dtype)          # (B, S, n*d), interleaved pairs


def _get_rope_fused():
    global _rope_fused_fn
    if _rope_fused_fn is None:
        try:
            _rope_fused_fn = torch.compile(_rope_apply_fused_impl, dynamic=True)
        except Exception:
            _rope_fused_fn = _rope_apply_fused_impl
    return _rope_fused_fn


def rope_apply(x, freqs, num_heads):
    if _FUSE_ROPE:
        try:
            return _get_rope_fused()(x, freqs.real, freqs.imag, num_heads)
        except Exception:
            pass  # fall back to the eager reference path below
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(
        x.shape[0], x.shape[1], x.shape[2], -1, 2))
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


# ----------------------------
# Norms & Blocks
# ----------------------------
def _rmsnorm_impl(x, weight, eps):
    dtype = x.dtype
    out = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps)
    return out.to(dtype) * weight


_rmsnorm_fused = _maybe_compile(_rmsnorm_impl)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        if _FUSE_NORM:
            return _rmsnorm_fused(x, self.weight, self.eps)
        dtype = x.dtype
        return self.norm(x.float()).to(dtype) * self.weight


class AttentionModule(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        
    def forward(self, q, k, v, attention_mask=None):
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads, attention_mask=attention_mask)
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        
        self.attn = AttentionModule(self.num_heads)
        self.local_attn_mask = None

    def forward(self, x, freqs, f=None, h=None, w=None, local_num=None, topk=None,
                train_img=False, block_id=None, kv_len=None, is_full_block=False,
                is_stream=False, pre_cache_k=None, pre_cache_v=None, local_range = 9):
        B, L, D = x.shape
        if is_stream and pre_cache_k is not None and pre_cache_v is not None:
            assert f==2, "f must be 2"
        if is_stream and (pre_cache_k is None or pre_cache_v is None):
            assert f==6, " start f must be 6"
        assert L == f * h * w, "Sequence length mismatch with provided (f,h,w)."

        with nvtx_range("qkv_norm"):
            if _fp8g.enabled("qkv"):
                # 2B-2: q/k/v consume the same activation -> quantize x once
                # (per-row scales) and run three e4m3 scaled_mm GEMMs.
                pre = _fp8g.quant(x.reshape(-1, x.shape[-1]))
                q = self.norm_q(_fp8g.linear(self.q, x, pre=pre))
                k = self.norm_k(_fp8g.linear(self.k, x, pre=pre))
                v = _fp8g.linear(self.v, x, pre=pre)
            else:
                q = self.norm_q(self.q(x))
                k = self.norm_k(self.k(x))
                v = self.v(x)
        with nvtx_range("rope"):
            q = rope_apply(q, freqs, self.num_heads)
            k = rope_apply(k, freqs, self.num_heads)

        win = (2, 8, 8)
        seqlen = f//win[0]
        use_arena = _KV_RINGBUF and is_stream and B == 1
        if use_arena:
            # 2A-2 arena path: partition K/V straight into the preallocated
            # cache; k_w/v_w are the live contiguous views (pre + new, in
            # temporal order) — bit-identical to the cat path below.
            with nvtx_range("win_part"):
                q = q.view(B, f, h, w, D)
                k = k.view(B, f, h, w, D)
                v = v.view(B, f, h, w, D)
                q_w = WindowPartition3D.partition(q, win)
                one_len = (h // win[1]) * (w // win[2])
                block_tokens = win[0] * win[1] * win[2]  # tokens per window (=128)
                arena_k = pre_cache_k if isinstance(pre_cache_k, _KVArena) else \
                    _KVArena(kv_len, one_len, block_tokens, D, x.dtype, x.device)
                arena_v = pre_cache_v if isinstance(pre_cache_v, _KVArena) else \
                    _KVArena(kv_len, one_len, block_tokens, D, x.dtype, x.device)
                k_w = arena_k.append_partition(k, win)
                v_w = arena_v.append_partition(v, win)
        else:
            with nvtx_range("win_part"):
                q = q.view(B, f, h, w, D)
                k = k.view(B, f, h, w, D)
                v = v.view(B, f, h, w, D)

                q_w = WindowPartition3D.partition(q, win)
                k_w = WindowPartition3D.partition(k, win)
                v_w = WindowPartition3D.partition(v, win)

            one_len = k_w.shape[0] // B // seqlen
            if pre_cache_k is not None and pre_cache_v is not None:
                with nvtx_range("kv_cat"):
                    k_w = torch.cat([pre_cache_k, k_w], dim=0)
                    v_w = torch.cat([pre_cache_v, v_w], dim=0)

        block_n = q_w.shape[0] // B
        block_s = q_w.shape[1]
        block_n_kv = k_w.shape[0] // B

        with nvtx_range("reorder"):
            reorder_q = rearrange(q_w, '(b block_n) (block_s) d -> b (block_n block_s) d', block_n=block_n, block_s=block_s)
            reorder_k = rearrange(k_w, '(b block_n) (block_s) d -> b (block_n block_s) d', block_n=block_n_kv, block_s=block_s)
            reorder_v = rearrange(v_w, '(b block_n) (block_s) d -> b (block_n block_s) d', block_n=block_n_kv, block_s=block_s)

        window_size = win[0]*h*w//128

        with nvtx_range("mask_gen"):
            if self.local_attn_mask is None or self.local_attn_mask_h!=h//8 or self.local_attn_mask_w!=w//8 or self.local_range!=local_range:
                self.local_attn_mask = build_local_block_mask_shifted_vec_normal_slide(h//8, w//8, local_range, local_range, include_self=True, device=k_w.device)
                self.local_attn_mask_h = h//8
                self.local_attn_mask_w = w//8
                self.local_range = local_range
            attention_mask = generate_draft_block_mask(B, self.num_heads, seqlen, q_w, k_w, topk=topk, local_attn_mask=self.local_attn_mask)

        with nvtx_range("attn_core"):
            x = self.attn(reorder_q, reorder_k, reorder_v, attention_mask)

        with nvtx_range("cache_trim"):
            if use_arena:
                # same semantics as the slice-trim below: drop the oldest
                # temporal slot once more than kv_len slots are cached.
                if arena_k.length // one_len > kv_len:
                    arena_k.trim(one_len)
                    arena_v.trim(one_len)
                cache_k = arena_k
                cache_v = arena_v
            else:
                cur_block_n, cur_block_s, _ = k_w.shape
                cache_num = cur_block_n // one_len
                if cache_num > kv_len:
                    cache_k = k_w[one_len:, :, :]
                    cache_v = v_w[one_len:, :, :]
                else:
                    cache_k = k_w
                    cache_v = v_w

        with nvtx_range("win_rev"):
            x = rearrange(x, 'b (block_n block_s) d -> (b block_n) (block_s) d', block_n=block_n, block_s=block_s)
            x = WindowPartition3D.reverse(x, win, (f, h, w))
            x = x.view(B, f*h*w, D)

        with nvtx_range("o_proj"):
            # 2B-2: e4m3 o-projection (per-row activation scales).
            out = _fp8g.linear(self.o, x) if _fp8g.enabled("o") else self.o(x)
        if is_stream:
            return out, cache_k, cache_v
        return out


class CrossAttention(nn.Module):
    """
    仅考虑文本 context；提供持久 KV 缓存。
    """
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)

        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

        self.attn = AttentionModule(self.num_heads)

        # 持久缓存
        self.cache_k = None
        self.cache_v = None

    @torch.no_grad()
    def init_cache(self, ctx: torch.Tensor):
        """ctx: [B, S_ctx, dim] —— 经过 text_embedding 之后的上下文"""
        self.cache_k = self.norm_k(self.k(ctx))
        self.cache_v = self.v(ctx)

    def clear_cache(self):
        self.cache_k = None
        self.cache_v = None

    def forward(self, x: torch.Tensor, y: torch.Tensor, is_stream: bool = False):
        """
        y 即文本上下文（未做其他分支）。
        """
        q = self.norm_q(self.q(x))
        assert self.cache_k is not None and self.cache_v is not None
        k = self.cache_k
        v = self.cache_v

        x = self.attn(q, k, v)
        return self.o(x)


def _gate_impl(x, gate, residual):
    return x + gate * residual


_gate_fused = _maybe_compile(_gate_impl)


class GateModule(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x, gate, residual):
        if _FUSE_NORM:
            return _gate_fused(x, gate, residual)
        return x + gate * residual


class DiTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = SelfAttention(dim, num_heads, eps)
        self.cross_attn = CrossAttention(dim, num_heads, eps)

        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(
            approximate='tanh'), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()
        # B1 lossless cache: (modulation + t_mod).chunk(6) is step-invariant.
        self._mod_cache = None
        self._mod_cache_key = None

    def forward(self, x, context, t_mod, freqs, f, h, w, local_num=None, topk=None,
                train_img=False, block_id=None, kv_len=None, is_full_block=False,
                is_stream=False, pre_cache_k=None, pre_cache_v=None, local_range = 9):
        with nvtx_range("mod1"):
            if _CACHE_MOD:
                key = (id(t_mod), t_mod.dtype, t_mod.device)
                if self._mod_cache_key != key:
                    self._mod_cache = (
                        self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
                    ).chunk(6, dim=1)
                    self._mod_cache_key = key
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self._mod_cache
            else:
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                    self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=1)
            input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        with nvtx_range("self_attn"):
            self_attn_output, self_attn_cache_k, self_attn_cache_v = self.self_attn(
                input_x, freqs, f, h, w, local_num, topk, train_img, block_id,
                kv_len=kv_len, is_full_block=is_full_block, is_stream=is_stream,
                pre_cache_k=pre_cache_k, pre_cache_v=pre_cache_v, local_range = local_range)

        with nvtx_range("gate1"):
            x = self.gate(x, gate_msa, self_attn_output)
        with nvtx_range("xattn"):
            x = x + self.cross_attn(self.norm3(x), context, is_stream=is_stream)
        with nvtx_range("ffn"):
            input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
            if _fp8g.enabled("ffn"):
                # 2B-2: e4m3 ffn1+ffn2; GELU is fused into ffn2's input
                # quantization kernel (fp32 gelu -> e4m3).
                x = self.gate(x, gate_mlp, _fp8g.ffn(self.ffn, input_x))
            elif _fp8g.enabled("ffn1"):
                # 2B-2 bisection scope: e4m3 ffn1 only (ffn2 stays bf16).
                x = self.gate(x, gate_mlp, _fp8g.ffn_partial(self.ffn, input_x))
            else:
                x = self.gate(x, gate_mlp, self.ffn(input_x))
        if is_stream:
            return x, self_attn_cache_k, self_attn_cache_v
        return x


class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, has_pos_emb=False):
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        self.has_pos_emb = has_pos_emb
        if has_pos_emb:
            self.emb_pos = torch.nn.Parameter(torch.zeros((1, 514, 1280)))

    def forward(self, x):
        if self.has_pos_emb:
            x = x + self.emb_pos.to(dtype=x.dtype, device=x.device)
        return self.proj(x)


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)
        # B1 lossless cache: (modulation + t_mod).chunk(2) is step-invariant.
        self._mod_cache = None
        self._mod_cache_key = None

    def forward(self, x, t_mod):
        if _CACHE_MOD:
            key = (id(t_mod), t_mod.dtype, t_mod.device)
            if self._mod_cache_key != key:
                self._mod_cache = (
                    self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
                ).chunk(2, dim=1)
                self._mod_cache_key = key
            shift, scale = self._mod_cache
        else:
            shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(2, dim=1)
        x = (self.head(self.norm(x) * (1 + scale) + shift))
        return x


# ----------------------------
# WanModel (no image branch) — init 时即产生 KV 缓存
# ----------------------------
class WanModel(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        # init_context: torch.Tensor,     # <<<< 必填：在 __init__ 里用它生成 cross-attn KV 缓存
        has_image_input: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.freq_dim = freq_dim
        self.patch_size = patch_size

        # patch embed
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)

        # text / time embed
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        self.blocks = nn.ModuleList([
            DiTBlock(dim, num_heads, ffn_dim, eps)
            for _ in range(num_layers)
        ])
        self.head = Head(dim, out_dim, patch_size, eps)

        head_dim = dim // num_heads
        self.freqs = precompute_freqs_cis_3d(head_dim)

        self._cross_kv_initialized = False

    # 可选：手动清空 / 重新初始化
    def clear_cross_kv(self):
        for blk in self.blocks:
            blk.cross_attn.clear_cache()
        self._cross_kv_initialized = False

    @torch.no_grad()
    def reinit_cross_kv(self, new_context: torch.Tensor):
        ctx_txt = self.text_embedding(new_context)
        for blk in self.blocks:
            blk.cross_attn.init_cache(ctx_txt)
        self._cross_kv_initialized = True

    def patchify(self, x: torch.Tensor):
        x = self.patch_embedding(x)
        grid_size = x.shape[2:]
        x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
        return x, grid_size  # x, grid_size: (f, h, w)

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0], h=grid_size[1], w=grid_size[2], 
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2]
        )

    def forward(self,
                x: torch.Tensor,
                timestep: torch.Tensor,
                context: torch.Tensor,
                use_gradient_checkpointing: bool = False,
                use_gradient_checkpointing_offload: bool = False,
                LQ_latents: Optional[List[torch.Tensor]] = None,
                train_img: bool = False,
                topk_ratio: Optional[float] = None,
                kv_ratio: Optional[float] = None,
                local_num: Optional[int] = None,
                is_full_block: bool = False,
                causal_idx: Optional[int] = None,
                **kwargs,
                ):
        # time / text embeds
        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))

        # 这里仍会嵌入 text（CrossAttention 若已有缓存会忽略它）
        # context = self.text_embedding(context)

        # 输入打补丁
        x, (f, h, w) = self.patchify(x)
        B = x.shape[0]

        # window / masks 超参
        win = (2, 8, 8)
        seqlen = f//win[0]
        if local_num is None:
            local_random = random.random()
            if local_random < 0.3:
                local_num = seqlen - 3
            elif local_random < 0.4:
                local_num = seqlen - 4
            elif local_random < 0.5:
                local_num = seqlen - 2
            else:
                local_num = seqlen

        window_size = win[0]*h*w//128
        square_num = window_size*window_size
        topk_ratio = 2.0
        topk = min(max(int(square_num*topk_ratio), 1), int(square_num*seqlen)-1)

        if kv_ratio is None:
            kv_ratio = (random.uniform(0., 1.0)**2)*(local_num-2-2)+2
        kv_len = min(max(int(window_size*kv_ratio), 1), int(window_size*seqlen)-1)

        decay_ratio = random.uniform(0.7, 1.0)

        # RoPE 3D
        freqs = torch.cat([
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward

        # blocks
        for block_id, block in enumerate(self.blocks):
            if LQ_latents is not None and block_id < len(LQ_latents):
                x += LQ_latents[block_id]

            if self.training and use_gradient_checkpointing:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x, context, t_mod, freqs, f, h, w, local_num, topk,
                            train_img, block_id, kv_len, is_full_block, False,
                            None, None,
                            use_reentrant=False,
                        )
                else:
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, context, t_mod, freqs, f, h, w, local_num, topk,
                        train_img, block_id, kv_len, is_full_block, False,
                        None, None, 
                        use_reentrant=False,
                    )
            else:
                x = block(x, context, t_mod, freqs, f, h, w, local_num, topk,
                          train_img, block_id, kv_len, is_full_block, False,
                          None, None)

        x = self.head(x, t)
        x = self.unpatchify(x, (f, h, w))
        return x

    @staticmethod
    def state_dict_converter():
        return WanModelStateDictConverter()
    

# ----------------------------
# State dict converter（保持原映射；已忽略 has_image_input 使用）
# ----------------------------
class WanModelStateDictConverter:
    def __init__(self):
        pass

    def from_diffusers(self, state_dict):
        rename_dict = {
            "blocks.0.attn1.norm_k.weight": "blocks.0.self_attn.norm_k.weight",
            "blocks.0.attn1.norm_q.weight": "blocks.0.self_attn.norm_q.weight",
            "blocks.0.attn1.to_k.bias": "blocks.0.self_attn.k.bias",
            "blocks.0.attn1.to_k.weight": "blocks.0.self_attn.k.weight",
            "blocks.0.attn1.to_out.0.bias": "blocks.0.self_attn.o.bias",
            "blocks.0.attn1.to_out.0.weight": "blocks.0.self_attn.o.weight",
            "blocks.0.attn1.to_q.bias": "blocks.0.self_attn.q.bias",
            "blocks.0.attn1.to_q.weight": "blocks.0.self_attn.q.weight",
            "blocks.0.attn1.to_v.bias": "blocks.0.self_attn.v.bias",
            "blocks.0.attn1.to_v.weight": "blocks.0.self_attn.v.weight",
            "blocks.0.attn2.norm_k.weight": "blocks.0.cross_attn.norm_k.weight",
            "blocks.0.attn2.norm_q.weight": "blocks.0.cross_attn.norm_q.weight",
            "blocks.0.attn2.to_k.bias": "blocks.0.cross_attn.k.bias",
            "blocks.0.attn2.to_k.weight": "blocks.0.cross_attn.k.weight",
            "blocks.0.attn2.to_out.0.bias": "blocks.0.cross_attn.o.bias",
            "blocks.0.attn2.to_out.0.weight": "blocks.0.cross_attn.o.weight",
            "blocks.0.attn2.to_q.bias": "blocks.0.cross_attn.q.bias",
            "blocks.0.attn2.to_q.weight": "blocks.0.cross_attn.q.weight",
            "blocks.0.attn2.to_v.bias": "blocks.0.cross_attn.v.bias",
            "blocks.0.attn2.to_v.weight": "blocks.0.cross_attn.v.weight",
            "blocks.0.ffn.net.0.proj.bias": "blocks.0.ffn.0.bias",
            "blocks.0.ffn.net.0.proj.weight": "blocks.0.ffn.0.weight",
            "blocks.0.ffn.net.2.bias": "blocks.0.ffn.2.bias",
            "blocks.0.ffn.net.2.weight": "blocks.0.ffn.2.weight",
            "blocks.0.norm2.bias": "blocks.0.norm3.bias",
            "blocks.0.norm2.weight": "blocks.0.norm3.weight",
            "blocks.0.scale_shift_table": "blocks.0.modulation",
            "condition_embedder.text_embedder.linear_1.bias": "text_embedding.0.bias",
            "condition_embedder.text_embedder.linear_1.weight": "text_embedding.0.weight",
            "condition_embedder.text_embedder.linear_2.bias": "text_embedding.2.bias",
            "condition_embedder.text_embedder.linear_2.weight": "text_embedding.2.weight",
            "condition_embedder.time_embedder.linear_1.bias": "time_embedding.0.bias",
            "condition_embedder.time_embedder.linear_1.weight": "time_embedding.0.weight",
            "condition_embedder.time_embedder.linear_2.bias": "time_embedding.2.bias",
            "condition_embedder.time_embedder.linear_2.weight": "time_embedding.2.weight",
            "condition_embedder.time_proj.bias": "time_projection.1.bias",
            "condition_embedder.time_proj.weight": "time_projection.1.weight",
            "patch_embedding.bias": "patch_embedding.bias",
            "patch_embedding.weight": "patch_embedding.weight",
            "scale_shift_table": "head.modulation",
            "proj_out.bias": "head.head.bias",
            "proj_out.weight": "head.head.weight",
        }
        state_dict_ = {}
        for name, param in state_dict.items():
            if name in rename_dict:
                state_dict_[rename_dict[name]] = param
            else:
                name_ = ".".join(name.split(".")[:1] + ["0"] + name.split(".")[2:])
                if name_ in rename_dict:
                    name_ = rename_dict[name_]
                    name_ = ".".join(name_.split(".")[:1] + [name.split(".")[1]] + name_.split(".")[2:])
                    state_dict_[name_] = param
        if hash_state_dict_keys(state_dict) == "cb104773c6c2cb6df4f9529ad5c60d0b":
            config = {
                "model_type": "t2v",
                "patch_size": (1, 2, 2),
                "text_len": 512,
                "in_dim": 16,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "window_size": (-1, -1),
                "qk_norm": True,
                "cross_attn_norm": True,
                "eps": 1e-6,
            }
        else:
            config = {}
        return state_dict_, config
    
    def from_civitai(self, state_dict):
        state_dict = {name: param for name, param in state_dict.items() if not name.startswith("vace")}
        # 保留原有哈希匹配返回的 config；实现本身不使用 has_image_input 分支
        if hash_state_dict_keys(state_dict) == "9269f8db9040a9d860eaca435be61814":
            config = {"has_image_input": False,"patch_size": [1, 2, 2],"in_dim": 16,"dim": 1536,"ffn_dim": 8960,"freq_dim": 256,"text_dim": 4096,"out_dim": 16,"num_heads": 12,"num_layers": 30,"eps": 1e-6}
        elif hash_state_dict_keys(state_dict) == "aafcfd9672c3a2456dc46e1cb6e52c70":
            config = {"has_image_input": False,"patch_size": [1, 2, 2],"in_dim": 16,"dim": 5120,"ffn_dim": 13824,"freq_dim": 256,"text_dim": 4096,"out_dim": 16,"num_heads": 40,"num_layers": 40,"eps": 1e-6}
        elif hash_state_dict_keys(state_dict) == "6bfcfb3b342cb286ce886889d519a77e":
            config = {"has_image_input": False,"patch_size": [1, 2, 2],"in_dim": 36,"dim": 5120,"ffn_dim": 13824,"freq_dim": 256,"text_dim": 4096,"out_dim": 16,"num_heads": 40,"num_layers": 40,"eps": 1e-6}
        elif hash_state_dict_keys(state_dict) == "6d6ccde6845b95ad9114ab993d917893":
            config = {"has_image_input": False,"patch_size": [1, 2, 2],"in_dim": 36,"dim": 1536,"ffn_dim": 8960,"freq_dim": 256,"text_dim": 4096,"out_dim": 16,"num_heads": 12,"num_layers": 30,"eps": 1e-6}
        elif hash_state_dict_keys(state_dict) == "349723183fc063b2bfc10bb2835cf677":
            config = {"has_image_input": False,"patch_size": [1, 2, 2],"in_dim": 48,"dim": 1536,"ffn_dim": 8960,"freq_dim": 256,"text_dim": 4096,"out_dim": 16,"num_heads": 12,"num_layers": 30,"eps": 1e-6}
        elif hash_state_dict_keys(state_dict) == "efa44cddf936c70abd0ea28b6cbe946c":
            config = {"has_image_input": False,"patch_size": [1, 2, 2],"in_dim": 48,"dim": 5120,"ffn_dim": 13824,"freq_dim": 256,"text_dim": 4096,"out_dim": 16,"num_heads": 40,"num_layers": 40,"eps": 1e-6}
        elif hash_state_dict_keys(state_dict) == "3ef3b1f8e1dab83d5b71fd7b617f859f":
            config = {"has_image_input": False,"patch_size": [1, 2, 2],"in_dim": 36,"dim": 5120,"ffn_dim": 13824,"freq_dim": 256,"text_dim": 4096,"out_dim": 16,"num_heads": 40,"num_layers": 40,"eps": 1e-6,"has_image_pos_emb": False}
        else:
            config = {}
        return state_dict, config
