"""Hopper (sm_90) warp-specialized block-sparse FlashAttention v2 (Gluon).

Phase 3 of the GH200 acceleration campaign. The v1 Triton kernel
(`triton_block_sparse_attn.py::_bsfa_tma_kernel_snd`) is a single-CTA-per-SM
software-pipelined loop: all 8 warps synchronize at every stage barrier, the
scheduler has no eligible warp 68.6% of cycles, and the tensor pipe idles
~60% (ncu: 2.0 ms/call, 40% SOL at the reference shape q 8448 x kv 33792,
h12 d128, density ~0.42-0.45).

v2 restructures the SAME math into an FA3-style producer/consumer pipeline
using Triton's Gluon dialect:

  - producer warp group (4 warps, low registers): scalar-loads the selected
    kv-block index list (exact CSR of the boolean block mask, unchanged from
    v1, next index software-prefetched) and streams K/V tiles into an
    NBUF-deep shared-memory ring via TMA, guarded by mbarriers;
  - two consumer warpgroups (4+4 warps) owning 64-row halves of the q block
    ("pingpong": while one warpgroup occupies the WGMMA pipe the other runs
    its softmax), each additionally deferring its P.V wgmma by one iteration
    (QK(j) is issued BEFORE PV(j-1); wgmma groups retire in order, so
    wait(pendings=1) completes QK(j) while PV(j-1) drains under the softmax).

Mask semantics are bit-identical to v1: the (H, Nqb, Nkvb) boolean block mask
is consumed through the exact same `_make_csr` (stable argsort -> ascending
kv-block order, degenerate all-false rows produce zero output rows through
the l==0 -> l_safe=1 guard). Numerics: the softmax scale is folded into the
exponent (exp2((qk - m) * scale*log2e)) like the reference FA2-style
`block_sparse_attn` kernel, instead of v1's bf16 q-prescale; both are gated
at cosine >= 0.9999 against `block_sparse_attn`.

Forward-only, bf16, D=128, no dropout. Opt-in via
FLASHVSR_ATTN_BACKEND=triton2, sm_90-guarded; ANY failure at import/compile/
launch raises to the caller, which falls back to the v1 triton path (then to
`block_sparse_attn`), preserving the established fallback ladder.
"""
import math
import os

import torch

_V2_OK = False
_V2_ERR = None
try:
    import triton  # noqa: F401
    from triton.experimental import gluon
    import triton.experimental.gluon.language as ttgl
    from triton.experimental.gluon.language.nvidia import hopper as _hop
    from triton.experimental.gluon.language.nvidia.hopper import (
        tma as _tma, mbarrier as _mb)
    from triton.experimental.gluon.nvidia.hopper import (
        TensorDescriptor as _GluonTensorDescriptor)
    _V2_OK = True
except Exception as e:  # pragma: no cover
    _V2_ERR = e

from .triton_block_sparse_attn import _make_csr

# ---------------------------------------------------------------------------
# Phase 3.5-2: fused CSR build (FLASHVSR_FUSED_CSR, default OFF).
#
# `_make_csr` uses torch.argsort(descending, stable) — a full radix sort
# (radixSortKVInPlace ~0.71 ms/chunk in-pipe) — only to move the True
# kv-blocks to the front in ascending order. The v2 kernel reads only
# idx[0:cnt], so a single-pass cumsum-scatter reproduces idx[0:cnt] and cnt
# bit-identically without a sort. Lossless (idx[0:cnt] + cnt equal to the
# argsort path; the unused tail idx[cnt:] is never read).
# ---------------------------------------------------------------------------
_FUSED_CSR = os.environ.get("FLASHVSR_FUSED_CSR", "0") != "0"

if _V2_OK:
    import triton
    import triton.language as tl

    @triton.jit
    def _csr_scan_kernel(BM, IDX, CNT, R, NKVB,
                         sbr, sbn, sir, sin_, scr,
                         BLOCK: tl.constexpr):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        valid = offs < NKVB
        m = tl.load(BM + row * sbr + offs * sbn, mask=valid, other=0).to(tl.int32)
        # inclusive prefix sum -> write slot of each True block; ascending offs
        # -> ascending slots -> idx[0:cnt] holds True kv-blocks in ascending
        # order (== argsort(descending, stable)[0:cnt]).
        pos = tl.cumsum(m, axis=0) - 1
        cnt = tl.sum(m, axis=0)
        keep = (m > 0) & valid
        tl.store(IDX + row * sir + pos * sin_, offs.to(tl.int32), mask=keep)
        tl.store(CNT + row * scr, cnt)


def _make_csr_fused(bm):
    """Sort-free CSR: (H,Nqb,Nkvb) bool -> (idx int32, cnt int32).

    idx[0:cnt] and cnt are bit-identical to `_make_csr`; the tail idx[cnt:]
    (never consumed by the kernel) is left uninitialized.
    """
    H, Nqb, Nkvb = bm.shape
    R = H * Nqb
    bmf = bm.reshape(R, Nkvb)
    idx = torch.empty(R, Nkvb, dtype=torch.int32, device=bm.device)
    cnt = torch.empty(R, dtype=torch.int32, device=bm.device)
    BLOCK = max(16, triton.next_power_of_2(Nkvb))
    bmi = bmf.to(torch.int8)
    _csr_scan_kernel[(R,)](
        bmi, idx, cnt, R, Nkvb,
        bmi.stride(0), bmi.stride(1), idx.stride(0), idx.stride(1), cnt.stride(0),
        BLOCK=BLOCK)
    return idx.view(H, Nqb, Nkvb), cnt.view(H, Nqb)


# Ring depth for the K/V shared-memory pipeline. NBUF=3 fits in 227 KB with
# BLOCK_M=BLOCK_N=D=128 (Q 32K + 3x(K 32K + V 32K)); measured fastest
# (sweep in PHASE_BENCH_LOG Phase 3).
_NBUF = max(2, min(4, int(os.environ.get("FLASHVSR_ATTN_V2_NBUF", "3"))))
# Producer warp-group register budget (multiple of 8 in [24, 256]).
_PROD_REGS = int(os.environ.get("FLASHVSR_ATTN_V2_PROD_REGS", "40"))
# KV tile width. Fixed at 128 (= the mask block granularity). A 64-wide
# exact-refinement variant (2-CTA/SM residency) was evaluated during Phase 3
# and rejected: it needs two distinct wgmma layouts (qk N=64, pv N=128) with
# per-iteration layout conversions, for an expected perf LOSS vs the
# measured-passing 12-warp warp-specialized config.
_BLOCK_N = 128


if _V2_OK:

    @gluon.jit
    def _producer(q_desc, k_desc, v_desc, KVIdx, cnt, row_q, col0,
                  q_smem, k_bufs, v_bufs, q_full, k_full, k_empty,
                  v_full, v_empty, v_f0, nhw, nw,
                  BLOCK_N: ttgl.constexpr, HEAD_DIM: ttgl.constexpr,
                  NBUF: ttgl.constexpr, RASTER_V: ttgl.constexpr):
        # Q tile once (two 64-column TMA boxes -> one barrier).
        QBYTES: ttgl.constexpr = q_smem.type.shape[0] * HEAD_DIM * 2
        _mb.expect(q_full, QBYTES)
        _tma.async_copy_global_to_shared(
            q_desc, [row_q, col0], q_full, q_smem.slice(0, 64, dim=1))
        _tma.async_copy_global_to_shared(
            q_desc, [row_q, col0 + 64], q_full, q_smem.slice(64, 64, dim=1))
        KVBYTES: ttgl.constexpr = BLOCK_N * HEAD_DIM * 2
        # software-prefetch the selected-block index list: issue the load for
        # j+1 before iteration j's barrier waits so the dependent-load latency
        # (the dominant long-scoreboard stall) hides behind the ring handshake
        kvb = ttgl.load(KVIdx, mask=cnt > 0, other=0)
        for j in range(cnt):
            buf = j % NBUF
            phase = (j // NBUF) & 1
            kvb_next = ttgl.load(KVIdx + j + 1, mask=j + 1 < cnt, other=0)
            row = kvb * BLOCK_N
            _mb.wait(k_empty.index(buf), phase)
            _mb.expect(k_full.index(buf), KVBYTES)
            kb = k_bufs.index(buf)
            _tma.async_copy_global_to_shared(
                k_desc, [row, col0], k_full.index(buf), kb.slice(0, 64, dim=1))
            _tma.async_copy_global_to_shared(
                k_desc, [row, col0 + 64], k_full.index(buf),
                kb.slice(64, 64, dim=1))
            _mb.wait(v_empty.index(buf), phase)
            _mb.expect(v_full.index(buf), KVBYTES)
            vb = v_bufs.index(buf)
            if RASTER_V:
                # 5A-v1 zero-copy: gather the (2,8,8) window straight from the
                # raster (F,H,W,Hh*D) V arena; box order == partition order
                # (bit-exact, probe-verified).
                slot = kvb // nhw
                r = kvb % nhw
                f0 = v_f0 + slot * 2
                h0 = (r // nw) * 8
                w0 = (r % nw) * 8
                _tma.async_copy_global_to_shared(
                    v_desc, [f0, h0, w0, col0], v_full.index(buf),
                    vb.slice(0, 64, dim=1))
                _tma.async_copy_global_to_shared(
                    v_desc, [f0, h0, w0, col0 + 64], v_full.index(buf),
                    vb.slice(64, 64, dim=1))
            else:
                _tma.async_copy_global_to_shared(
                    v_desc, [row, col0], v_full.index(buf),
                    vb.slice(0, 64, dim=1))
                _tma.async_copy_global_to_shared(
                    v_desc, [row, col0 + 64], v_full.index(buf),
                    vb.slice(64, 64, dim=1))
            kvb = kvb_next

    @gluon.jit
    def _consumer(O, cnt, scale_log2e, row_q, col_o, som, sok,
                  q_smem, k_bufs, v_bufs, q_full, k_full, k_empty,
                  v_full, v_empty, nhw, nw, hrast, wrast,
                  row_off: ttgl.constexpr,
                  HALF_M: ttgl.constexpr, BLOCK_N: ttgl.constexpr,
                  HEAD_DIM: ttgl.constexpr, NBUF: ttgl.constexpr,
                  RASTER_OUT: ttgl.constexpr):
        # One consumer WARPGROUP owning a HALF_M-row horizontal stripe of the
        # q block (FA3 "pingpong": two such warpgroups run the same loop on
        # different stripes; while one occupies the WGMMA pipe the other runs
        # its softmax, hiding the f32/MUFU chain).
        mma: ttgl.constexpr = ttgl.NVMMADistributedLayout(
            version=[3, 0], warps_per_cta=[4, 1],
            instr_shape=[16, HEAD_DIM, 16])
        dot_a: ttgl.constexpr = ttgl.DotOperandLayout(
            operand_index=0, parent=mma, k_width=2)
        q_half = q_smem.slice(row_off, HALF_M, dim=0)
        # Track the running max in the ALREADY-SCALED (log2e) domain: ms = m*s.
        # Since s>0, maximum(m_i, rowmax)*s == maximum(ms_i, rowmax*s), so this
        # is algebraically identical but lets the exponent be a single FFMA
        # (qk*s - ms) instead of a broadcast-sub-then-mul ((qk-m)*s, which the
        # compiler cannot contract). 3.5-1a; gated cos>=0.9999 + PSNR>=49.
        ms_i = ttgl.full([HALF_M], -float("inf"), ttgl.float32,
                         ttgl.SliceLayout(1, mma))
        l_i = ttgl.zeros([HALF_M], ttgl.float32, ttgl.SliceLayout(1, mma))
        acc = ttgl.zeros([HALF_M, HEAD_DIM], ttgl.float32, mma)
        qk0 = ttgl.zeros([HALF_M, BLOCK_N], ttgl.float32, mma)
        # p tile of the previous iteration whose P.V wgmma is deferred so it
        # overlaps this iteration's softmax (wgmma groups retire in order, so
        # issuing QK(j) BEFORE PV(j-1) lets wait(pendings=1) complete QK(j)
        # while PV(j-1) is still in flight).
        p_prev = ttgl.zeros([HALF_M, BLOCK_N], ttgl.bfloat16, dot_a)
        _mb.wait(q_full, 0)
        if cnt > 0:
            # peeled iteration 0: QK only; its P.V is deferred into the loop
            _mb.wait(k_full.index(0), 0)
            qk_tok = _hop.warpgroup_mma(
                q_half, k_bufs.index(0).permute((1, 0)), qk0,
                use_acc=False, is_async=True)
            qk = _hop.warpgroup_mma_wait(0, deps=[qk_tok])
            _mb.arrive(k_empty.index(0), count=1)
            ms_ij = ttgl.maximum(ms_i, ttgl.max(qk, 1) * scale_log2e)
            p = ttgl.exp2(qk * scale_log2e - ttgl.expand_dims(ms_ij, 1))
            l_i = ttgl.sum(p, 1)
            p_prev = ttgl.convert_layout(p.to(ttgl.bfloat16), dot_a)
            ms_i = ms_ij
        for j in range(1, cnt):
            buf = j % NBUF
            phase = (j // NBUF) & 1
            pbuf = (j - 1) % NBUF
            pphase = ((j - 1) // NBUF) & 1
            _mb.wait(k_full.index(buf), phase)
            qk_tok = _hop.warpgroup_mma(
                q_half, k_bufs.index(buf).permute((1, 0)), qk0,
                use_acc=False, is_async=True)
            _mb.wait(v_full.index(pbuf), pphase)
            acc = _hop.warpgroup_mma(p_prev, v_bufs.index(pbuf), acc,
                                     is_async=True)
            qk, acc = _hop.warpgroup_mma_wait(1, deps=[qk_tok, acc])
            _mb.arrive(k_empty.index(buf), count=1)
            ms_ij = ttgl.maximum(ms_i, ttgl.max(qk, 1) * scale_log2e)
            p = ttgl.exp2(qk * scale_log2e - ttgl.expand_dims(ms_ij, 1))
            alpha = ttgl.exp2(ms_i - ms_ij)
            l_i = l_i * alpha + ttgl.sum(p, 1)
            acc = _hop.warpgroup_mma_wait(0, deps=[acc])
            _mb.arrive(v_empty.index(pbuf), count=1)
            acc = acc * ttgl.expand_dims(alpha, 1)
            p_prev = ttgl.convert_layout(p.to(ttgl.bfloat16), dot_a)
            ms_i = ms_ij
        if cnt > 0:
            lbuf = (cnt - 1) % NBUF
            lphase = ((cnt - 1) // NBUF) & 1
            _mb.wait(v_full.index(lbuf), lphase)
            acc = _hop.warpgroup_mma(p_prev, v_bufs.index(lbuf), acc,
                                     is_async=True)
            acc = _hop.warpgroup_mma_wait(0, deps=[acc])
            _mb.arrive(v_empty.index(lbuf), count=1)
        l_safe = ttgl.where(l_i == 0.0, 1.0, l_i)
        acc = acc / ttgl.expand_dims(l_safe, 1)
        if RASTER_OUT:
            # 5A-v1: store rows straight to RASTER token order (kills the
            # win_rev reverse-partition pass). q block qb covers window
            # (slot, hq, wq); row r in [0,128) covers (fl, hl, wl).
            qb = row_q // 128
            slot = qb // nhw
            rr = qb % nhw
            hq = (rr // nw) * 8
            wq = (rr % nw) * 8
            r = row_off + ttgl.arange(0, HALF_M, ttgl.SliceLayout(1, mma))
            fl = r // 64
            hl = (r % 64) // 8
            wl = r % 8
            tok = ((slot * 2 + fl) * hrast + hq + hl) * wrast + wq + wl
            offs_k = ttgl.arange(0, HEAD_DIM, ttgl.SliceLayout(0, mma))
            ptrs = O + col_o + (ttgl.expand_dims(tok, 1) * som
                                + ttgl.expand_dims(offs_k, 0) * sok)
            ttgl.store(ptrs, acc.to(O.dtype.element_ty))
        else:
            offs_m = row_q + row_off + ttgl.arange(
                0, HALF_M, ttgl.SliceLayout(1, mma))
            offs_k = ttgl.arange(0, HEAD_DIM, ttgl.SliceLayout(0, mma))
            ptrs = O + col_o + (ttgl.expand_dims(offs_m, 1) * som
                                + ttgl.expand_dims(offs_k, 0) * sok)
            ttgl.store(ptrs, acc.to(O.dtype.element_ty))

    @gluon.jit
    def _bsfa_v2_kernel(q_desc, k_desc, v_desc, O, KVIdx, KVCnt,
                        scale_log2e, soh, som, sok, sih, sim, sic, sch, scm,
                        v_f0, nhw, nw, hrast, wrast,
                        BLOCK_M: ttgl.constexpr, BLOCK_N: ttgl.constexpr,
                        HEAD_DIM: ttgl.constexpr, NBUF: ttgl.constexpr,
                        PROD_REGS: ttgl.constexpr,
                        RASTER_V: ttgl.constexpr, RASTER_OUT: ttgl.constexpr):
        start_m = ttgl.program_id(0)
        off_h = ttgl.program_id(1)
        cnt = ttgl.load(KVCnt + off_h * sch + start_m * scm)
        idx_base = KVIdx + off_h * sih + start_m * sim
        row_q = start_m * BLOCK_M
        col0 = off_h * HEAD_DIM

        smem: ttgl.constexpr = ttgl.NVMMASharedLayout(
            swizzle_byte_width=128, element_bitwidth=16, rank=2)
        q_smem = ttgl.allocate_shared_memory(
            ttgl.bfloat16, [BLOCK_M, HEAD_DIM], smem)
        k_bufs = ttgl.allocate_shared_memory(
            ttgl.bfloat16, [NBUF, BLOCK_N, HEAD_DIM], smem)
        v_bufs = ttgl.allocate_shared_memory(
            ttgl.bfloat16, [NBUF, BLOCK_N, HEAD_DIM], smem)
        bl: ttgl.constexpr = _mb.MBarrierLayout()
        q_full = ttgl.allocate_shared_memory(ttgl.int64, [1], bl)
        k_full = ttgl.allocate_shared_memory(ttgl.int64, [NBUF, 1], bl)
        k_empty = ttgl.allocate_shared_memory(ttgl.int64, [NBUF, 1], bl)
        v_full = ttgl.allocate_shared_memory(ttgl.int64, [NBUF, 1], bl)
        v_empty = ttgl.allocate_shared_memory(ttgl.int64, [NBUF, 1], bl)
        _mb.init(q_full, count=1)
        for i in ttgl.static_range(NBUF):
            _mb.init(k_full.index(i), count=1)
            # ring slots are released by BOTH consumer warpgroups
            _mb.init(k_empty.index(i), count=2)
            _mb.init(v_full.index(i), count=1)
            _mb.init(v_empty.index(i), count=2)
            # mark all ring slots initially empty (completes phase 0)
            _mb.arrive(k_empty.index(i), count=2)
            _mb.arrive(v_empty.index(i), count=2)

        col_o = off_h * soh
        HALF_M: ttgl.constexpr = BLOCK_M // 2
        ttgl.warp_specialize(
            [(_consumer, (O, cnt, scale_log2e, row_q, col_o, som, sok,
                          q_smem, k_bufs, v_bufs, q_full, k_full, k_empty,
                          v_full, v_empty, nhw, nw, hrast, wrast,
                          0, HALF_M, BLOCK_N, HEAD_DIM,
                          NBUF, RASTER_OUT)),
             (_consumer, (O, cnt, scale_log2e, row_q, col_o, som, sok,
                          q_smem, k_bufs, v_bufs, q_full, k_full, k_empty,
                          v_full, v_empty, nhw, nw, hrast, wrast,
                          HALF_M, HALF_M, BLOCK_N, HEAD_DIM,
                          NBUF, RASTER_OUT)),
             (_producer, (q_desc, k_desc, v_desc, idx_base, cnt, row_q, col0,
                          q_smem, k_bufs, v_bufs, q_full, k_full, k_empty,
                          v_full, v_empty, v_f0, nhw, nw,
                          BLOCK_N, HEAD_DIM, NBUF, RASTER_V))],
            [4, 4], [232, PROD_REGS],
        )


def _build_call(q, k, v, bm, sm_scale, BLOCK_M=128, BLOCK_N=None):
    """Shared setup for the pipeline wrapper and the kernel-only bench hook."""
    if BLOCK_N is None:
        BLOCK_N = _BLOCK_N
    assert _V2_OK, f"gluon unavailable: {_V2_ERR}"
    Nq, H, D = q.shape
    Nkv = k.shape[0]
    assert D == 128, "v2 kernel requires head_dim == 128"
    assert Nq % BLOCK_M == 0 and Nkv % BLOCK_N == 0, \
        "v2 kernel requires 128-aligned sequence lengths"
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    Nqb = Nq // BLOCK_M
    Nkvb = Nkv // BLOCK_N
    bm = bm[..., :Nqb, :Nkv // 128]
    assert bm.shape == (H, Nqb, Nkv // 128)
    if BLOCK_N != 128:
        # exact refinement: every 128-wide mask block is covered by
        # 128/BLOCK_N consecutive tiles inheriting the same mask bit
        bm = bm.repeat_interleave(128 // BLOCK_N, dim=2)
    idx, cnt = _make_csr_fused(bm) if _FUSED_CSR else _make_csr(bm)
    assert idx.stride(2) == 1  # producer walks the index list contiguously
    o = torch.empty_like(q)
    scale_log2e = float(sm_scale) * 1.4426950408889634
    q2 = q.view(Nq, H * D)
    k2 = k.view(Nkv, H * D)
    v2 = v.view(Nkv, H * D)
    layout = ttgl.NVMMASharedLayout(swizzle_byte_width=128,
                                    element_bitwidth=16, rank=2)
    q_desc = _GluonTensorDescriptor.from_tensor(q2, [BLOCK_M, 64], layout)
    k_desc = _GluonTensorDescriptor.from_tensor(k2, [BLOCK_N, 64], layout)
    v_desc = _GluonTensorDescriptor.from_tensor(v2, [BLOCK_N, 64], layout)
    grid = (Nqb, H)

    def call():
        _bsfa_v2_kernel[grid](
            q_desc, k_desc, v_desc, o, idx, cnt, scale_log2e,
            o.stride(1), o.stride(0), o.stride(2),
            idx.stride(0), idx.stride(1), idx.stride(2),
            cnt.stride(0), cnt.stride(1),
            0, 1, 1, 1, 1,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=D, NBUF=_NBUF,
            PROD_REGS=_PROD_REGS, RASTER_V=False, RASTER_OUT=False,
            num_warps=4)
        return o
    return call


def triton_block_sparse_attention_v2(q, k, v, block_mask, sm_scale=None):
    """Warp-specialized block-sparse attention (Phase 3).

    q: (Nq, H, D), k/v: (Nkv, H, D) token-major contiguous (same interface as
    the v1 strided-IO path). block_mask: (H, Nqb, Nkvb) bool, True = compute.
    Returns (Nq, H, D). Raises on any unsupported input; caller falls back.
    """
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[-1])
    return _build_call(q, k, v, block_mask, sm_scale)()


def bsfa_v2_kernel_only(q, k, v, bm, sm_scale=None):
    """Bench hook: returns a zero-setup-cost closure launching only the kernel."""
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[-1])
    return _build_call(q, k, v, bm, sm_scale)


def triton_block_sparse_attention_v2_zc(q, k, v_buf, v_f0, hrast, wrast,
                                        block_mask, sm_scale=None):
    """Phase 5A-v1 zero-copy variant: V is gathered straight from the RASTER
    (F_cap, hrast, wrast, Hh*D) arena buffer (window (2,8,8) per kv-block,
    starting at frame `v_f0`) and the output is stored straight in RASTER
    token order (the caller skips WindowPartition3D.reverse).

    q: (Nq, H, D) window-token-major (as today), k: (Nkv, H, D) window-major.
    Returns o: (Nq, H, D) in RASTER token order. Raises on any unsupported
    input; caller falls back to the partitioned path.
    """
    assert _V2_OK, f"gluon unavailable: {_V2_ERR}"
    BLOCK_M = 128
    BLOCK_N = 128
    Nq, H, D = q.shape
    Nkv = k.shape[0]
    assert D == 128 and Nq % 128 == 0 and Nkv % 128 == 0
    assert q.is_contiguous() and k.is_contiguous() and v_buf.is_contiguous()
    assert v_buf.dim() == 4 and v_buf.shape[1] == hrast and v_buf.shape[2] == wrast
    assert v_buf.shape[3] == H * D
    nh, nw = hrast // 8, wrast // 8
    nhw = nh * nw
    Nqb = Nq // 128
    Nkvb = Nkv // 128
    assert Nkvb % nhw == 0 and Nqb % nhw == 0
    assert v_f0 + (Nkvb // nhw) * 2 <= v_buf.shape[0]
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)
    bm = block_mask[..., :Nqb, :Nkvb]
    assert bm.shape == (H, Nqb, Nkvb)
    idx, cnt = _make_csr_fused(bm) if _FUSED_CSR else _make_csr(bm)
    assert idx.stride(2) == 1
    o = torch.empty_like(q)                    # (Nq, H, D), RASTER token order
    scale_log2e = float(sm_scale) * 1.4426950408889634
    layout = ttgl.NVMMASharedLayout(swizzle_byte_width=128,
                                    element_bitwidth=16, rank=2)
    q_desc = _GluonTensorDescriptor.from_tensor(
        q.view(Nq, H * D), [BLOCK_M, 64], layout)
    k_desc = _GluonTensorDescriptor.from_tensor(
        k.view(Nkv, H * D), [BLOCK_N, 64], layout)
    v_desc = _GluonTensorDescriptor.from_tensor(
        v_buf, [2, 8, 8, 64], layout)
    grid = (Nqb, H)
    _bsfa_v2_kernel[grid](
        q_desc, k_desc, v_desc, o, idx, cnt, scale_log2e,
        o.stride(1), o.stride(0), o.stride(2),
        idx.stride(0), idx.stride(1), idx.stride(2),
        cnt.stride(0), cnt.stride(1),
        v_f0, nhw, nw, hrast, wrast,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=D, NBUF=_NBUF,
        PROD_REGS=_PROD_REGS, RASTER_V=True, RASTER_OUT=True,
        num_warps=4)
    return o
