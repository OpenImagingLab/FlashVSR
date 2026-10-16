#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase-3 standalone bench for the block-sparse attention kernel rewrite.

Times kernel candidates at the exact steady-state reference shape
(q 8448 x kv 25344, h12, d128, block 128, density ~0.606) against:
  - v1 `_bsfa_tma_kernel_snd` (the current production kernel, kernel-only)
  - cuDNN dense SDPA (ceiling reference)
  - `block_sparse_attn` (correctness reference)

Usage (from examples/WanVSR/):
  python profiling/bench_attn_v2.py --capture   # one-time: save a REAL mask
  python profiling/bench_attn_v2.py             # bench all available routes
  python profiling/bench_attn_v2.py --routes v1,ws,occ,v2

The real mask is captured from a short pipeline run (chunk 2, last DiT block)
so the per-row cnt distribution and spatial locality match production.
"""
import argparse
import math
import os
import sys
import time

_here = os.path.dirname(os.path.abspath(__file__))
_wanvsr = os.path.dirname(_here)
_root = os.path.dirname(os.path.dirname(_wanvsr))
sys.path.insert(0, _wanvsr)
sys.path.insert(0, _root)

import torch  # noqa: E402

DEV = "cuda"
DT = torch.bfloat16
MASK_CACHE = os.path.join(_here, "cache", "attn_mask_768_steady.pt")

# Reference shape (steady chunk @768x1408). NOTE: the attention consumes the
# UNTRIMMED kv window (pre 3 slots + 1 new = 264 blocks = 33792 tokens) at
# density 0.4545 — FLOP-identical to ANALYSIS's "kv 25344 @ 0.606" framing
# (avg ~120 active kv blocks per q row either way), but the kernel's real
# iteration space is 264 blocks.
NQ, NKV, H, D = 8448, 33792, 12, 128


def bench(fn, it=30, warm=10):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(it):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / it * 1e3  # ms


def cos_max(a, b):
    af, bf = a.float().flatten(), b.float().flatten()
    cos = torch.nn.functional.cosine_similarity(af, bf, dim=0).item()
    return cos, (af - bf).abs().max().item()


# ---------------------------------------------------------------------------
# Real-mask capture (monkeypatch, no production code changes)
# ---------------------------------------------------------------------------

def capture_real_mask():
    os.environ.setdefault("FLASHVSR_CONV3D_BACKEND", "gemm")
    os.environ.setdefault("FLASHVSR_TCDECODER_CHANNELS_LAST", "1")
    os.environ.setdefault("FLASHVSR_FUSE_NORM", "1")
    os.environ.setdefault("FLASHVSR_ATTN_BACKEND", "triton")
    os.environ.setdefault("FLASHVSR_CACHE_MOD", "1")
    os.environ.setdefault("FLASHVSR_CACHE_MASK_BIAS", "1")
    os.environ.setdefault("FLASHVSR_FUSE_ROPE", "1")
    os.environ.setdefault("FLASHVSR_KV_RINGBUF", "1")
    os.environ.setdefault("FLASHVSR_ATTN_STRIDED_IO", "1")
    os.environ.setdefault("FLASHVSR_MASKGEN_LEAN", "1")
    os.environ.setdefault("FLASHVSR_LQPROJ_LEAN", "1")
    os.environ["FLASHVSR_PROF_FRAMES"] = "41"   # chunks 0..2
    os.environ["FLASHVSR_PROF_STEADY"] = "off"
    os.environ["FLASHVSR_PROF_WARMUP"] = "0"

    import importlib.util
    import diffsynth.models.wan_video_dit as ditmod

    captured = {}
    orig = ditmod.generate_draft_block_mask

    def wrapper(*args, **kwargs):
        m = orig(*args, **kwargs)
        # keep the LAST steady-chunk mask: q side 66 blocks, kv fully grown
        if m.shape[-2] * 128 == NQ and m.shape[-1] * 128 == NKV:
            captured["mask"] = m.detach().to("cpu", torch.bool).clone()
        return m

    ditmod.generate_draft_block_mask = wrapper
    try:
        spec = importlib.util.spec_from_file_location(
            "infer_v1_1_tiny", os.path.join(_wanvsr, "infer_flashvsr_v1.1_tiny.py"))
        _infer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_infer)
        pipe = _infer.init_pipeline()
        sys.path.insert(0, _here)
        from run_pipe_target import build_lq  # reuse the LQ cache
        LQ, _, _ = build_lq("./inputs/example0.mp4", 768, 1408, 41)
        with torch.no_grad():
            pipe(prompt="", negative_prompt="", cfg_scale=1.0,
                 num_inference_steps=1, seed=0, LQ_video=LQ, num_frames=41,
                 height=1408, width=768, is_full_block=False, if_buffer=True,
                 topk_ratio=2.0 * 768 * 1280 / (1408 * 768), kv_ratio=3.0,
                 local_range=11, color_fix=True)
    finally:
        ditmod.generate_draft_block_mask = orig
    assert "mask" in captured, "no steady-shaped mask seen"
    m = captured["mask"]
    torch.save(m, MASK_CACHE)
    d = m.float().mean().item()
    print(f"[capture] saved {MASK_CACHE} shape={tuple(m.shape)} density={d:.4f}")


def load_mask():
    if os.path.exists(MASK_CACHE):
        m = torch.load(MASK_CACHE, map_location="cpu", weights_only=True)
        print(f"[mask] real mask {tuple(m.shape)} density={m.float().mean():.4f}")
        return m.to(DEV)
    print("[mask] WARNING: no captured mask; synthesizing density-0.4545")
    torch.manual_seed(0)
    m = torch.rand(1, H, NQ // 128, NKV // 128) < 0.4545
    return m.to(DEV)


# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------

def ref_block_sparse(q_snd, k_snd, v_snd, mask4):
    from block_sparse_attn import block_sparse_attn_func
    seqlen, seqlen_kv = q_snd.shape[0], k_snd.shape[0]
    cu_q = torch.tensor([0, seqlen], device=DEV, dtype=torch.int32)
    cu_k = torch.tensor([0, seqlen_kv], device=DEV, dtype=torch.int32)
    hmt = torch.tensor([1] * H, device=DEV, dtype=torch.int32)

    def call():
        return block_sparse_attn_func(
            q_snd, k_snd, v_snd, cu_q, cu_k, hmt, None, mask4,
            seqlen, seqlen_kv, 0.0, deterministic=False, softmax_scale=None,
            is_causal=False, exact_streaming=False, return_attn_probs=False)
    return call


def ref_dense(q_snd, k_snd, v_snd):
    from torch.nn.attention import sdpa_kernel, SDPBackend
    qd = q_snd.transpose(0, 1).unsqueeze(0).contiguous()
    kd = k_snd.transpose(0, 1).unsqueeze(0).contiguous()
    vd = v_snd.transpose(0, 1).unsqueeze(0).contiguous()

    def call():
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            return torch.nn.functional.scaled_dot_product_attention(qd, kd, vd)
    return call


# ---------------------------------------------------------------------------
# v1 kernel-only launcher (mirrors triton_block_sparse_attention_snd's TMA path)
# ---------------------------------------------------------------------------

def make_v1_kernel_only(q, k, v, bm, BLOCK_M=128, BLOCK_N=128, num_warps=8,
                        num_stages=3, kernel=None):
    import triton
    from triton.tools.tensor_descriptor import TensorDescriptor
    from diffsynth.models.triton_block_sparse_attn import (
        _make_csr, _bsfa_tma_kernel_snd)
    if kernel is None:
        kernel = _bsfa_tma_kernel_snd
    Nq, Hh, Dd = q.shape
    Nkv = k.shape[0]
    sm_scale = 1.0 / math.sqrt(Dd)
    Nqb = triton.cdiv(Nq, BLOCK_M)
    Nkvb = triton.cdiv(Nkv, BLOCK_N)
    # expand 128-granular mask rows/cols exactly onto smaller tiles
    em = bm
    if BLOCK_M != 128:
        em = em.repeat_interleave(128 // BLOCK_M, dim=1)
    if BLOCK_N != 128:
        em = em.repeat_interleave(128 // BLOCK_N, dim=2)
    assert em.shape[1] == Nqb and em.shape[2] == Nkvb
    idx, cnt = _make_csr(em)
    o = torch.empty_like(q)
    q2 = q.view(Nq, Hh * Dd)
    k2 = k.view(Nkv, Hh * Dd)
    v2 = v.view(Nkv, Hh * Dd)
    q_desc = TensorDescriptor.from_tensor(q2, [BLOCK_M, Dd])
    k_desc = TensorDescriptor.from_tensor(k2, [BLOCK_N, Dd])
    v_desc = TensorDescriptor.from_tensor(v2, [BLOCK_N, Dd])
    grid = (Nqb, Hh)

    def call():
        kernel[grid](
            q_desc, k_desc, v_desc, o, idx, cnt, sm_scale,
            o.stride(1), o.stride(0), o.stride(2),
            idx.stride(0), idx.stride(1), idx.stride(2),
            cnt.stride(0), cnt.stride(1),
            Hh, Nq, Nkv, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=Dd,
            num_warps=num_warps, num_stages=num_stages)
        return o
    return call


# ---------------------------------------------------------------------------
# Route WS: one-line tl.range(warp_specialize=True) variant of the v1 kernel
# ---------------------------------------------------------------------------
try:
    import triton
    import triton.language as tl

    @triton.jit
    def _bsfa_tma_kernel_snd_ws(
        q_desc, k_desc, v_desc, O, KVIdx, KVCnt, sm_scale,
        soh, som, sok, sih, sim, sic, sch, scm, H, N_Q, N_KV,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
    ):
        start_m = tl.program_id(0)
        off_h = tl.program_id(1)
        col0 = off_h * HEAD_DIM
        q = q_desc.load([start_m * BLOCK_M, col0])
        qs = (q * sm_scale).to(q.dtype)
        m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
        l_i = tl.zeros([BLOCK_M], tl.float32)
        acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
        offs_n = tl.arange(0, BLOCK_N)
        cnt = tl.load(KVCnt + off_h * sch + start_m * scm)
        base = KVIdx + off_h * sih + start_m * sim
        for j in tl.range(0, cnt, warp_specialize=True):
            kvb = tl.load(base + j * sic)
            kk = k_desc.load([kvb * BLOCK_N, col0])
            qk = tl.dot(qs, kk.T)
            n = kvb * BLOCK_N + offs_n
            qk = tl.where(n[None, :] < N_KV, qk, -float("inf"))
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            p = tl.math.exp2((qk - m_ij[:, None]) * 1.44269504)
            alpha = tl.math.exp2((m_i - m_ij) * 1.44269504)
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None]
            vv = v_desc.load([kvb * BLOCK_N, col0])
            acc += tl.dot(p.to(vv.dtype), vv)
            m_i = m_ij
        l_safe = tl.where(l_i == 0.0, 1.0, l_i)
        acc = acc / l_safe[:, None]
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_k = tl.arange(0, HEAD_DIM)
        tl.store(O + off_h * soh + offs_m[:, None] * som + offs_k[None, :] * sok,
                 acc.to(O.dtype.element_ty), mask=offs_m[:, None] < N_Q)
except Exception:  # pragma: no cover
    _bsfa_tma_kernel_snd_ws = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", action="store_true")
    ap.add_argument("--routes", default="v1,dense,ws,occ,v2")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--check", action="store_true", help="cos vs block_sparse")
    args = ap.parse_args()

    if args.capture:
        capture_real_mask()
        return

    torch.manual_seed(0)
    routes = args.routes.split(",")
    bm4 = load_mask()                     # (1, H, Nqb, Nkvb) bool
    bm = bm4[0]
    q = torch.randn(NQ, H, D, device=DEV, dtype=DT)
    k = torch.randn(NKV, H, D, device=DEV, dtype=DT)
    v = torch.randn(NKV, H, D, device=DEV, dtype=DT)
    density = bm.float().mean().item()
    print(f"[shape] q {NQ} kv {NKV} h{H} d{D}  density {density:.4f}")

    ref_out = None
    if args.check:
        ref_out = ref_block_sparse(q, k, v, bm4)()

    results = {}
    if "v1" in routes:
        call = make_v1_kernel_only(q, k, v, bm)
        t = bench(call, args.iters)
        results["v1 kernel-only (M128 N128 w8 s3)"] = (t, call())
    if "dense" in routes:
        t = bench(ref_dense(q, k, v), args.iters)
        results["cuDNN dense (full attention)"] = (t, None)
        print(f"[ref ] ideal sparse = dense x {density:.3f} = {t*density:.3f} ms")
    if "ws" in routes and _bsfa_tma_kernel_snd_ws is not None:
        for ns in (2, 3, 4):
            try:
                call = make_v1_kernel_only(q, k, v, bm, num_stages=ns,
                                           kernel=_bsfa_tma_kernel_snd_ws)
                t = bench(call, args.iters)
                results[f"ws one-liner (M128 N128 w8 s{ns})"] = (t, call())
            except Exception as e:
                print(f"[ws s{ns}] FAILED: {type(e).__name__}: {str(e)[:200]}")
    if "occ" in routes:
        for (bmz, bnz, w, ns) in ((128, 64, 8, 3), (128, 64, 8, 4),
                                  (64, 128, 4, 2), (64, 64, 4, 2),
                                  (64, 64, 4, 3), (64, 64, 8, 3)):
            try:
                call = make_v1_kernel_only(q, k, v, bm, BLOCK_M=bmz,
                                           BLOCK_N=bnz, num_warps=w,
                                           num_stages=ns)
                t = bench(call, args.iters)
                results[f"occ (M{bmz} N{bnz} w{w} s{ns})"] = (t, call())
            except Exception as e:
                print(f"[occ M{bmz}N{bnz}w{w}s{ns}] FAILED: {type(e).__name__}: {str(e)[:160]}")
    if "v2" in routes:
        try:
            from diffsynth.models.triton_block_sparse_attn_v2 import (
                triton_block_sparse_attention_v2, bsfa_v2_kernel_only)
            call = bsfa_v2_kernel_only(q, k, v, bm)
            t = bench(call, args.iters)
            results["v2 gluon WS"] = (t, call())
        except ImportError:
            print("[v2] module not present yet")
        except Exception as e:
            print(f"[v2] FAILED: {type(e).__name__}: {str(e)[:300]}")

    # 2 GEMMs (QK^T, PV) x 2 flops/MAC x D per 128x128 active block
    flops = 4 * D * (bm.float().sum().item() * 128 * 128)
    print(f"\n{'route':42s} {'ms':>8s} {'TF/s':>7s}  cos/max|d| vs sparse")
    for name, (t, out) in results.items():
        tf = flops / t / 1e9
        extra = ""
        if ref_out is not None and out is not None:
            c, mx = cos_max(out, ref_out)
            extra = f"cos={c:.6f} max|d|={mx:.4f}"
        print(f"{name:42s} {t:8.3f} {tf:7.0f}  {extra}")


if __name__ == "__main__":
    main()
