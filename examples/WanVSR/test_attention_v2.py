#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase-3 correctness gate for the warp-specialized attention v2
(FLASHVSR_ATTN_BACKEND=triton2).

1. Kernel level (real steady shape q 8448 x kv 33792, h12, d128):
   cosine >= 0.9999 vs `block_sparse_attn` on randomized inputs across mask
   densities INCLUDING the captured real mask and degenerate all-true /
   all-false rows (all-false rows must be exactly zero, matching the v1
   l==0 -> output 0 semantics; block_sparse_attn emits the same).
2. E2E (example0 @768x1408, full 2A recommended stack): PSNR(sparse, triton2)
   >= 49 dB, plus 2x triton2 repeat bit-identical (determinism + multi-chunk
   streaming KV/cache bit-stability).

Run from examples/WanVSR/:
    python test_attention_v2.py [kernel|e2e|all]
"""
import math
import os
import sys
import time

# 2A recommended stack + knobs must be set before diffsynth imports.
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

import importlib.util
import numpy as np
import torch

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.dirname(os.path.dirname(_here)))

NQ, NKV, H, D = 8448, 33792, 12, 128
COS_GATE = 0.9999
PSNR_GATE = 49.0
MASK_CACHE = os.path.join(_here, "profiling", "cache",
                          "attn_mask_768_steady.pt")


def cos_max(a, b):
    af, bf = a.float().flatten(), b.float().flatten()
    cos = torch.nn.functional.cosine_similarity(af, bf, dim=0).item()
    return cos, (af - bf).abs().max().item()


def ref_sparse(q, k, v, bm4):
    from block_sparse_attn import block_sparse_attn_func
    sq, sk = q.shape[0], k.shape[0]
    cu_q = torch.tensor([0, sq], device="cuda", dtype=torch.int32)
    cu_k = torch.tensor([0, sk], device="cuda", dtype=torch.int32)
    hmt = torch.tensor([1] * H, device="cuda", dtype=torch.int32)
    return block_sparse_attn_func(
        q, k, v, cu_q, cu_k, hmt, None, bm4, sq, sk, 0.0,
        deterministic=False, softmax_scale=None, is_causal=False,
        exact_streaming=False, return_attn_probs=False)


def test_kernel():
    from diffsynth.models.triton_block_sparse_attn_v2 import (
        triton_block_sparse_attention_v2 as v2)
    torch.manual_seed(0)
    q = torch.randn(NQ, H, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(NKV, H, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(NKV, H, D, device="cuda", dtype=torch.bfloat16)
    Nqb, Nkvb = NQ // 128, NKV // 128

    cases = []
    if os.path.exists(MASK_CACHE):
        m = torch.load(MASK_CACHE, map_location="cpu",
                       weights_only=True).cuda()[0]
        cases.append((f"real mask (d={m.float().mean():.3f})", m))
    for dens in (0.1, 0.3, 0.45, 0.9):
        cases.append((f"random d={dens}",
                      torch.rand(H, Nqb, Nkvb, device="cuda") < dens))
    cases.append(("all-true", torch.ones(H, Nqb, Nkvb, dtype=torch.bool,
                                         device="cuda")))
    cases.append(("all-false", torch.zeros(H, Nqb, Nkvb, dtype=torch.bool,
                                           device="cuda")))
    bmx = torch.rand(H, Nqb, Nkvb, device="cuda") < 0.45
    bmx[:, 5] = False          # all-false q rows in every head
    bmx[3] = False             # a fully masked head
    bmx[:, 17] = True          # all-true q rows
    cases.append(("degenerate rows/heads", bmx))

    ok = True
    for name, bm in cases:
        out = v2(q, k, v, bm)
        ref = ref_sparse(q, k, v, bm.unsqueeze(0))
        if bm.any():
            # compare on rows that have any active block (masked-out rows are
            # zero in BOTH new kernels; block_sparse leaves them zero too)
            c, md = cos_max(out, ref)
            passed = c >= COS_GATE
        else:
            md = out.float().abs().max().item()
            c, passed = float("nan"), md == 0.0
        # all-false rows must be exactly zero
        zero_ok = True
        for h in range(H):
            rows = (~bm[h].any(-1)).nonzero().flatten()
            for r in rows[:2]:
                blk = out[r * 128:(r + 1) * 128, h]
                zero_ok &= bool((blk == 0).all())
        ok &= passed and zero_ok
        print(f"  {name:28s} cos={c:.6f} max|d|={md:.5f} "
              f"zero-rows={'ok' if zero_ok else 'FAIL'} "
              f"{'PASS' if passed and zero_ok else 'FAIL'}")
    # determinism: same inputs -> bit-identical outputs
    o1 = v2(q, k, v, cases[0][1])
    o2 = v2(q, k, v, cases[0][1])
    det = torch.equal(o1, o2)
    ok &= det
    print(f"  kernel determinism (2 calls bit-identical): "
          f"{'PASS' if det else 'FAIL'}")
    return ok


def _run_pipe(pipe, LQ, F, H_, W_):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        vid = pipe(prompt="", negative_prompt="", cfg_scale=1.0,
                   num_inference_steps=1, seed=0, LQ_video=LQ, num_frames=F,
                   height=H_, width=W_, is_full_block=False, if_buffer=True,
                   topk_ratio=2.0 * 768 * 1280 / (H_ * W_), kv_ratio=3.0,
                   local_range=11, color_fix=True)
    torch.cuda.synchronize()
    return vid.float().cpu(), time.perf_counter() - t0


def test_e2e():
    import diffsynth.models.wan_video_dit as ditmod
    spec = importlib.util.spec_from_file_location(
        "infer_v1_1_tiny", os.path.join(_here, "infer_flashvsr_v1.1_tiny.py"))
    _infer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_infer)
    pipe = _infer.init_pipeline()
    sys.path.insert(0, os.path.join(_here, "profiling"))
    from run_pipe_target import build_lq
    W_, H_, F = 768, 1408, 81
    LQ, _, _ = build_lq("./inputs/example0.mp4", W_, H_, F)

    ditmod._ATTN_BACKEND = "sparse"
    vid_s, dt_s = _run_pipe(pipe, LQ, F, H_, W_)
    ditmod._ATTN_BACKEND = "triton2"
    vid_a, dt_a = _run_pipe(pipe, LQ, F, H_, W_)
    vid_b, _ = _run_pipe(pipe, LQ, F, H_, W_)

    mse = torch.mean((vid_s - vid_a) ** 2).item()
    psnr = float("inf") if mse <= 1e-12 else 10 * math.log10(4.0 / mse)
    md = (vid_s - vid_a).abs().max().item()
    rep = torch.equal(vid_a, vid_b)
    fps = (F - 4) / dt_a
    print(f"  PSNR(sparse, triton2) = {psnr:.2f} dB (gate >= {PSNR_GATE}) "
          f"max|d|={md:.4f}")
    print(f"  triton2 repeat bit-identical (multi-chunk KV stability): "
          f"{'PASS' if rep else 'FAIL'}")
    print(f"  [info] sparse {(F-4)/dt_s:.2f} FPS -> triton2 {fps:.2f} FPS")
    return psnr >= PSNR_GATE and rep


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    ok = True
    if what in ("kernel", "all"):
        print("[kernel-level vs block_sparse_attn]")
        ok &= test_kernel()
    if what in ("e2e", "all"):
        print("[E2E example0 @768x1408]")
        ok &= test_e2e()
    print(f"\n=== test_attention_v2 [{what}]: {'PASS' if ok else 'FAIL'} ===")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
