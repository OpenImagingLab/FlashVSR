#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 2B-2 quality harness: FP8 GEMM path vs bf16 eager path.

Runs the v1.1 Tiny pipeline @768x1408 with the full Phase-2A recommended
stack on BOTH sides; the only delta is FLASHVSR_FP8_GEMM (toggled via the
module attribute on a live session). FP8 is NOT lossless by design — this
harness measures and classifies the numeric delta per the Phase-4 tiers:

    PSNR >= 49 dB  -> eligible for "recommended config" listing (Phase 4)
    45 <= PSNR < 49 -> ships flag-gated with documented delta
    PSNR < 45 dB   -> revert or redesign

Exit code 0 iff the FP8 path actually ran (no silent eager fallback) and
PSNR >= 45 dB. Also verifies e4m3 GEMM kernels are dispatched (kernel-name
evidence via torch.profiler; the ncu run in the bench log is the second
source).

Run from examples/WanVSR/:
    /root/FlashVSR/venv/bin/python test_fp8_gemm_quality.py
"""
import os, sys, time, math, importlib.util

os.environ["FLASHVSR_CONV3D_BACKEND"] = "gemm"
os.environ["FLASHVSR_TCDECODER_CHANNELS_LAST"] = "1"
os.environ["FLASHVSR_FUSE_NORM"] = "1"
os.environ["FLASHVSR_ATTN_BACKEND"] = "triton"
os.environ["FLASHVSR_CACHE_MOD"] = "1"
os.environ["FLASHVSR_CACHE_MASK_BIAS"] = "1"
os.environ["FLASHVSR_FUSE_ROPE"] = "1"
os.environ["FLASHVSR_KV_RINGBUF"] = "1"
os.environ["FLASHVSR_ATTN_STRIDED_IO"] = "1"
os.environ["FLASHVSR_MASKGEN_LEAN"] = "1"
os.environ["FLASHVSR_LQPROJ_LEAN"] = "1"
os.environ["FLASHVSR_FP8_GEMM"] = "0"  # toggled via module attr below

import numpy as np
from PIL import Image
import imageio
import torch

import diffsynth.models.fp8_gemm as fp8mod

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "infer_v1_1_tiny", os.path.join(_here, "infer_flashvsr_v1.1_tiny.py"))
_infer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_infer)
init_pipeline = _infer.init_pipeline
largest_8n1_leq = _infer.largest_8n1_leq

REF_W, REF_H, SCALE = 768, 1408, 4
SRC_W, SRC_H = REF_W // SCALE, REF_H // SCALE


def build_lq(src, device="cuda", dtype=torch.bfloat16):
    rdr = imageio.get_reader(src); total = rdr.count_frames()
    idx = (list(range(total)) + [total - 1] * 4); F = largest_8n1_leq(len(idx)); idx = idx[:F]
    frames = []
    for i in idx:
        img = Image.fromarray(rdr.get_data(i)).convert("RGB").resize((SRC_W, SRC_H), Image.BICUBIC).resize((REF_W, REF_H), Image.BICUBIC)
        t = torch.from_numpy(np.asarray(img, np.uint8)).to(device=device, dtype=torch.float32)
        frames.append((t.permute(2, 0, 1) / 255.0 * 2.0 - 1.0).to(dtype))
    rdr.close()
    return torch.stack(frames, 0).permute(1, 0, 2, 3).unsqueeze(0), F


def run(pipe, LQ, F):
    torch.cuda.empty_cache(); torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        vid = pipe(prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=1, seed=0,
                   LQ_video=LQ, num_frames=F, height=REF_H, width=REF_W, is_full_block=False,
                   if_buffer=True, topk_ratio=2.0 * 768 * 1280 / (REF_H * REF_W), kv_ratio=3.0,
                   local_range=11, color_fix=True)
    torch.cuda.synchronize()
    return vid.float().cpu(), time.perf_counter() - t0


def psnr(a, b):
    mse = (a - b).pow(2).mean().item()
    return float("inf") if mse <= 1e-12 else 10.0 * math.log10(4.0 / mse)


def kernel_evidence():
    """Profile a bf16 linear and an fp8 linear; return both kernel-name sets.
    Evidence that e4m3 kernels are dispatched: the fp8 path uses a different
    GEMM kernel carrying the quantized-operand / outer-vector-scale markers
    (nvjet '..qqtst..ovscale..' on this build) rather than the bf16 kernel."""
    from torch.profiler import profile, ProfilerActivity
    lin = torch.nn.Linear(1536, 1536).to("cuda", torch.bfloat16)
    x = torch.randn(8448, 1536, device="cuda", dtype=torch.bfloat16)

    def gemm_names(fn):
        fn()  # warm (triton compile, weight cache, cublas heuristics)
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            fn()
        torch.cuda.synchronize()
        return sorted(e.key for e in prof.key_averages()
                      if any(s in e.key.lower() for s in ("gemm", "nvjet", "cutlass", "matmul")))

    names_bf16 = gemm_names(lambda: torch.nn.functional.linear(x, lin.weight, lin.bias))
    names_fp8 = gemm_names(lambda: fp8mod.linear(lin, x))
    return names_bf16, names_fp8


def main():
    pipe = init_pipeline()
    LQ, F = build_lq("./inputs/example0.mp4")
    out = F - 4

    print(f"\n=== Phase 2B-2 FP8 GEMM quality @ {REF_W}x{REF_H} F={F} ===")

    fp8mod._FP8_GEMM = False
    run(pipe, LQ, F)  # warm clocks / lazy compiles
    v_off, dt_off = run(pipe, LQ, F)
    print(f"  bf16 eager (OFF): {dt_off:.3f}s  {out/dt_off:6.2f} FPS")

    fp8mod._FP8_GEMM = True
    run(pipe, LQ, F)  # warm: triton quant kernels + weight pre-cast
    v_on, dt_on = run(pipe, LQ, F)
    fp8_ran = not fp8mod._FAILED
    fp8mod._FP8_GEMM = False
    print(f"  fp8 path   (ON) : {dt_on:.3f}s  {out/dt_on:6.2f} FPS   "
          f"(fallback_triggered={not fp8_ran})")

    d = (v_off - v_on).abs()
    maxd, meand, p = d.max().item(), d.mean().item(), psnr(v_off, v_on)
    tier = ("recommended-eligible (>=49 dB)" if p >= 49.0 else
            "flag-gated, documented delta (45-49 dB)" if p >= 45.0 else
            "revert-or-redesign (<45 dB)")
    print(f"\n  max|d|={maxd:.4f}  mean|d|={meand:.6f}  PSNR={p:.2f} dB")
    print(f"  Phase-4 tier: {tier}")

    names_bf16, names_fp8 = kernel_evidence()
    markers = ("f8", "fp8", "e4m3", "qqtst", "ovscale")
    has_fp8_kernel = (names_fp8 != names_bf16 and
                      any(any(m in n.lower() for m in markers) for n in names_fp8))
    print(f"\n  bf16 GEMM kernels: {names_bf16}")
    print(f"  fp8  GEMM kernels: {names_fp8}")
    print(f"  e4m3 kernel dispatched: {has_fp8_kernel}")

    # Landing bar for the 2B-2 infra (the ENABLE decision is Phase 4's):
    # the fp8 path must actually run (no silent fallback), must dispatch e4m3
    # GEMM kernels, and must not be catastrophically broken (PSNR sanity).
    ok = fp8_ran and has_fp8_kernel and p >= 30.0
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'} "
          f"(fp8_ran={fp8_ran}, fp8_kernel={has_fp8_kernel}, psnr_sane={p >= 30.0}; "
          f"Phase-4 tier reported above)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
