#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase-1 E2E validation for the Hopper im2col+GEMM conv3d backend.

Runs the v1.1 Tiny pipeline twice on the same input/seed:
  (1) FLASHVSR_CONV3D_BACKEND=auto  -> baseline (cuDNN conv3d)
  (2) FLASHVSR_CONV3D_BACKEND=gemm  -> tensor-core im2col+GEMM

Reports, per backend:
  - denoise wall time + pixel-normalized FPS,
  - peak CUDA memory,
and compares the two output videos via PSNR (bf16 noise level expected).

Run from examples/WanVSR/ :
    python test_e2e_conv3d_gemm.py
"""
import os, time, math, importlib.util
import numpy as np
import torch

import utils.utils as wanutils  # to toggle the conv3d backend at runtime

# The infer script's filename contains a dot, so load it via importlib.
_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "infer_v1_1_tiny", os.path.join(_here, "infer_flashvsr_v1.1_tiny.py")
)
_infer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_infer)
prepare_input_tensor = _infer.prepare_input_tensor
init_pipeline = _infer.init_pipeline


def run_once(pipe, LQ, th, tw, F, seed, sparse_ratio):
    torch.cuda.empty_cache(); torch.cuda.ipc_collect()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    video = pipe(
        prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=1, seed=seed,
        LQ_video=LQ, num_frames=F, height=th, width=tw, is_full_block=False, if_buffer=True,
        topk_ratio=sparse_ratio*768*1280/(th*tw),
        kv_ratio=3.0, local_range=11, color_fix=True,
    )
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9
    return video.float().cpu(), dt, peak


def psnr(a, b):
    # a, b in [-1, 1]
    mse = torch.mean((a - b) ** 2).item()
    if mse <= 1e-12:
        return float("inf")
    return 10.0 * math.log10((2.0 ** 2) / mse)


def main():
    INPUT = os.environ.get("FLASHVSR_TEST_INPUT", "./inputs/example0.mp4")
    seed, scale, sparse_ratio = 0, 4.0, 2.0
    device = "cuda"

    pipe = init_pipeline()
    LQ, th, tw, F, fps = prepare_input_tensor(INPUT, scale=scale, device=device)
    out_frames = F - 4  # padding frames dropped
    print(f"\nInput: {INPUT}  target {tw}x{th}  frames(out)={out_frames}")

    # ---- baseline (auto) ----
    wanutils._CONV3D_BACKEND = "auto"
    print("\n=== Backend: auto (cuDNN conv3d) ===")
    vid_auto, dt_auto, peak_auto = run_once(pipe, LQ, th, tw, F, seed, sparse_ratio)
    fps_auto = out_frames / dt_auto
    norm_auto = fps_auto * (th * tw) / (768 * 1408)
    print(f"  denoise: {dt_auto:.2f}s  FPS={fps_auto:.2f}  norm@768x1408={norm_auto:.2f}  peak={peak_auto:.1f} GB")

    # ---- gemm ----
    wanutils._CONV3D_BACKEND = "gemm"
    print("\n=== Backend: gemm (im2col + tensor-core GEMM) ===")
    vid_gemm, dt_gemm, peak_gemm = run_once(pipe, LQ, th, tw, F, seed, sparse_ratio)
    fps_gemm = out_frames / dt_gemm
    norm_gemm = fps_gemm * (th * tw) / (768 * 1408)
    print(f"  denoise: {dt_gemm:.2f}s  FPS={fps_gemm:.2f}  norm@768x1408={norm_gemm:.2f}  peak={peak_gemm:.1f} GB")

    # ---- compare ----
    p = psnr(vid_auto, vid_gemm)
    maxdiff = (vid_auto - vid_gemm).abs().max().item()
    print("\n=== Phase-1 E2E summary ===")
    print(f"  speedup (denoise):   {dt_auto/dt_gemm:.2f}x   ({dt_auto:.2f}s -> {dt_gemm:.2f}s)")
    print(f"  FPS:                 {fps_auto:.2f} -> {fps_gemm:.2f}  (norm {norm_auto:.2f} -> {norm_gemm:.2f})")
    print(f"  peak mem:            {peak_auto:.1f} GB -> {peak_gemm:.1f} GB  (+{peak_gemm-peak_auto:.1f} GB)")
    print(f"  output PSNR(auto,gemm): {p:.2f} dB   max|diff|={maxdiff:.4f}")
    ok = p >= 40.0  # bf16-level agreement
    print("  RESULT:", "PASS (PSNR>=40 dB)" if ok else f"CHECK (PSNR={p:.1f} dB)")


if __name__ == "__main__":
    main()
