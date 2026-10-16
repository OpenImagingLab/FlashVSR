#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A100-reference comparison at 768x1408 (the resolution the README quotes ~17 FPS for).

Builds an LQ tensor whose 4x SR output is exactly 768x1408 (W x H = 768 wide, 1408 tall,
i.e. tH=1408, tW=768) by resizing a source clip to 192x352 and upscaling 4x. Then runs
the v1.1 Tiny pipeline with both conv3d backends and reports denoise FPS vs the A100
reference (~17 FPS @ 768x1408).

Run from examples/WanVSR/ :
    python test_a100_ref_768x1408.py
"""
import os, time, math, importlib.util
import numpy as np
from PIL import Image
import imageio
import torch

import utils.utils as wanutils

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "infer_v1_1_tiny", os.path.join(_here, "infer_flashvsr_v1.1_tiny.py")
)
_infer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_infer)
init_pipeline = _infer.init_pipeline
largest_8n1_leq = _infer.largest_8n1_leq

# A100 reference resolution from README: 768 x 1408 (width x height)
REF_W, REF_H = 768, 1408
SCALE = 4
SRC_W, SRC_H = REF_W // SCALE, REF_H // SCALE  # 192 x 352 LQ


def build_lq(src_video, device="cuda", dtype=torch.bfloat16):
    rdr = imageio.get_reader(src_video)
    total = rdr.count_frames()
    idx = list(range(total)) + [total - 1] * 4
    F = largest_8n1_leq(len(idx))
    idx = idx[:F]
    frames = []
    for i in idx:
        img = Image.fromarray(rdr.get_data(i)).convert("RGB").resize((SRC_W, SRC_H), Image.BICUBIC)
        # 4x upscale to exact REF_W x REF_H, like the infer pipeline (bicubic then no crop needed)
        up = img.resize((REF_W, REF_H), Image.BICUBIC)
        t = torch.from_numpy(np.asarray(up, np.uint8)).to(device=device, dtype=torch.float32)
        t = t.permute(2, 0, 1) / 255.0 * 2.0 - 1.0
        frames.append(t.to(dtype))
    rdr.close()
    vid = torch.stack(frames, 0).permute(1, 0, 2, 3).unsqueeze(0)  # 1 C F H W
    return vid, F


def run_once(pipe, LQ, th, tw, F, seed, sparse_ratio):
    torch.cuda.empty_cache(); torch.cuda.ipc_collect()
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    t0 = time.perf_counter()
    video = pipe(
        prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=1, seed=seed,
        LQ_video=LQ, num_frames=F, height=th, width=tw, is_full_block=False, if_buffer=True,
        topk_ratio=sparse_ratio * 768 * 1280 / (th * tw),
        kv_ratio=3.0, local_range=11, color_fix=True,
    )
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9
    return video.float().cpu(), dt, peak


def main():
    src = os.environ.get("FLASHVSR_TEST_INPUT", "./inputs/example0.mp4")
    seed, sparse_ratio = 0, 2.0
    pipe = init_pipeline()

    LQ, F = build_lq(src)
    th, tw = REF_H, REF_W
    out_frames = F - 4
    print(f"\nA100-ref test: output {tw}x{th} (W x H)  frames(out)={out_frames}  src={src}")
    print(f"README A100 reference: ~17 FPS @ 768x1408\n")

    wanutils._CONV3D_BACKEND = "auto"
    _, dt_a, peak_a = run_once(pipe, LQ, th, tw, F, seed, sparse_ratio)
    fps_a = out_frames / dt_a

    wanutils._CONV3D_BACKEND = "gemm"
    _, dt_g, peak_g = run_once(pipe, LQ, th, tw, F, seed, sparse_ratio)
    fps_g = out_frames / dt_g

    print("=== Results @ 768x1408 (same res as A100 reference) ===")
    print(f"  GH200 auto (cuDNN conv3d):   {dt_a:6.2f}s   {fps_a:6.2f} FPS   peak={peak_a:.1f} GB")
    print(f"  GH200 gemm (tensor-core):    {dt_g:6.2f}s   {fps_g:6.2f} FPS   peak={peak_g:.1f} GB")
    print(f"  A100 (README):                  ---      ~17.00 FPS")
    print()
    print(f"  speedup gemm vs auto:        {dt_a/dt_g:.2f}x")
    print(f"  GH200(gemm) vs A100:         {fps_g/17.0:.2f}x faster")
    print(f"  GH200(auto) vs A100:         {fps_a/17.0:.2f}x faster")


if __name__ == "__main__":
    main()
