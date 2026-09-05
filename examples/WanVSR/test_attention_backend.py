#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E2E speed + quality (PSNR) for the adaptive attention backend @768x1408.

Compares attention backends with conv3d=gemm and TCDecoder channels_last fixed:
  - sparse : original block_sparse (baseline reference for PSNR)
  - auto   : density-adaptive (cuDNN dense when mask density >= threshold)

Reports denoise FPS and PSNR(sparse, auto) so we can see both the speedup and
how much the dense routing changes the output.

Run from examples/WanVSR/ :
    python test_attention_backend.py
"""
import os, time, math, importlib.util
import numpy as np
from PIL import Image
import imageio
import torch

os.environ["FLASHVSR_CONV3D_BACKEND"] = "gemm"
os.environ["FLASHVSR_TCDECODER_CHANNELS_LAST"] = "1"
import utils.utils as wanutils; wanutils._CONV3D_BACKEND = "gemm"
import diffsynth.models.wan_video_dit as ditmod

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("infer_v1_1_tiny", os.path.join(_here, "infer_flashvsr_v1.1_tiny.py"))
_infer = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_infer)
init_pipeline = _infer.init_pipeline; largest_8n1_leq = _infer.largest_8n1_leq

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


def run(pipe, LQ, th, tw, F):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        vid = pipe(prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=1, seed=0,
                   LQ_video=LQ, num_frames=F, height=th, width=tw, is_full_block=False, if_buffer=True,
                   topk_ratio=2.0 * 768 * 1280 / (th * tw), kv_ratio=3.0, local_range=11, color_fix=True)
    torch.cuda.synchronize()
    return vid.float().cpu(), time.perf_counter() - t0


def psnr(a, b):
    mse = torch.mean((a - b) ** 2).item()
    return float("inf") if mse <= 1e-12 else 10 * math.log10(4.0 / mse)


def main():
    pipe = init_pipeline()
    LQ, F = build_lq("./inputs/example0.mp4")
    th, tw = REF_H, REF_W
    out_frames = F - 4

    ditmod._ATTN_BACKEND = "sparse"
    vid_s, dt_s = run(pipe, LQ, th, tw, F)
    fps_s = out_frames / dt_s

    ditmod._ATTN_BACKEND = "auto"
    vid_a, dt_a = run(pipe, LQ, th, tw, F)
    fps_a = out_frames / dt_a

    print(f"\n=== Attention backend comparison @ {tw}x{th} (conv3d=gemm, TCDec=NHWC) ===")
    print(f"  sparse (block_sparse): {dt_s:.2f}s  {fps_s:6.2f} FPS  {fps_s/17:.2f}x A100")
    print(f"  auto   (adaptive dense): {dt_a:.2f}s  {fps_a:6.2f} FPS  {fps_a/17:.2f}x A100")
    print(f"  speedup auto vs sparse: {dt_s/dt_a:.2f}x")
    print(f"  PSNR(sparse, auto):     {psnr(vid_s, vid_a):.2f} dB   max|diff|={(vid_s-vid_a).abs().max().item():.4f}")


if __name__ == "__main__":
    main()
