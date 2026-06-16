#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E2E parity + speed for the norm/elementwise fusion (FLASHVSR_FUSE_NORM).

conv3d=gemm, TCDecoder NHWC, attention=sparse fixed. Runs the pipeline with
fusion OFF then ON (toggled at runtime) and reports FPS + PSNR(off, on).

Run from examples/WanVSR/ :
    python test_fuse_norm.py
"""
import os, time, math, importlib.util
import numpy as np
from PIL import Image
import imageio
import torch

os.environ["FLASHVSR_CONV3D_BACKEND"] = "gemm"
os.environ["FLASHVSR_TCDECODER_CHANNELS_LAST"] = "1"
os.environ["FLASHVSR_ATTN_BACKEND"] = "sparse"
os.environ["FLASHVSR_FUSE_NORM"] = "1"  # import-time so compiled fns exist
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
    torch.cuda.empty_cache(); torch.cuda.synchronize()
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
    th, tw, out_frames = REF_H, REF_W, F - 4

    ditmod._FUSE_NORM = False
    vid_off, dt_off = run(pipe, LQ, th, tw, F)
    fps_off = out_frames / dt_off

    ditmod._FUSE_NORM = True
    # warmup compile
    run(pipe, LQ, th, tw, F)
    vid_on, dt_on = run(pipe, LQ, th, tw, F)
    fps_on = out_frames / dt_on

    print(f"\n=== FUSE_NORM @ {tw}x{th} ===")
    print(f"  OFF: {dt_off:.2f}s  {fps_off:6.2f} FPS  {fps_off/17:.2f}x A100")
    print(f"  ON : {dt_on:.2f}s  {fps_on:6.2f} FPS  {fps_on/17:.2f}x A100")
    print(f"  speedup: {dt_off/dt_on:.2f}x")
    print(f"  PSNR(off,on): {psnr(vid_off, vid_on):.2f} dB   max|diff|={(vid_off-vid_on).abs().max().item():.4f}")


if __name__ == "__main__":
    main()
