#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggressive-sparsity sweep: lower topk -> lower density -> faster sparse attn.

Holds conv3d=gemm, TCDecoder NHWC, attention=sparse. Sweeps the sparse_ratio
(which scales topk_ratio) and reports denoise FPS and PSNR vs the default
(sparse_ratio=2.0) baseline, so we can see the speed/quality tradeoff.

Run from examples/WanVSR/ :
    python test_topk_sweep.py
"""
import os, time, math, importlib.util
import numpy as np
from PIL import Image
import imageio
import torch

os.environ["FLASHVSR_CONV3D_BACKEND"] = "gemm"
os.environ["FLASHVSR_TCDECODER_CHANNELS_LAST"] = "1"
os.environ["FLASHVSR_ATTN_BACKEND"] = "sparse"
import utils.utils as wanutils; wanutils._CONV3D_BACKEND = "gemm"

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


def run(pipe, LQ, th, tw, F, sparse_ratio):
    torch.cuda.empty_cache(); torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        vid = pipe(prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=1, seed=0,
                   LQ_video=LQ, num_frames=F, height=th, width=tw, is_full_block=False, if_buffer=True,
                   topk_ratio=sparse_ratio * 768 * 1280 / (th * tw), kv_ratio=3.0, local_range=11, color_fix=True)
    torch.cuda.synchronize()
    return vid.float().cpu(), time.perf_counter() - t0


def psnr(a, b):
    mse = torch.mean((a - b) ** 2).item()
    return float("inf") if mse <= 1e-12 else 10 * math.log10(4.0 / mse)


def main():
    pipe = init_pipeline()
    LQ, F = build_lq("./inputs/example0.mp4")
    th, tw, out_frames = REF_H, REF_W, F - 4

    # baseline = default sparse_ratio 2.0
    vid_base, dt_base = run(pipe, LQ, th, tw, F, 2.0)
    fps_base = out_frames / dt_base
    print(f"\n=== topk / sparsity sweep @ {tw}x{th} (baseline sparse_ratio=2.0) ===")
    print(f"{'sparse_ratio':>12s} {'FPS':>7s} {'xA100':>6s} {'PSNR_vs_base':>12s}")
    print(f"{2.0:12.2f} {fps_base:7.2f} {fps_base/17:6.2f} {'(baseline)':>12s}")

    for sr in [1.5, 1.0, 0.75, 0.5]:
        vid, dt = run(pipe, LQ, th, tw, F, sr)
        fps = out_frames / dt
        print(f"{sr:12.2f} {fps:7.2f} {fps/17:6.2f} {psnr(vid_base, vid):12.2f}")


if __name__ == "__main__":
    main()
