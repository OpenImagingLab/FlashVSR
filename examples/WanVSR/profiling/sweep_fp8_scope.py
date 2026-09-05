#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 2B-2 dev tool: per-site PSNR/FPS bisection for FLASHVSR_FP8_GEMM.

Toggles fp8_gemm._FP8_SCOPE on a live session to attribute the quality loss
per call site (qkv / o / ffn / lq) and find the best quality/speed scope.
Attribution-only; headline numbers still come from run_pipe_target.py.
"""
import os, sys, time, math, importlib.util

_here = os.path.dirname(os.path.abspath(__file__))
_wanvsr = os.path.dirname(_here)
os.chdir(_wanvsr)
sys.path.insert(0, _wanvsr)

for k, v in {
    "FLASHVSR_CONV3D_BACKEND": "gemm", "FLASHVSR_TCDECODER_CHANNELS_LAST": "1",
    "FLASHVSR_FUSE_NORM": "1", "FLASHVSR_ATTN_BACKEND": "triton",
    "FLASHVSR_CACHE_MOD": "1", "FLASHVSR_CACHE_MASK_BIAS": "1",
    "FLASHVSR_FUSE_ROPE": "1", "FLASHVSR_KV_RINGBUF": "1",
    "FLASHVSR_ATTN_STRIDED_IO": "1", "FLASHVSR_MASKGEN_LEAN": "1",
    "FLASHVSR_LQPROJ_LEAN": "1", "FLASHVSR_FP8_GEMM": "0",
}.items():
    os.environ[k] = v

import numpy as np
from PIL import Image
import imageio
import torch

import diffsynth.models.fp8_gemm as fp8mod

_spec = importlib.util.spec_from_file_location(
    "infer_v1_1_tiny", os.path.join(_wanvsr, "infer_flashvsr_v1.1_tiny.py"))
_infer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_infer)

REF_W, REF_H = 768, 1408
SRC_W, SRC_H = REF_W // 4, REF_H // 4


def build_lq(src):
    rdr = imageio.get_reader(src); total = rdr.count_frames()
    idx = (list(range(total)) + [total - 1] * 4)
    F = _infer.largest_8n1_leq(len(idx)); idx = idx[:F]
    frames = []
    for i in idx:
        img = Image.fromarray(rdr.get_data(i)).convert("RGB").resize((SRC_W, SRC_H), Image.BICUBIC).resize((REF_W, REF_H), Image.BICUBIC)
        t = torch.from_numpy(np.asarray(img, np.uint8)).to(device="cuda", dtype=torch.float32)
        frames.append((t.permute(2, 0, 1) / 255.0 * 2.0 - 1.0).to(torch.bfloat16))
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


def main():
    pipe = _infer.init_pipeline()
    LQ, F = build_lq("./inputs/example0.mp4")
    out = F - 4

    fp8mod._FP8_GEMM = False
    run(pipe, LQ, F)
    v_off, dt = run(pipe, LQ, F)
    print(f"\nbf16 reference: {dt:.3f}s {out/dt:6.2f} FPS")

    scopes = ["ffn1", "qkv,o,lq", "qkv,o,ffn1,lq", "o,ffn1,lq"]
    fp8mod._FP8_GEMM = True
    for sc in scopes:
        fp8mod._FP8_SCOPE = frozenset(s.strip() for s in sc.split(","))
        v_on, dt = run(pipe, LQ, F)
        d = (v_off - v_on).abs()
        print(f"scope {sc:15s}: {dt:.3f}s {out/dt:6.2f} FPS  "
              f"max|d|={d.max().item():.4f} mean|d|={d.mean().item():.6f} "
              f"PSNR={psnr(v_off, v_on):6.2f} dB")
    fp8mod._FP8_GEMM = False
    fp8mod._FP8_SCOPE = frozenset("qkv,o,ffn,lq".split(","))


if __name__ == "__main__":
    main()
