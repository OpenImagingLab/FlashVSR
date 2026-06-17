#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lossless parity + speed for the step-invariant caches (Phase B) @768x1408.

Two opt-in caches, both bit-identical to the original (max|diff| must be 0):
  - FLASHVSR_CACHE_MOD       : cache (modulation + t_mod).chunk(...) per block/head
  - FLASHVSR_CACHE_MASK_BIAS : cache the geometry-only local_attn_mask additive bias

Runs the v1.1 Tiny pipeline with conv3d=gemm / TCDec=NHWC / fuse_norm / triton attn
fixed, toggling each cache OFF vs ON and asserting max|diff| == 0. Also reports
denoise FPS for each.

Run from examples/WanVSR/ :
    python test_cache_lossless.py
"""
import os, time, importlib.util
import numpy as np
from PIL import Image
import imageio
import torch

os.environ["FLASHVSR_CONV3D_BACKEND"] = "gemm"
os.environ["FLASHVSR_TCDECODER_CHANNELS_LAST"] = "1"
os.environ["FLASHVSR_FUSE_NORM"] = "1"
os.environ["FLASHVSR_ATTN_BACKEND"] = "triton"
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


def main():
    pipe = init_pipeline()
    LQ, F = build_lq("./inputs/example0.mp4")
    th, tw = REF_H, REF_W
    out = F - 4

    # baseline: both caches OFF (run twice; first warms clocks/allocator)
    ditmod._CACHE_MOD = False; ditmod._CACHE_MASK_BIAS = False
    run(pipe, LQ, th, tw, F)
    v_off, dt_off = run(pipe, LQ, th, tw, F)

    results = []
    for name, mod, mb in [("CACHE_MOD", True, False),
                          ("CACHE_MASK_BIAS", False, True),
                          ("BOTH", True, True)]:
        ditmod._CACHE_MOD = mod; ditmod._CACHE_MASK_BIAS = mb
        v_on, dt_on = run(pipe, LQ, th, tw, F)
        maxd = (v_off - v_on).abs().max().item()
        results.append((name, dt_on, out / dt_on, maxd))

    print(f"\n=== Phase-B lossless cache parity + FPS @ {tw}x{th} ===")
    print(f"  baseline (OFF):       {dt_off:.3f}s  {out/dt_off:6.2f} FPS")
    ok = True
    for name, dt, fps, maxd in results:
        status = "OK (bit-identical)" if maxd == 0.0 else f"FAIL max|diff|={maxd:.3e}"
        ok = ok and (maxd == 0.0)
        print(f"  {name:16s} ON:  {dt:.3f}s  {fps:6.2f} FPS   {status}")
    print(f"\nRESULT: {'PASS (all bit-identical)' if ok else 'FAIL (non-zero diff)'}")


if __name__ == "__main__":
    main()
