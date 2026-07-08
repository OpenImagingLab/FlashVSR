#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase-2A parity harness: flag OFF vs ON on the same pipeline session.

For each requested Phase-2A flag, runs the v1.1 Tiny pipeline @768x1408 with
the flag OFF and ON (same seed/input, full-knob baseline config fixed) and
reports max|diff| / mean|diff| / PSNR. Gate type per flag:
  bit    -> requires max|diff| == 0 (lossless claims)
  psnr49 -> requires PSNR >= 49 dB (numerically-neutral claims)

Run from examples/WanVSR/ :
    python test_phase2a_lossless.py CACHE_ROPE_FREQS [FUSE_ROPE ...]
"""
import os, sys, time, math, importlib.util

import numpy as np
from PIL import Image
import imageio
import torch

os.environ["FLASHVSR_CONV3D_BACKEND"] = "gemm"
os.environ["FLASHVSR_TCDECODER_CHANNELS_LAST"] = "1"
os.environ["FLASHVSR_FUSE_NORM"] = "1"
os.environ["FLASHVSR_ATTN_BACKEND"] = "triton"
os.environ["FLASHVSR_CACHE_MOD"] = "1"
os.environ["FLASHVSR_CACHE_MASK_BIAS"] = "1"

import utils.utils as wanutils; wanutils._CONV3D_BACKEND = "gemm"
import diffsynth.models.wan_video_dit as ditmod
import diffsynth.pipelines.flashvsr_tiny as pipemod

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("infer_v1_1_tiny", os.path.join(_here, "infer_flashvsr_v1.1_tiny.py"))
_infer = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_infer)
init_pipeline = _infer.init_pipeline; largest_8n1_leq = _infer.largest_8n1_leq

REF_W, REF_H, SCALE = 768, 1408, 4
SRC_W, SRC_H = REF_W // SCALE, REF_H // SCALE

# flag name -> (module, attribute, gate)
FLAGS = {
    "CACHE_ROPE_FREQS": (pipemod, "_CACHE_ROPE_FREQS", "bit"),
    "FUSE_ROPE":        (ditmod,  "_FUSE_ROPE",        "psnr49"),
    "KV_RINGBUF":       (ditmod,  "_KV_RINGBUF",       "bit"),
    "ATTN_STRIDED_IO":  (ditmod,  "_ATTN_STRIDED_IO",  "psnr49"),
    "MASKGEN_LEAN":     (ditmod,  "_MASKGEN_LEAN",     "bit"),
    "LQPROJ_LEAN":      (wanutils, "_LQPROJ_LEAN",     "bit"),
}


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
    mse = (a - b).pow(2).mean().item()
    return float("inf") if mse <= 1e-12 else 10.0 * math.log10(4.0 / mse)


def main():
    names = [n for n in sys.argv[1:] if n in FLAGS]
    if not names:
        print(f"usage: {sys.argv[0]} FLAG [FLAG...]   FLAG in {list(FLAGS)}"); sys.exit(2)
    missing = [n for n in names if not hasattr(FLAGS[n][0], FLAGS[n][1])]
    if missing:
        print(f"FAIL: module attribute not found for {missing}"); sys.exit(2)

    pipe = init_pipeline()
    LQ, F = build_lq("./inputs/example0.mp4")
    th, tw = REF_H, REF_W
    out = F - 4

    def set_all(val_map):
        for n, (mod, attr, _gate) in FLAGS.items():
            if hasattr(mod, attr):
                setattr(mod, attr, val_map.get(n, False))

    # baseline: all Phase-2A flags OFF (run twice; first warms clocks/compile)
    set_all({})
    run(pipe, LQ, th, tw, F)
    v_off, dt_off = run(pipe, LQ, th, tw, F)

    print(f"\n=== Phase-2A parity @ {tw}x{th} F={F} ===")
    print(f"  baseline (all OFF): {dt_off:.3f}s  {out/dt_off:6.2f} FPS")
    ok = True
    for n in names:
        mod, attr, gate = FLAGS[n]
        set_all({n: True})
        run(pipe, LQ, th, tw, F)  # warm any lazy compile/cache for this flag
        v_on, dt_on = run(pipe, LQ, th, tw, F)
        d = (v_off - v_on).abs()
        maxd, meand, p = d.max().item(), d.mean().item(), psnr(v_off, v_on)
        good = (maxd == 0.0) if gate == "bit" else (p >= 49.0)
        ok = ok and good
        print(f"  {n:18s} ON: {dt_on:.3f}s  {out/dt_on:6.2f} FPS   "
              f"max|d|={maxd:.3e} mean|d|={meand:.3e} PSNR={p:.1f}dB  "
              f"[{'OK' if good else 'FAIL'} gate={gate}]")
    set_all({})
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
