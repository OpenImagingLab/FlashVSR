#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 2B-1 parity harness: serialized decode vs decoder overlap.

Compares the Phase-2A serialized decoder path (FLASHVSR_DECODER_OVERLAP=0)
against the Phase-2B-1 side-stream overlap path (=1) on the same pipeline
session, with the full Phase-2A recommended stack enabled on BOTH sides — the
only delta under test is the overlap flag. Gate: bit-identical
(max|diff| == 0).

Coverage:
  * full clip (example0, F=81 -> 8 chunks): exercises steady-state streaming,
    chunk-0 trim, per-chunk cond slicing, multi-chunk mem-block state;
  * short clip (same input, F=25 -> 1 chunk): exercises the single-chunk /
    trim-only edge case;
  * overlap path run 3x back-to-back: concurrency race detection (all repeats
    must be bit-identical);
  * shape / frame count / dtype / device / per-frame ordering equality.

Run from anywhere:
    /root/FlashVSR/venv/bin/python profiling/test_decoder_overlap_lossless.py
"""
import os, sys, time, importlib.util

_here = os.path.dirname(os.path.abspath(__file__))
_wanvsr = os.path.dirname(_here)
os.chdir(_wanvsr)
sys.path.insert(0, _wanvsr)

# Full-knob baseline + complete Phase-2A recommended stack (read at import
# time by the respective modules). Overlap flag itself is toggled at runtime
# via the module attribute so both paths share one pipeline session.
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
os.environ["FLASHVSR_DECODER_OVERLAP"] = "0"

import numpy as np
from PIL import Image
import imageio
import torch

import diffsynth.pipelines.flashvsr_tiny as pipemod

_spec = importlib.util.spec_from_file_location(
    "infer_v1_1_tiny", os.path.join(_wanvsr, "infer_flashvsr_v1.1_tiny.py"))
_infer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_infer)
init_pipeline = _infer.init_pipeline
largest_8n1_leq = _infer.largest_8n1_leq

REF_W, REF_H, SCALE = 768, 1408, 4
SRC_W, SRC_H = REF_W // SCALE, REF_H // SCALE
F_SHORT = 25  # -> process_total_num = 1 (single chunk, trim edge case)


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
    dt = time.perf_counter() - t0
    meta = dict(shape=tuple(vid.shape), dtype=str(vid.dtype), device=str(vid.device.type))
    return vid.float().cpu(), meta, dt


def compare(tag, v_ref, m_ref, v_new, m_new):
    """Full comparison report; returns True iff bit-identical + metadata equal."""
    shape_ok = m_ref["shape"] == m_new["shape"]
    dtype_ok = m_ref["dtype"] == m_new["dtype"]
    dev_ok = m_ref["device"] == m_new["device"]
    if not shape_ok:
        print(f"  {tag}: FAIL shape {m_ref['shape']} vs {m_new['shape']}")
        return False
    d = (v_ref - v_new).abs()
    maxd, meand = d.max().item(), d.mean().item()
    # per-frame equality vector (output is (C, T, H, W)) -> ordering evidence
    T = v_ref.shape[1]
    frame_eq = (v_ref == v_new).all(dim=0).flatten(1).all(dim=1)  # (T,)
    n_eq = int(frame_eq.sum())
    ok = (maxd == 0.0) and shape_ok and dtype_ok and dev_ok and (n_eq == T)
    print(f"  {tag}: max|d|={maxd:.3e} mean|d|={meand:.3e} "
          f"frames_equal={n_eq}/{T} shape={m_new['shape']} "
          f"dtype={m_new['dtype']}({'ok' if dtype_ok else 'MISMATCH'}) "
          f"device={m_new['device']}({'ok' if dev_ok else 'MISMATCH'}) "
          f"[{'OK' if ok else 'FAIL'}]")
    return ok


def main():
    pipe = init_pipeline()
    LQ, F_full = build_lq("./inputs/example0.mp4")
    print(f"\n=== Phase 2B-1 decoder-overlap parity @ {REF_W}x{REF_H} "
          f"(full F={F_full}, short F={F_SHORT}) ===")

    # ---- serialized reference (overlap OFF) ----
    pipemod._DECODER_OVERLAP = False
    run(pipe, LQ, F_full)  # warmup: clocks, lazy compiles, cuDNN algo cache
    v_off_full, m_off_full, dt = run(pipe, LQ, F_full)
    print(f"  serialized  full : {dt:.3f}s  {(F_full-4)/dt:6.2f} FPS")
    v_off_short, m_off_short, dt = run(pipe, LQ, F_SHORT)
    print(f"  serialized  short: {dt:.3f}s")

    # ---- overlap path (3 repeats on the full clip for race detection) ----
    pipemod._DECODER_OVERLAP = True
    v_on_full, m_on_full = [], []
    for r in range(3):
        v, m, dt = run(pipe, LQ, F_full)
        v_on_full.append(v); m_on_full.append(m)
        print(f"  overlap     full (rep {r+1}): {dt:.3f}s  {(F_full-4)/dt:6.2f} FPS")
    v_on_short, m_on_short, dt = run(pipe, LQ, F_SHORT)
    print(f"  overlap     short: {dt:.3f}s")
    pipemod._DECODER_OVERLAP = False

    print("\n-- serialized vs overlap --")
    ok = True
    for r in range(3):
        ok &= compare(f"full rep{r+1} vs serialized", v_off_full, m_off_full,
                      v_on_full[r], m_on_full[r])
    ok &= compare("short vs serialized      ", v_off_short, m_off_short,
                  v_on_short, m_on_short)

    print("\n-- overlap repeat stability (race detection) --")
    for r in range(1, 3):
        ok &= compare(f"overlap rep{r+1} vs rep1     ", v_on_full[0], m_on_full[0],
                      v_on_full[r], m_on_full[r])

    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
