#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 2B-1 parity harness: decoder-overlap OFF vs ON on the same session.

Runs the v1.1 Tiny pipeline with the FULL Phase-2A recommended stack enabled
(FUSE_ROPE + KV_RINGBUF + ATTN_STRIDED_IO + MASKGEN_LEAN + LQPROJ_LEAN, all
ON) - this is the Phase 2B-1 baseline per PHASE_ROADMAP.md / the "final Phase
2A cumulative configuration" required by the Phase 2B-1 task, NOT the bare
full-knobs baseline. FLASHVSR_DECODER_OVERLAP is then compared OFF
(serialized end-of-loop decode, unchanged code path) vs ON (per-chunk decode
on a side CUDA stream, Phase 2B-1).

For each of a SHORT clip (1 chunk - degenerate case, chunk0's decode has no
later chunk to overlap with) and the FULL profiling-default clip (8 chunks),
reports:
  - max|diff|, mean|diff| (OFF vs ON, and ON vs repeated ON runs)
  - shape / frame-count / dtype / device equality
  - per-frame max|diff| (explicit output-ordering proof - a frame swap would
    spike an individual frame's diff even when the aggregate looks small)
  - repeated-ON-run bit-stability (race-condition detection: 3 ON runs must
    be pairwise identical)

Target: max|diff| == 0 in every comparison. Any non-zero difference is a
FAIL - this is a lossless-scheduling change, not a numerics change.

Run from examples/WanVSR/:
    python profiling/test_decoder_overlap_lossless.py
"""
import os, sys, time, importlib.util

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

_here = os.path.dirname(os.path.abspath(__file__))
_wanvsr = os.path.dirname(_here)
os.chdir(_wanvsr)
sys.path.insert(0, _wanvsr)

import utils.utils as wanutils; wanutils._CONV3D_BACKEND = "gemm"
import diffsynth.models.wan_video_dit as ditmod
import diffsynth.pipelines.flashvsr_tiny as pipemod

_spec = importlib.util.spec_from_file_location(
    "infer_v1_1_tiny", os.path.join(_wanvsr, "infer_flashvsr_v1.1_tiny.py"))
_infer = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_infer)
init_pipeline = _infer.init_pipeline; largest_8n1_leq = _infer.largest_8n1_leq

# @768x1408 is the mandatory primary resolution; REF_W/REF_H are env-
# overridable so the same harness can run the @1536x2560 closure spot-check
# (FLASHVSR_TEST_W=1536 FLASHVSR_TEST_H=2560) without duplicating the file.
REF_W = int(os.environ.get("FLASHVSR_TEST_W", "768"))
REF_H = int(os.environ.get("FLASHVSR_TEST_H", "1408"))
SCALE = 4
SRC_W, SRC_H = REF_W // SCALE, REF_H // SCALE


def _enable_phase2a_kept_stack():
    """Fixed Phase-2A 'kept' stack this task's flag is layered on top of.

    Not toggled by this test (only FLASHVSR_DECODER_OVERLAP is toggled) -
    per task instructions, Phase 2B-1 must be validated against the final
    Phase 2A cumulative configuration, not the bare full-knobs baseline.
    """
    ditmod._FUSE_ROPE = True
    ditmod._KV_RINGBUF = True
    ditmod._ATTN_STRIDED_IO = True
    ditmod._MASKGEN_LEAN = True
    wanutils._LQPROJ_LEAN = True


def build_lq(src, F_req, device="cuda", dtype=torch.bfloat16):
    rdr = imageio.get_reader(src); total = rdr.count_frames()
    idx = list(range(total)) + [total - 1] * 4
    F = largest_8n1_leq(min(F_req, len(idx)))
    idx = idx[:F]
    frames = []
    for i in idx:
        img = (Image.fromarray(rdr.get_data(i)).convert("RGB")
               .resize((SRC_W, SRC_H), Image.BICUBIC).resize((REF_W, REF_H), Image.BICUBIC))
        t = torch.from_numpy(np.asarray(img, np.uint8)).to(device=device, dtype=torch.float32)
        frames.append((t.permute(2, 0, 1) / 255.0 * 2.0 - 1.0).to(dtype))
    rdr.close()
    return torch.stack(frames, 0).permute(1, 0, 2, 3).unsqueeze(0), F


def run(pipe, LQ, th, tw, F):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        vid = pipe(prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=1, seed=0,
                   LQ_video=LQ, num_frames=F, height=th, width=tw, is_full_block=False, if_buffer=True,
                   topk_ratio=2.0 * 768 * 1280 / (th * tw), kv_ratio=3.0, local_range=11, color_fix=True)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    meta = dict(shape=tuple(vid.shape), dtype=vid.dtype, device=str(vid.device))
    return vid.float().cpu(), dt, meta


def compare(name_a, a, meta_a, name_b, b, meta_b):
    shape_eq = tuple(a.shape) == tuple(b.shape)
    dtype_eq = meta_a["dtype"] == meta_b["dtype"]
    device_eq = meta_a["device"] == meta_b["device"]
    frames_eq = shape_eq and a.shape[1] == b.shape[1]  # (C, T, H, W) -> dim1 = T
    if not shape_eq:
        print(f"    {name_a} vs {name_b}: SHAPE MISMATCH {meta_a['shape']} vs {meta_b['shape']}  [FAIL]")
        return False
    d = (a - b).abs()
    maxd, meand = d.max().item(), d.mean().item()
    per_frame_maxd = [float(d[:, t].max()) for t in range(d.shape[1])]
    order_ok = max(per_frame_maxd, default=0.0) == 0.0
    ok = frames_eq and dtype_eq and device_eq and (maxd == 0.0) and order_ok
    print(f"    {name_a} vs {name_b}: max|d|={maxd:.6e} mean|d|={meand:.6e} "
          f"frames_eq={frames_eq} dtype_eq={dtype_eq} device_eq={device_eq} "
          f"order_ok={order_ok}  [{'OK' if ok else 'FAIL'}]")
    return ok


def one_config(pipe, tag, F_req):
    LQ, F = build_lq("./inputs/example0.mp4", F_req)
    th, tw = REF_H, REF_W
    n_chunks = (F - 1) // 8 - 2
    print(f"\n=== {tag}: F={F} ({n_chunks} chunks) ===")

    _enable_phase2a_kept_stack()
    pipemod._DECODER_OVERLAP = False
    run(pipe, LQ, th, tw, F)  # warmup
    v_off, dt_off, m_off = run(pipe, LQ, th, tw, F)
    print(f"  OFF (serialized): {dt_off:.3f}s  shape={m_off['shape']} "
          f"dtype={m_off['dtype']} device={m_off['device']}")

    pipemod._DECODER_OVERLAP = True
    run(pipe, LQ, th, tw, F)  # warmup: creates the side stream, primes lazy state
    v_on1, dt_on1, m_on1 = run(pipe, LQ, th, tw, F)
    v_on2, dt_on2, m_on2 = run(pipe, LQ, th, tw, F)
    v_on3, dt_on3, m_on3 = run(pipe, LQ, th, tw, F)
    print(f"  ON  (overlap x3): {dt_on1:.3f}s / {dt_on2:.3f}s / {dt_on3:.3f}s  "
          f"shape={m_on1['shape']} dtype={m_on1['dtype']} device={m_on1['device']}")

    ok = True
    ok &= compare("OFF", v_off, m_off, "ON#1", v_on1, m_on1)
    ok &= compare("ON#1", v_on1, m_on1, "ON#2", v_on2, m_on2)
    ok &= compare("ON#1", v_on1, m_on1, "ON#3", v_on3, m_on3)
    pipemod._DECODER_OVERLAP = False
    return ok


def main():
    # FLASHVSR_TEST_SHORT_ONLY=1 -> just the 1-chunk edge case (used for the
    # quick @1536x2560 closure spot-check; the full 8-chunk clip is
    # comparatively expensive at high resolution and the scheduling logic
    # under test has no resolution-dependent branching, so the short clip is
    # sufficient due diligence there).
    short_only = os.environ.get("FLASHVSR_TEST_SHORT_ONLY", "0") != "0"
    pipe = init_pipeline()
    ok = True
    ok &= one_config(pipe, "SHORT clip (1 chunk, chunk0-only decode, no overlap partner)", 25)
    if not short_only:
        ok &= one_config(pipe, "FULL clip (matches profiling default, 8 chunks)", 85)
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
