#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Profiling target for FlashVSR v1.1 Tiny (Nsight Systems / Nsight Compute).

Runs the pipeline once (optional warmup first) with an env-driven workload and
marks a steady-state chunk window with cudaProfilerStart/Stop so that
`nsys --capture-range=cudaProfilerApi` / `ncu --profile-from-start off` capture
exactly the chunks we want (skipping warmup, model load and the atypical
chunk 0 / chunk 1).

Env (workload):
  FLASHVSR_PROF_W        output width  (default 768, multiple of 128)
  FLASHVSR_PROF_H        output height (default 1408, multiple of 128)
  FLASHVSR_PROF_FRAMES   frame count F, 8n+1 (default 85 -> 8 chunks)
  FLASHVSR_PROF_INPUT    source clip (default ./inputs/example0.mp4)
  FLASHVSR_PROF_WARMUP   1 = full-pipe warmup call before measured run (default 1)
  FLASHVSR_PROF_STEADY   "start:stop" chunk window for cudaProfilerStart/Stop
                         (default "2:7"; "0:-1" = capture whole measured call;
                         "off" = never call cudaProfiler*)
  FLASHVSR_PROF_SAVE     1 = save output mp4 next to the reports (default 0)

Perf knobs (FLASHVSR_CONV3D_BACKEND, FLASHVSR_ATTN_BACKEND, FLASHVSR_FUSE_NORM,
FLASHVSR_TCDECODER_CHANNELS_LAST, FLASHVSR_CACHE_MOD, FLASHVSR_CACHE_MASK_BIAS,
FLASHVSR_NVTX, ...) must be set by the caller BEFORE python starts (they are
read at import time by the respective modules).

Run from anywhere; the script chdirs to examples/WanVSR.
"""
import importlib.util
import json
import os
import sys
import time

_here = os.path.dirname(os.path.abspath(__file__))
_wanvsr = os.path.dirname(_here)
os.chdir(_wanvsr)
sys.path.insert(0, _wanvsr)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
import imageio  # noqa: E402


def largest_8n1_leq(n: int) -> int:
    return 0 if n < 1 else ((n - 1) // 8) * 8 + 1


def env_int(name, default):
    return int(os.environ.get(name, str(default)))


def build_lq(src, W, H, F, device="cuda", dtype=torch.bfloat16):
    """LQ tensor at target resolution (bicubic down to W/4 x H/4, then bicubic
    up to W x H), frames cycled if the source is shorter than F.
    Cached on disk: repeated profiling runs skip the CPU-side decode/resize."""
    cache_dir = os.path.join(_here, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    key = os.path.basename(src).replace(".", "_")
    cache_f = os.path.join(cache_dir, f"lq_{key}_{W}x{H}_f{F}.pt")
    if os.path.exists(cache_f):
        t0 = time.perf_counter()
        LQ = torch.load(cache_f, map_location="cpu", weights_only=True)
        LQ = LQ.to(device=device, dtype=dtype)
        return LQ, time.perf_counter() - t0, True

    t0 = time.perf_counter()
    rdr = imageio.get_reader(src)
    total = rdr.count_frames()
    idx = [i % total for i in range(F)]
    sw, sh = W // 4, H // 4
    frames = []
    for i in idx:
        img = Image.fromarray(rdr.get_data(i)).convert("RGB")
        img = img.resize((sw, sh), Image.BICUBIC).resize((W, H), Image.BICUBIC)
        t = torch.from_numpy(np.asarray(img, np.uint8)).to(torch.float32)
        t = t.permute(2, 0, 1) / 255.0 * 2.0 - 1.0
        frames.append(t.to(torch.bfloat16))
    rdr.close()
    LQ = torch.stack(frames, 0).permute(1, 0, 2, 3).unsqueeze(0)  # 1 C F H W
    torch.save(LQ, cache_f)
    LQ = LQ.to(device=device, dtype=dtype)
    return LQ, time.perf_counter() - t0, False


def main():
    W = env_int("FLASHVSR_PROF_W", 768)
    H = env_int("FLASHVSR_PROF_H", 1408)
    F = largest_8n1_leq(env_int("FLASHVSR_PROF_FRAMES", 85))
    src = os.environ.get("FLASHVSR_PROF_INPUT", "./inputs/example0.mp4")
    warmup = env_int("FLASHVSR_PROF_WARMUP", 1)
    steady = os.environ.get("FLASHVSR_PROF_STEADY", "2:7")
    save = env_int("FLASHVSR_PROF_SAVE", 0)
    assert W % 128 == 0 and H % 128 == 0, "W/H must be multiples of 128"

    n_chunks = (F - 1) // 8 - 2
    knobs = {k: v for k, v in sorted(os.environ.items()) if k.startswith("FLASHVSR")}
    print(f"[target] {W}x{H} F={F} chunks={n_chunks} steady={steady} warmup={warmup}")
    print(f"[target] knobs: {json.dumps(knobs)}")

    # Profiler window must stay OFF during pipeline init + warmup.
    os.environ.pop("FLASHVSR_PROFILER_START_CHUNK", None)
    os.environ.pop("FLASHVSR_PROFILER_STOP_CHUNK", None)

    t0 = time.perf_counter()
    spec = importlib.util.spec_from_file_location(
        "infer_v1_1_tiny", os.path.join(_wanvsr, "infer_flashvsr_v1.1_tiny.py"))
    _infer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_infer)
    pipe = _infer.init_pipeline()
    t_init = time.perf_counter() - t0

    LQ, t_lq, lq_cached = build_lq(src, W, H, F)
    print(f"[target] init {t_init:.1f}s  lq_build {t_lq:.1f}s (cached={lq_cached})")

    chunk_times = []
    loop_meta = {}

    def timed_bar(iterable):
        """tqdm replacement that records per-chunk wall time (no added syncs;
        in steady state per-chunk wall == pipeline throughput per chunk)."""
        def gen():
            prev = time.perf_counter()
            for it in iterable:
                yield it
                now = time.perf_counter()
                chunk_times.append(now - prev)
                prev = now
            # End of chunk loop: everything after this is the post-loop tail
            # (decode [serialized mode], decode-wait remainder [overlap mode],
            # color fix, return). Used for decode-tail attribution in the log.
            loop_meta["loop_end"] = time.perf_counter()
        return gen()

    kwargs = dict(
        prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=1,
        seed=0, LQ_video=LQ, num_frames=F, height=H, width=W,
        is_full_block=False, if_buffer=True,
        topk_ratio=2.0 * 768 * 1280 / (H * W), kv_ratio=3.0, local_range=11,
        color_fix=True, progress_bar_cmd=timed_bar)

    if warmup:
        t0 = time.perf_counter()
        with torch.no_grad():
            pipe(**kwargs)
        torch.cuda.synchronize()
        print(f"[target] warmup done in {time.perf_counter()-t0:.1f}s")
        chunk_times.clear()
        loop_meta.clear()

    # Arm the steady-state cudaProfiler window for the measured call.
    if steady != "off":
        s_start, s_stop = steady.split(":")
        os.environ["FLASHVSR_PROFILER_START_CHUNK"] = s_start
        os.environ["FLASHVSR_PROFILER_STOP_CHUNK"] = s_stop

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        frames = pipe(**kwargs)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0

    fps = (F - 4) / wall
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"[target] measured: wall {wall*1e3:.0f} ms  fps {fps:.2f}  "
          f"px-norm-fps {fps*(H*W)/(768*1408):.2f}  peak_mem {peak:.1f} GiB")
    ct = [round(t * 1e3, 1) for t in chunk_times]
    steady_ct = ct[2:7] if len(ct) >= 7 else ct[1:]
    steady_ms = round(sum(steady_ct) / max(len(steady_ct), 1), 2)
    print(f"[chunks] per-chunk ms: {ct}  steady_avg_ms: {steady_ms}")
    # Post-loop tail: time between the end of the last chunk body and the end
    # of the measured call (serialized decode + color fix, or overlap-mode
    # decode remainder + color fix). CPU wall, includes the final sync.
    tail_ms = round((t0 + wall - loop_meta["loop_end"]) * 1e3, 1) if "loop_end" in loop_meta else -1.0
    print(f"[tail] post_loop_ms: {tail_ms}")
    print(f"[result] {json.dumps(dict(W=W, H=H, F=F, wall_s=round(wall,3), fps=round(fps,3), peak_gib=round(peak,2), steady_chunk_ms=steady_ms, tail_ms=tail_ms))}")

    if save:
        out = frames.float().clamp(-1, 1).add(1).div(2).mul(255).byte()
        out = out.permute(1, 2, 3, 0).cpu().numpy()  # F H W C
        path = os.path.join(_here, f"out_{W}x{H}_f{F}.mp4")
        w = imageio.get_writer(path, fps=30, quality=6)
        for fr in out:
            w.append_data(fr)
        w.close()
        print(f"[target] saved {path}")


if __name__ == "__main__":
    main()
