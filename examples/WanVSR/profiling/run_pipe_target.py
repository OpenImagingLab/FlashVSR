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
import hashlib
import json
import os
import subprocess
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


def source_identity(src):
    path = os.path.realpath(src)
    stat = os.stat(path)
    raw = f"{path}|{stat.st_size}|{stat.st_mtime_ns}".encode()
    return path, hashlib.sha256(raw).hexdigest()[:16]


def build_lq(src, W, H, F, device="cuda", dtype=torch.bfloat16):
    """LQ tensor at target resolution (bicubic down to W/4 x H/4, then bicubic
    up to W x H), frames cycled if the source is shorter than F.
    Cached on disk: repeated profiling runs skip the CPU-side decode/resize."""
    cache_dir = os.path.join(_here, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    src_path, fingerprint = source_identity(src)
    key = f"{os.path.basename(src).replace('.', '_')}_{fingerprint}"
    cache_f = os.path.join(cache_dir, f"lq_{key}_{W}x{H}_f{F}.pt")
    if os.path.exists(cache_f):
        t0 = time.perf_counter()
        LQ = torch.load(cache_f, map_location="cpu", weights_only=True)
        LQ = LQ.to(device=device, dtype=dtype)
        return LQ, time.perf_counter() - t0, True, src_path, fingerprint

    t0 = time.perf_counter()
    rdr = imageio.get_reader(src_path)
    total = rdr.count_frames()
    idx = [i % total for i in range(F)]
    sw, sh = W // 4, H // 4
    frames = []
    for i in idx:
        img = Image.fromarray(rdr.get_data(i)).convert("RGB")
        img = img.resize((sw, sh), Image.BICUBIC).resize((W, H), Image.BICUBIC)
        t = torch.from_numpy(np.array(img, dtype=np.uint8, copy=True)).to(torch.float32)
        t = t.permute(2, 0, 1) / 255.0 * 2.0 - 1.0
        frames.append(t.to(torch.bfloat16))
    rdr.close()
    LQ = torch.stack(frames, 0).permute(1, 0, 2, 3).unsqueeze(0)  # 1 C F H W
    torch.save(LQ, cache_f)
    LQ = LQ.to(device=device, dtype=dtype)
    return LQ, time.perf_counter() - t0, False, src_path, fingerprint


def parse_slice(spec, size):
    start_s, stop_s = spec.split(":", 1)
    start = int(start_s) if start_s else 0
    stop = int(stop_s) if stop_s else size
    if stop < 0:
        stop = size + stop + 1
    return max(0, start), min(size, stop)


def strict_fastpath_errors(counts, errors, n_chunks, width, height):
    failures = []

    def exact(name, expected):
        actual = counts.get(name, 0)
        if actual != expected:
            failures.append(f"{name}: expected {expected}, got {actual}")

    attn_calls = 30 * n_chunks
    backend = os.environ.get("FLASHVSR_ATTN_BACKEND", "sparse").lower()
    zero_copy = env_int("FLASHVSR_ATTN_ZEROCOPY", 0) != 0
    fused_csr = env_int("FLASHVSR_FUSED_CSR", 0) != 0
    strided_io = env_int("FLASHVSR_ATTN_STRIDED_IO", 0) != 0
    if backend not in ("sparse", "triton", "triton2", "auto", "dense"):
        failures.append(f"unsupported FLASHVSR_ATTN_BACKEND={backend!r}")
    if zero_copy and backend != "triton2":
        failures.append("FLASHVSR_ATTN_ZEROCOPY requires ATTN_BACKEND=triton2")
    if fused_csr and backend != "triton2":
        failures.append("FLASHVSR_FUSED_CSR requires ATTN_BACKEND=triton2")
    if strided_io and backend not in ("triton", "triton2"):
        failures.append(
            "FLASHVSR_ATTN_STRIDED_IO requires ATTN_BACKEND=triton or triton2")
    if backend == "triton2" and zero_copy:
        exact("attn_zc_v2", attn_calls)
        exact("attn_v2", 0)
    elif backend == "triton2":
        exact("attn_v2", attn_calls)
    elif backend == "triton":
        if strided_io:
            exact("attn_v1_strided", attn_calls)
            exact("attn_v1_contiguous", 0)
        else:
            exact("attn_v1_contiguous", attn_calls)
            exact("attn_v1_strided", 0)
    elif backend == "sparse":
        exact("attn_sparse", attn_calls)
    elif backend == "dense":
        exact("attn_dense", attn_calls)
    elif backend == "auto":
        actual = counts.get("attn_dense", 0) + counts.get("attn_sparse", 0)
        if actual != attn_calls:
            failures.append(
                f"auto attention routes: expected {attn_calls}, got {actual}")

    rope_kernel = os.environ.get("FLASHVSR_ROPE_KERNEL", "").lower()
    if rope_kernel not in ("", "triton"):
        failures.append(f"unsupported FLASHVSR_ROPE_KERNEL={rope_kernel!r}")
    if rope_kernel == "triton":
        exact("rope_triton", 2 * attn_calls)
        exact("rope_fused", 0)
        exact("rope_eager", 0)
    elif env_int("FLASHVSR_FUSE_ROPE", 0):
        exact("rope_fused", 2 * attn_calls)
        exact("rope_triton", 0)
        exact("rope_eager", 0)
    else:
        exact("rope_eager", 2 * attn_calls)
        exact("rope_triton", 0)
        exact("rope_fused", 0)
    if backend == "triton2" and fused_csr:
        exact("csr_fused", attn_calls)
        exact("csr_argsort", 0)
    if env_int("FLASHVSR_POOLED_K_CACHE", 0):
        exact("pooled_k_rebuild", 30)
        exact("pooled_k_incremental", 30 * (n_chunks - 1))
    if env_int("FLASHVSR_DIT_ROW_FUSION", 0):
        exact("dit_row_ln_modulate", 2 * attn_calls)
        exact("dit_row_gate", 2 * attn_calls)
    if env_int("FLASHVSR_MASKGEN_THRESHOLD_CACHE", 0):
        # Chunk 0 (six-frame Q) and chunk 1 (first steady Q/KV geometry)
        # seed distinct thresholds. All later chunks reuse the steady value.
        seed_chunks = min(n_chunks, 2)
        exact("mask_threshold_kthvalue", 30 * seed_chunks)
        exact("mask_threshold_cached", 30 * (n_chunks - seed_chunks))
    conv_backend = os.environ.get("FLASHVSR_CONV3D_BACKEND", "auto").lower()
    lean_lq = env_int("FLASHVSR_LQPROJ_LEAN", 0) != 0
    packer = os.environ.get("FLASHVSR_CONV3D_PACKER", "eager").lower()
    conv_calls = 13 + 4 * (n_chunks - 1)
    if conv_backend not in ("auto", "gemm"):
        failures.append(f"unsupported FLASHVSR_CONV3D_BACKEND={conv_backend!r}")
    if lean_lq and conv_backend != "gemm":
        failures.append("FLASHVSR_LQPROJ_LEAN requires CONV3D_BACKEND=gemm")
    if packer not in ("eager", "triton"):
        failures.append(f"unsupported FLASHVSR_CONV3D_PACKER={packer!r}")
    if packer == "triton" and conv_backend != "gemm":
        failures.append("FLASHVSR_CONV3D_PACKER=triton requires CONV3D_BACKEND=gemm")
    if conv_backend == "gemm":
        if lean_lq:
            exact("conv3d_lean_gemm", conv_calls)
            exact("conv3d_gemm", 0)
        else:
            exact("conv3d_gemm", conv_calls)
            exact("conv3d_lean_gemm", 0)
        exact("conv3d_cudnn", 0)
        h_lq, w_lq = height // 16, width // 16
        budget = int(float(os.environ.get(
            "FLASHVSR_CONV3D_IM2COL_BUDGET_GB", "2.0")) * 1e9)
        bytes_per_row1 = 2 * w_lq * (768 * 4 * 3 * 3) * 2
        bytes_per_row2 = w_lq * (2048 * 4 * 3 * 3) * 2
        rows1 = h_lq if budget <= 0 else max(
            1, min(h_lq, budget // bytes_per_row1))
        rows2 = h_lq if budget <= 0 else max(
            1, min(h_lq, budget // bytes_per_row2))
        packs1 = (h_lq + rows1 - 1) // rows1
        packs2 = (h_lq + rows2 - 1) // rows2
        pack_calls = (2 * n_chunks + 5) * packs1 \
            + (2 * n_chunks + 4) * packs2
        if packer == "triton":
            exact("conv3d_packer_triton", pack_calls)
            exact("conv3d_packer_eager", 0)
        else:
            exact("conv3d_packer_eager", pack_calls)
            exact("conv3d_packer_triton", 0)
    elif conv_backend == "auto":
        exact("conv3d_cudnn", conv_calls)
        exact("conv3d_lean_gemm", 0)
        exact("conv3d_gemm", 0)

    overlap = env_int("FLASHVSR_DECODER_OVERLAP", 0) != 0
    direct_output = env_int("FLASHVSR_TCDECODER_DIRECT_OUTPUT", 0) != 0
    if direct_output and not overlap:
        failures.append("FLASHVSR_TCDECODER_DIRECT_OUTPUT requires DECODER_OVERLAP=1")
    decoder_triton = [
        name for name in (
            "FLASHVSR_TCDECODER_FUSE_POINTWISE",
            "FLASHVSR_TCDECODER_UPSAMPLE",
            "FLASHVSR_TCDECODER_CONCAT",
            "FLASHVSR_TCDECODER_TGROW_UP",
        ) if env_int(name, 0)
    ]
    if decoder_triton and not env_int("FLASHVSR_TCDECODER_CHANNELS_LAST", 0):
        failures.append(
            f"{', '.join(decoder_triton)} require TCDECODER_CHANNELS_LAST=1")
    if overlap:
        exact("decoder_overlap_chunks", n_chunks)
        exact("decoder_overlap_unavailable", 0)
        exact("decoder_serialized_calls", 0)
        if direct_output:
            exact("decoder_direct_output_chunks", n_chunks)
            exact("decoder_direct_output_complete", 1)
            exact("decoder_final_cat", 0)
        else:
            exact("decoder_final_cat", 1)
            exact("decoder_direct_output_chunks", 0)
            exact("decoder_direct_output_complete", 0)
    else:
        exact("decoder_serialized_calls", 1)
        exact("decoder_overlap_chunks", 0)
        exact("decoder_direct_output_chunks", 0)
        exact("decoder_direct_output_complete", 0)
    latent_frames = 2 * (n_chunks + 2)
    state_updates = 3 * (latent_frames - 1) + 3 * (latent_frames - 1) \
        + 3 * (2 * latent_frames - 1)
    if env_int("FLASHVSR_TCDECODER_POINTER_STATE", 0):
        exact("decoder_state_pointer_updates", state_updates)
        exact("decoder_state_copies", 0)
    else:
        exact("decoder_state_copies", state_updates)
        exact("decoder_state_pointer_updates", 0)
    cudnn_fused = env_int("FLASHVSR_TCDECODER_CUDNN_FUSED", 0) != 0
    splitk_conv = env_int("FLASHVSR_TCDECODER_SPLITK_CONV", 0) != 0
    if splitk_conv and not cudnn_fused:
        failures.append("FLASHVSR_TCDECODER_SPLITK_CONV requires CUDNN_FUSED=1")
    if cudnn_fused:
        exact("decoder_memblock_splitk" if splitk_conv else "decoder_memblock_cudnn",
              12 * latent_frames)
        if splitk_conv:
            exact("decoder_memblock_cudnn", 0)
        exact("decoder_memblock_fused", 0)
        exact("decoder_memblock_native", 0)
    elif env_int("FLASHVSR_TCDECODER_FUSE_POINTWISE", 0):
        exact("decoder_memblock_fused", 12 * latent_frames)
        exact("decoder_memblock_native", 0)
        exact("decoder_memblock_cudnn", 0)
    else:
        exact("decoder_memblock_native", 12 * latent_frames)
        exact("decoder_memblock_fused", 0)
        exact("decoder_memblock_cudnn", 0)
    if env_int("FLASHVSR_TCDECODER_TGROW_UP", 0):
        # The fused reorder consumes all three Upsample->TGrow pairs (four
        # site executions per latent frame); the standalone upsample and the
        # native TGrow routes must not fire at all.
        exact("decoder_tgrow_fused", 4 * latent_frames)
        exact("decoder_tgrow_native", 0)
        exact("decoder_upsample_triton", 0)
        exact("decoder_upsample_native", 0)
    else:
        exact("decoder_tgrow_native", 3 * latent_frames)
        exact("decoder_tgrow_fused", 0)
        if env_int("FLASHVSR_TCDECODER_UPSAMPLE", 0):
            exact("decoder_upsample_triton", 4 * latent_frames)
            exact("decoder_upsample_native", 0)
        else:
            exact("decoder_upsample_native", 4 * latent_frames)
            exact("decoder_upsample_triton", 0)
    if splitk_conv:
        exact("decoder_concat_triton", 0)
        exact("decoder_concat_native", 0)
    elif env_int("FLASHVSR_TCDECODER_CONCAT", 0):
        exact("decoder_concat_triton", 12 * latent_frames)
        exact("decoder_concat_native", 0)
    else:
        exact("decoder_concat_native", 12 * latent_frames)
        exact("decoder_concat_triton", 0)
    if cudnn_fused:
        exact("decoder_conv_relu_cudnn", 10 * latent_frames)
        exact("decoder_relu_native", 0)
    else:
        exact("decoder_relu_native", 10 * latent_frames)
        exact("decoder_conv_relu_cudnn", 0)
    exact("color_fix_success", 1)

    error_counts = {k: v for k, v in counts.items() if k.endswith("_error") and v}
    if error_counts:
        failures.append(f"fallback/error counters: {error_counts}")
    if errors:
        failures.append(f"recorded fast-path errors: {errors}")
    return failures


def main():
    require_fastpaths = env_int("FLASHVSR_REQUIRE_FASTPATHS", 0)
    if require_fastpaths:
        os.environ["FLASHVSR_TELEMETRY"] = "1"
    W = env_int("FLASHVSR_PROF_W", 768)
    H = env_int("FLASHVSR_PROF_H", 1408)
    F = largest_8n1_leq(env_int("FLASHVSR_PROF_FRAMES", 85))
    src = os.environ.get("FLASHVSR_PROF_INPUT", "./inputs/example0.mp4")
    warmup = env_int("FLASHVSR_PROF_WARMUP", 1)
    steady = os.environ.get("FLASHVSR_PROF_STEADY", "2:7")
    bench_steady = os.environ.get("FLASHVSR_BENCH_STEADY", "2:7")
    save = env_int("FLASHVSR_PROF_SAVE", 0)
    assert W % 128 == 0 and H % 128 == 0, "W/H must be multiples of 128"

    n_chunks = (F - 1) // 8 - 2
    knobs = {k: v for k, v in sorted(os.environ.items()) if k.startswith("FLASHVSR")}
    print(f"[target] {W}x{H} F={F} chunks={n_chunks} steady={steady} "
          f"bench_steady={bench_steady} warmup={warmup}")
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

    from diffsynth import perf_stats

    LQ, t_lq, lq_cached, src_path, src_fingerprint = build_lq(src, W, H, F)
    print(f"[target] init {t_init:.1f}s  lq_build {t_lq:.1f}s (cached={lq_cached}) "
          f"input={src_path} fingerprint={src_fingerprint}")

    chunk_times = []
    loop_meta = {}

    def timed_bar(iterable):
        """tqdm replacement that records per-chunk wall time (no added syncs;
        in steady state per-chunk wall == pipeline throughput per chunk)."""
        def gen():
            prev = time.perf_counter()
            loop_meta["loop_start"] = prev
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
        perf_stats.reset(preserve_errors=True)
    else:
        perf_stats.reset()

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
    peak_reserved = torch.cuda.max_memory_reserved() / 2**30
    print(f"[target] measured: wall {wall*1e3:.0f} ms  fps {fps:.2f}  "
           f"px-norm-fps {fps*(H*W)/(768*1408):.2f}  peak_mem {peak:.1f} GiB  "
           f"peak_reserved {peak_reserved:.1f} GiB")
    ct = [round(t * 1e3, 1) for t in chunk_times]
    b_start, b_stop = parse_slice(bench_steady, len(ct))
    steady_ct = ct[b_start:b_stop]
    steady_ms = round(float(np.mean(steady_ct)), 2) if steady_ct else -1.0
    steady_median = round(float(np.median(steady_ct)), 2) if steady_ct else -1.0
    steady_p95 = round(float(np.percentile(steady_ct, 95)), 2) if steady_ct else -1.0
    steady_std = round(float(np.std(steady_ct)), 2) if steady_ct else -1.0
    steady_max = round(float(np.max(steady_ct)), 2) if steady_ct else -1.0
    print(f"[chunks] per-chunk ms: {ct}  range={b_start}:{b_stop} "
          f"mean={steady_ms} median={steady_median} p95={steady_p95} "
          f"max={steady_max} std={steady_std}")
    # Post-loop tail: time between the end of the last chunk body and the end
    # of the measured call (serialized decode + color fix, or overlap-mode
    # decode remainder + color fix). CPU wall, includes the final sync.
    tail_ms = round((t0 + wall - loop_meta["loop_end"]) * 1e3, 1) if "loop_end" in loop_meta else -1.0
    loop_ms = round((loop_meta["loop_end"] - loop_meta["loop_start"]) * 1e3, 1) \
        if "loop_end" in loop_meta and "loop_start" in loop_meta else -1.0
    print(f"[tail] post_loop_ms: {tail_ms}")
    telemetry = perf_stats.snapshot()
    print(f"[backends] {json.dumps(telemetry['counts'])}")
    print(f"[fallbacks] {json.dumps(telemetry['errors'])}")
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=os.path.dirname(_wanvsr),
            check=True, capture_output=True, text=True).stdout.strip()
    except Exception:
        commit = "unknown"
    result = dict(
        W=W, H=H, F=F, output_frames=int(frames.shape[1]), n_chunks=n_chunks,
        output_shape=list(frames.shape), output_dtype=str(frames.dtype),
        wall_s=round(wall, 3), fps=round(fps, 3),
        px_norm_fps=round(fps * (H * W) / (768 * 1408), 3),
        loop_ms=loop_ms, tail_ms=tail_ms, chunk_ms=ct,
        steady_range=[b_start, b_stop], steady_chunk_ms=steady_ms,
        steady_median_ms=steady_median, steady_p95_ms=steady_p95,
        steady_max_ms=steady_max, steady_std_ms=steady_std,
        peak_gib=round(peak, 2), peak_reserved_gib=round(peak_reserved, 2),
        commit=commit, gpu=torch.cuda.get_device_name(),
        capability=list(torch.cuda.get_device_capability()), torch=torch.__version__,
        cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
        input=src_path, input_fingerprint=src_fingerprint,
    )
    print(f"[result] {json.dumps(result)}")

    if require_fastpaths:
        failures = strict_fastpath_errors(
            telemetry["counts"], telemetry["errors"], n_chunks, W, H)
        if failures:
            for failure in failures:
                print(f"[strict] FAIL: {failure}")
            raise SystemExit(1)
        print("[strict] PASS: all requested fast paths executed without fallback")

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
