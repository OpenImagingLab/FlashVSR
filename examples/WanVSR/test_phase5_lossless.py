#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase-5 lossless regression harness for FlashVSR v1.1 Tiny on GH200.

Modes:
  e2e      Phase-3.5 base vs the Phase-5 P1-P3 stack at F=89.
  overlap  Serialized vs decoder overlap at F=25 and F=89; overlap runs 3x.
  pointer  Copy-based vs pointer-based recurrent decoder state at F=25/F=89.
  direct   Chunk cat vs direct final-output placement at F=25/F=29/F=89.
  pointwise Native vs fused MemBlock pointwise operations at F=25/F=89.
  upsample Native vs Triton channels-last nearest upsample at F=25/F=89.
  lqpacker Eager vs Triton LQ Conv3D patch packing at F=25/F=89.
  concat   Native vs Triton recurrent MemBlock concat at F=25/F=89.
  tgrowup  Native vs fused TGrow->upsample reorder at F=25/F=29/F=89
           (quality budget: bitwise preferred, >=49 dB PSNR gate).
  cudnnfuse Separate epilogues vs cuDNN runtime-fused decoder convs at
           F=25/F=89 (quality-gated: >=49 dB PSNR, not bit-exact).
  phase6   Original Phase-5 production stack vs all new lossless paths.
  all      Run both modes (default).

Run from anywhere:
    /root/FlashVSR/venv/bin/python examples/WanVSR/test_phase5_lossless.py [mode]
"""
import importlib.util
import os
import sys
import time


_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(os.path.dirname(_here))
os.chdir(_here)
sys.path.insert(0, _root)
sys.path.insert(0, _here)

# Every import-time knob is explicit. P1-P3 and decoder overlap are toggled
# through their module attributes below so all comparisons share one pipeline.
os.environ.update({
    "FLASHVSR_CONV3D_BACKEND": "gemm",
    "FLASHVSR_CONV3D_PACKER": "eager",
    "FLASHVSR_TCDECODER_CHANNELS_LAST": "1",
    "FLASHVSR_TCDECODER_POINTER_STATE": "0",
    "FLASHVSR_TCDECODER_DIRECT_OUTPUT": "0",
    "FLASHVSR_TCDECODER_FUSE_POINTWISE": "0",
    "FLASHVSR_TCDECODER_UPSAMPLE": "0",
    "FLASHVSR_TCDECODER_CONCAT": "0",
    "FLASHVSR_FUSE_NORM": "1",
    "FLASHVSR_ATTN_BACKEND": "triton2",
    "FLASHVSR_CACHE_MOD": "1",
    "FLASHVSR_CACHE_MASK_BIAS": "1",
    "FLASHVSR_CACHE_ROPE_FREQS": "0",
    "FLASHVSR_FUSE_ROPE": "1",
    "FLASHVSR_KV_RINGBUF": "1",
    "FLASHVSR_ATTN_STRIDED_IO": "1",
    "FLASHVSR_MASKGEN_LEAN": "1",
    "FLASHVSR_LQPROJ_LEAN": "1",
    "FLASHVSR_FUSED_CSR": "1",
    "FLASHVSR_ROPE_KERNEL": "triton",
    "FLASHVSR_POOLED_K_CACHE": "1",
    "FLASHVSR_ATTN_ZEROCOPY": "1",
    "FLASHVSR_FP8_GEMM": "0",
    "FLASHVSR_DECODER_OVERLAP": "0",
    "FLASHVSR_TELEMETRY": "1",
})

import imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from diffsynth import perf_stats  # noqa: E402
import diffsynth.models.fp8_gemm as fp8mod  # noqa: E402
import diffsynth.models.triton_block_sparse_attn_v2 as v2mod  # noqa: E402
import diffsynth.models.wan_video_dit as ditmod  # noqa: E402
import diffsynth.pipelines.flashvsr_tiny as pipemod  # noqa: E402
from utils import TCDecoder as tcdecmod  # noqa: E402
import utils.utils as wanutils  # noqa: E402


_spec = importlib.util.spec_from_file_location(
    "infer_v1_1_tiny", os.path.join(_here, "infer_flashvsr_v1.1_tiny.py"))
_infer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_infer)
init_pipeline = _infer.init_pipeline

REF_W = int(os.environ.get("FLASHVSR_TEST_W", "768"))
REF_H = int(os.environ.get("FLASHVSR_TEST_H", "1408"))
SCALE = 4
SRC_W, SRC_H = REF_W // SCALE, REF_H // SCALE
F_SHORT = 25
F_EDGE = 29
F_LONG = int(os.environ.get("FLASHVSR_TEST_FRAMES", "89"))
if F_LONG < F_EDGE or F_LONG % 8 != 1:
    raise ValueError("FLASHVSR_TEST_FRAMES must be >=29 and satisfy F % 8 == 1")
TINY_BLOCKS = 30

_ROUTE_KEYS = (
    "attn_zc_v2", "attn_v2", "attn_v1_strided", "attn_v1_contiguous",
    "attn_sparse", "attn_dense", "rope_triton", "rope_fused",
    "rope_eager", "csr_fused", "csr_argsort", "pooled_k_rebuild",
    "pooled_k_incremental", "conv3d_lean_gemm", "conv3d_gemm",
    "conv3d_cudnn", "conv3d_packer_eager", "conv3d_packer_triton",
    "decoder_overlap_chunks", "decoder_final_cat",
    "decoder_direct_output_chunks", "decoder_direct_output_complete",
    "decoder_state_copies", "decoder_state_pointer_updates",
    "decoder_memblock_native", "decoder_memblock_fused",
    "decoder_memblock_cudnn", "decoder_conv_relu_cudnn",
    "decoder_upsample_native", "decoder_upsample_triton",
    "decoder_concat_native", "decoder_concat_triton",
    "decoder_tgrow_native", "decoder_tgrow_fused", "decoder_relu_native",
    "decoder_overlap_unavailable", "decoder_serialized_calls",
    "color_fix_success",
)


def build_lq(src, frame_count, device="cuda", dtype=torch.bfloat16):
    """Build exactly frame_count frames, cycling the source when necessary."""
    rdr = imageio.get_reader(src)
    try:
        total = rdr.count_frames()
        if total <= 0:
            raise RuntimeError(f"input has no frames: {src}")
        frames = []
        for i in range(frame_count):
            img = Image.fromarray(rdr.get_data(i % total)).convert("RGB")
            img = img.resize((SRC_W, SRC_H), Image.BICUBIC)
            img = img.resize((REF_W, REF_H), Image.BICUBIC)
            tensor = torch.from_numpy(np.array(img, dtype=np.uint8, copy=True)).to(
                device=device, dtype=torch.float32)
            frames.append(
                (tensor.permute(2, 0, 1) / 255.0 * 2.0 - 1.0).to(dtype))
    finally:
        rdr.close()
    lq = torch.stack(frames, 0).permute(1, 0, 2, 3).unsqueeze(0)
    assert lq.shape[2] == frame_count
    return lq


def set_phase5(enabled):
    ditmod._ROPE_KERNEL = "triton" if enabled else ""
    ditmod._POOLED_K_CACHE = enabled
    ditmod._ATTN_ZEROCOPY = enabled


def set_overlap(enabled):
    pipemod._DECODER_OVERLAP = enabled


def set_pointer_state(enabled):
    tcdecmod._TCDEC_POINTER_STATE = enabled


def set_direct_output(enabled):
    pipemod._DECODER_DIRECT_OUTPUT = enabled


def set_pointwise(enabled):
    tcdecmod._TCDEC_FUSE_POINTWISE = enabled


def set_upsample(enabled):
    tcdecmod._TCDEC_UPSAMPLE = enabled


def set_lq_packer(enabled):
    wanutils._CONV3D_PACKER = "triton" if enabled else "eager"


def set_concat(enabled):
    tcdecmod._TCDEC_CONCAT = enabled


def set_tgrow_up(enabled):
    tcdecmod._TCDEC_TGROW_UP = enabled


def set_cudnn_fused(enabled):
    tcdecmod._TCDEC_CUDNN_FUSED = enabled


def run(pipe, lq, frame_count):
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        video = pipe(
            prompt="", negative_prompt="", cfg_scale=1.0,
            num_inference_steps=1, seed=0, LQ_video=lq,
            num_frames=frame_count, height=REF_H, width=REF_W,
            is_full_block=False, if_buffer=True,
            topk_ratio=2.0 * 768 * 1280 / (REF_H * REF_W),
            kv_ratio=3.0, local_range=11, color_fix=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    metadata = {
        "shape": tuple(video.shape),
        "dtype": str(video.dtype),
        "device": str(video.device),
    }
    output = video.detach().cpu()
    del video
    return output, metadata, elapsed


def expected_routes(frame_count, phase5, overlap, pointer_state=False,
                    direct_output=False, pointwise=False, upsample=False,
                    lq_packer=False, concat=False, tgrow_up=False,
                    cudnn_fused=False):
    n_chunks = (frame_count - 1) // 8 - 2
    attn_calls = TINY_BLOCKS * n_chunks
    h_lq, w_lq = REF_H // 16, REF_W // 16
    budget = int(float(os.environ.get(
        "FLASHVSR_CONV3D_IM2COL_BUDGET_GB", "2.0")) * 1e9)
    bytes_per_row1 = 2 * w_lq * (768 * 4 * 3 * 3) * 2
    bytes_per_row2 = w_lq * (2048 * 4 * 3 * 3) * 2
    rows1 = h_lq if budget <= 0 else max(
        1, min(h_lq, budget // bytes_per_row1))
    rows2 = h_lq if budget <= 0 else max(
        1, min(h_lq, budget // bytes_per_row2))
    pack_calls = ((2 * n_chunks + 5) * ((h_lq + rows1 - 1) // rows1)
                  + (2 * n_chunks + 4) * ((h_lq + rows2 - 1) // rows2))
    expected = {name: 0 for name in _ROUTE_KEYS}
    expected.update({
        "csr_fused": attn_calls,
        "conv3d_lean_gemm": 13 + 4 * (n_chunks - 1),
        "color_fix_success": 1,
    })
    if phase5:
        expected.update({
            "attn_zc_v2": attn_calls,
            "rope_triton": 2 * attn_calls,
            "pooled_k_rebuild": TINY_BLOCKS,
            "pooled_k_incremental": TINY_BLOCKS * (n_chunks - 1),
        })
    else:
        expected.update({
            "attn_v2": attn_calls,
            "rope_fused": 2 * attn_calls,
        })
    if overlap:
        expected["decoder_overlap_chunks"] = n_chunks
        if direct_output:
            expected["decoder_direct_output_chunks"] = n_chunks
            expected["decoder_direct_output_complete"] = 1
        else:
            expected["decoder_final_cat"] = 1
    else:
        expected["decoder_serialized_calls"] = 1
    latent_frames = 2 * (n_chunks + 2)
    state_updates = 3 * (latent_frames - 1) + 3 * (latent_frames - 1) \
        + 3 * (2 * latent_frames - 1)
    expected["decoder_state_pointer_updates" if pointer_state
             else "decoder_state_copies"] = state_updates
    if cudnn_fused:
        expected["decoder_memblock_cudnn"] = 12 * latent_frames
    else:
        expected["decoder_memblock_fused" if pointwise
                 else "decoder_memblock_native"] = 12 * latent_frames
    if tgrow_up:
        # All three Upsample->TGrow pairs are consumed by the fused path
        # (4 site executions per latent frame), so neither the standalone
        # upsample nor the native TGrow route fires.
        expected["decoder_tgrow_fused"] = 4 * latent_frames
    else:
        expected["decoder_tgrow_native"] = 3 * latent_frames
        expected["decoder_upsample_triton" if upsample
                 else "decoder_upsample_native"] = 4 * latent_frames
    expected["conv3d_packer_triton" if lq_packer
             else "conv3d_packer_eager"] = pack_calls
    expected["decoder_concat_triton" if concat
             else "decoder_concat_native"] = 12 * latent_frames
    if cudnn_fused:
        expected["decoder_conv_relu_cudnn"] = 10 * latent_frames
    else:
        expected["decoder_relu_native"] = 10 * latent_frames
    return expected


def expected_output_frames(frame_count):
    n_chunks = (frame_count - 1) // 8 - 2
    return 21 + 8 * (n_chunks - 1)


def check_routes(label, telemetry, expected):
    counts = telemetry["counts"]
    failures = []
    for name, wanted in expected.items():
        actual = counts.get(name, 0)
        if actual != wanted:
            failures.append(f"{name}: expected {wanted}, got {actual}")
    error_counts = {
        name: count for name, count in counts.items()
        if name.endswith("_error") and count
    }
    if error_counts:
        failures.append(f"error counters: {error_counts}")
    if telemetry["errors"]:
        failures.append(f"recorded errors: {telemetry['errors']}")
    if failures:
        print(f"  {label} routes: FAIL")
        for failure in failures:
            print(f"    {failure}")
        print(f"    counts={counts}")
        return False
    print(f"  {label} routes: OK")
    return True


def measured_run(pipe, lq, frame_count, phase5, overlap, label,
                 pointer_state=False, direct_output=False, pointwise=False,
                 upsample=False, lq_packer=False, concat=False,
                 tgrow_up=False, cudnn_fused=False):
    set_phase5(phase5)
    set_overlap(overlap)
    set_pointer_state(pointer_state)
    set_direct_output(direct_output)
    set_pointwise(pointwise)
    set_upsample(upsample)
    set_lq_packer(lq_packer)
    set_concat(concat)
    set_tgrow_up(tgrow_up)
    set_cudnn_fused(cudnn_fused)
    perf_stats.reset()
    output, metadata, elapsed = run(pipe, lq, frame_count)
    telemetry = perf_stats.snapshot()
    routes_ok = check_routes(
        label, telemetry,
        expected_routes(frame_count, phase5, overlap, pointer_state,
                        direct_output, pointwise, upsample, lq_packer, concat,
                        tgrow_up, cudnn_fused))
    output_frames = expected_output_frames(frame_count)
    frame_ok = len(metadata["shape"]) == 4 and metadata["shape"][1] == output_frames
    if not frame_ok:
        print(f"  {label} frame count: FAIL expected {output_frames}, "
              f"got shape={metadata['shape']}")
    fps = output_frames / elapsed
    print(f"  {label}: {elapsed:.3f}s {fps:6.2f} FPS "
          f"shape={metadata['shape']} dtype={metadata['dtype']}")
    return output, metadata, routes_ok and frame_ok


def max_abs_diff(a, b):
    if a.shape != b.shape:
        return float("inf")
    maximum = 0.0
    # Compute the independent max-diff gate without a clip-sized temporary.
    for i in range(a.shape[1]):
        diff = (a[:, i].float() - b[:, i].float()).abs().max().item()
        maximum = max(maximum, diff)
    return maximum


def compare(label, reference, reference_meta, candidate, candidate_meta):
    exact = torch.equal(reference, candidate)
    max_diff = max_abs_diff(reference, candidate)
    metadata_equal = reference_meta == candidate_meta
    ok = exact and max_diff == 0.0 and metadata_equal
    print(f"  {label}: torch.equal={exact} max|diff|={max_diff:.3e} "
          f"metadata_equal={metadata_equal} [{'OK' if ok else 'FAIL'}]")
    if not metadata_equal:
        print(f"    metadata: {reference_meta} vs {candidate_meta}")
    return ok


def psnr_db(reference, candidate):
    """PSNR over the [-1, 1] output range, accumulated frame-by-frame."""
    if reference.shape != candidate.shape:
        return float("-inf")
    total, count = 0.0, 0
    for i in range(reference.shape[1]):
        diff = (reference[:, i].double() - candidate[:, i].double())
        total += diff.pow(2).sum().item()
        count += diff.numel()
    if total == 0.0:
        return float("inf")
    import math
    return 10.0 * math.log10(4.0 / (total / count))


def compare_quality(label, reference, reference_meta, candidate,
                    candidate_meta, min_db=49.0):
    """Bitwise if possible; otherwise gate on the >=49 dB PSNR budget."""
    exact = torch.equal(reference, candidate)
    max_diff = max_abs_diff(reference, candidate)
    metadata_equal = reference_meta == candidate_meta
    quality = float("inf") if exact else psnr_db(reference, candidate)
    ok = metadata_equal and (exact or quality >= min_db)
    print(f"  {label}: torch.equal={exact} max|diff|={max_diff:.3e} "
          f"psnr={quality:.2f}dB metadata_equal={metadata_equal} "
          f"[{'OK' if ok else 'FAIL'}]")
    if not metadata_equal:
        print(f"    metadata: {reference_meta} vs {candidate_meta}")
    return ok


def check_import_configuration():
    checks = {
        "conv3d backend": wanutils._CONV3D_BACKEND == "gemm",
        "channels-last decoder": tcdecmod._TCDEC_CHANNELS_LAST,
        "fuse norm": ditmod._FUSE_NORM,
        "triton2": ditmod._ATTN_BACKEND == "triton2",
        "cache mod": ditmod._CACHE_MOD,
        "cache mask": ditmod._CACHE_MASK_BIAS,
        "RoPE frequency cache off": not pipemod._CACHE_ROPE_FREQS,
        "fuse rope": ditmod._FUSE_ROPE,
        "KV ring": ditmod._KV_RINGBUF,
        "strided IO": ditmod._ATTN_STRIDED_IO,
        "mask lean": ditmod._MASKGEN_LEAN,
        "LQ lean": wanutils._LQPROJ_LEAN,
        "fused CSR": v2mod._FUSED_CSR,
        "RoPE kernel": ditmod._ROPE_KERNEL == "triton",
        "pooled-K": ditmod._POOLED_K_CACHE,
        "zero-copy": ditmod._ATTN_ZEROCOPY,
        "FP8 off": not fp8mod._FP8_GEMM,
        "telemetry": perf_stats.enabled(),
    }
    failures = [name for name, good in checks.items() if not good]
    if failures:
        print(f"FAIL: import-time stack mismatch: {failures}")
        return False
    return True


def run_e2e(pipe, lq):
    print(f"\n=== Phase-5 P1-P3 lossless parity @ {REF_W}x{REF_H} F={F_LONG} ===")
    reference, reference_meta, ok = measured_run(
        pipe, lq, F_LONG, phase5=False, overlap=False,
        label="Phase-3.5 base")
    candidate, candidate_meta, candidate_ok = measured_run(
        pipe, lq, F_LONG, phase5=True, overlap=False,
        label="Phase-5 P1-P3")
    ok = ok and candidate_ok
    ok = compare("P1-P3 ON vs OFF", reference, reference_meta,
                 candidate, candidate_meta) and ok
    del candidate, reference
    return ok


def run_overlap_case(pipe, lq, frame_count):
    print(f"\n-- serialized vs overlap F={frame_count} --")
    reference, reference_meta, ok = measured_run(
        pipe, lq, frame_count, phase5=True, overlap=False,
        label="serialized")
    for repeat in range(1, 4):
        candidate, candidate_meta, candidate_ok = measured_run(
            pipe, lq, frame_count, phase5=True, overlap=True,
            label=f"overlap rep{repeat}")
        ok = candidate_ok and ok
        ok = compare(f"overlap rep{repeat} vs serialized", reference,
                     reference_meta, candidate, candidate_meta) and ok
        # Every repeat is exact against the same reference, which also proves
        # repeat stability without retaining three full outputs.
        del candidate
    del reference
    return ok


def run_pointer_case(pipe, lq, frame_count):
    print(f"\n-- recurrent pointer state F={frame_count} --")
    reference, reference_meta, ok = measured_run(
        pipe, lq, frame_count, phase5=True, overlap=False,
        label="copy state", pointer_state=False)
    candidate, candidate_meta, candidate_ok = measured_run(
        pipe, lq, frame_count, phase5=True, overlap=False,
        label="pointer state", pointer_state=True)
    ok = candidate_ok and ok
    ok = compare("pointer vs copy", reference, reference_meta,
                 candidate, candidate_meta) and ok
    del candidate
    if frame_count == F_LONG:
        overlap, overlap_meta, overlap_ok = measured_run(
            pipe, lq, frame_count, phase5=True, overlap=True,
            label="pointer state + overlap", pointer_state=True)
        ok = overlap_ok and ok
        ok = compare("pointer overlap vs copy serialized", reference,
                     reference_meta, overlap, overlap_meta) and ok
        del overlap
    del reference
    return ok


def run_pointer(pipe, lq):
    print(f"\n=== TCDecoder pointer-state parity @ {REF_W}x{REF_H} ===")
    short_ok = run_pointer_case(pipe, lq[:, :, :F_SHORT], F_SHORT)
    long_ok = run_pointer_case(pipe, lq, F_LONG)
    set_pointer_state(False)
    set_overlap(False)
    return short_ok and long_ok


def run_direct_case(pipe, lq, frame_count):
    print(f"\n-- direct decoder output F={frame_count} --")
    reference, reference_meta, ok = measured_run(
        pipe, lq, frame_count, phase5=True, overlap=True,
        label="chunk cat", pointer_state=True, direct_output=False)
    candidate, candidate_meta, candidate_ok = measured_run(
        pipe, lq, frame_count, phase5=True, overlap=True,
        label="direct output", pointer_state=True, direct_output=True)
    ok = candidate_ok and ok
    ok = compare("direct output vs chunk cat", reference, reference_meta,
                 candidate, candidate_meta) and ok
    del candidate, reference
    return ok


def run_direct(pipe, lq):
    print(f"\n=== TCDecoder direct-output parity @ {REF_W}x{REF_H} ===")
    short_ok = run_direct_case(pipe, lq[:, :, :F_SHORT], F_SHORT)
    edge_ok = run_direct_case(pipe, lq[:, :, :F_EDGE], F_EDGE)
    long_ok = run_direct_case(pipe, lq, F_LONG)
    set_direct_output(False)
    set_pointer_state(False)
    set_overlap(False)
    return short_ok and edge_ok and long_ok


def run_pointwise_case(pipe, lq, frame_count):
    print(f"\n-- fused MemBlock pointwise F={frame_count} --")
    reference, reference_meta, ok = measured_run(
        pipe, lq, frame_count, phase5=True, overlap=True,
        label="native pointwise", pointer_state=True, direct_output=True,
        pointwise=False)
    candidate, candidate_meta, candidate_ok = measured_run(
        pipe, lq, frame_count, phase5=True, overlap=True,
        label="fused pointwise", pointer_state=True, direct_output=True,
        pointwise=True)
    ok = candidate_ok and ok
    ok = compare("fused vs native pointwise", reference, reference_meta,
                 candidate, candidate_meta) and ok
    del candidate, reference
    return ok


def run_pointwise(pipe, lq):
    print(f"\n=== TCDecoder fused-pointwise parity @ {REF_W}x{REF_H} ===")
    short_ok = run_pointwise_case(pipe, lq[:, :, :F_SHORT], F_SHORT)
    long_ok = run_pointwise_case(pipe, lq, F_LONG)
    set_pointwise(False)
    set_direct_output(False)
    set_pointer_state(False)
    set_overlap(False)
    return short_ok and long_ok


def run_upsample_case(pipe, lq, frame_count):
    print(f"\n-- Triton nearest upsample F={frame_count} --")
    reference, reference_meta, ok = measured_run(
        pipe, lq, frame_count, phase5=True, overlap=True,
        label="native upsample", pointer_state=True, direct_output=True,
        pointwise=True, upsample=False)
    candidate, candidate_meta, candidate_ok = measured_run(
        pipe, lq, frame_count, phase5=True, overlap=True,
        label="Triton upsample", pointer_state=True, direct_output=True,
        pointwise=True, upsample=True)
    ok = candidate_ok and ok
    ok = compare("Triton vs native upsample", reference, reference_meta,
                 candidate, candidate_meta) and ok
    del candidate, reference
    return ok


def run_upsample(pipe, lq):
    print(f"\n=== TCDecoder Triton upsample parity @ {REF_W}x{REF_H} ===")
    short_ok = run_upsample_case(pipe, lq[:, :, :F_SHORT], F_SHORT)
    long_ok = run_upsample_case(pipe, lq, F_LONG)
    set_upsample(False)
    set_pointwise(False)
    set_direct_output(False)
    set_pointer_state(False)
    set_overlap(False)
    return short_ok and long_ok


def run_lqpacker_case(pipe, lq, frame_count):
    print(f"\n-- Triton LQ im2col packer F={frame_count} --")
    reference, reference_meta, ok = measured_run(
        pipe, lq, frame_count, phase5=True, overlap=True,
        label="eager LQ packer", pointer_state=True, direct_output=True,
        pointwise=True, upsample=True, lq_packer=False)
    candidate, candidate_meta, candidate_ok = measured_run(
        pipe, lq, frame_count, phase5=True, overlap=True,
        label="Triton LQ packer", pointer_state=True, direct_output=True,
        pointwise=True, upsample=True, lq_packer=True)
    ok = candidate_ok and ok
    ok = compare("Triton vs eager LQ packer", reference, reference_meta,
                 candidate, candidate_meta) and ok
    del candidate, reference
    return ok


def run_lqpacker(pipe, lq):
    print(f"\n=== Triton LQ im2col parity @ {REF_W}x{REF_H} ===")
    short_ok = run_lqpacker_case(pipe, lq[:, :, :F_SHORT], F_SHORT)
    long_ok = run_lqpacker_case(pipe, lq, F_LONG)
    set_lq_packer(False)
    set_upsample(False)
    set_pointwise(False)
    set_direct_output(False)
    set_pointer_state(False)
    set_overlap(False)
    return short_ok and long_ok


def run_concat_case(pipe, lq, frame_count):
    print(f"\n-- Triton recurrent concat F={frame_count} --")
    reference, reference_meta, ok = measured_run(
        pipe, lq, frame_count, phase5=True, overlap=True,
        label="native concat", pointer_state=True, direct_output=True,
        pointwise=True, upsample=True, lq_packer=True, concat=False)
    candidate, candidate_meta, candidate_ok = measured_run(
        pipe, lq, frame_count, phase5=True, overlap=True,
        label="Triton concat", pointer_state=True, direct_output=True,
        pointwise=True, upsample=True, lq_packer=True, concat=True)
    ok = candidate_ok and ok
    ok = compare("Triton vs native concat", reference, reference_meta,
                 candidate, candidate_meta) and ok
    del candidate, reference
    return ok


def run_concat(pipe, lq):
    print(f"\n=== TCDecoder Triton concat parity @ {REF_W}x{REF_H} ===")
    short_ok = run_concat_case(pipe, lq[:, :, :F_SHORT], F_SHORT)
    long_ok = run_concat_case(pipe, lq, F_LONG)
    set_concat(False)
    set_lq_packer(False)
    set_upsample(False)
    set_pointwise(False)
    set_direct_output(False)
    set_pointer_state(False)
    set_overlap(False)
    return short_ok and long_ok


def run_tgrow_up_case(pipe, lq, frame_count):
    print(f"\n-- fused TGrow upsample F={frame_count} --")
    reference, reference_meta, ok = measured_run(
        pipe, lq, frame_count, phase5=True, overlap=True,
        label="native TGrow path", pointer_state=True, direct_output=True,
        pointwise=True, upsample=True, lq_packer=True, concat=True,
        tgrow_up=False)
    candidate, candidate_meta, candidate_ok = measured_run(
        pipe, lq, frame_count, phase5=True, overlap=True,
        label="fused TGrow path", pointer_state=True, direct_output=True,
        pointwise=True, upsample=True, lq_packer=True, concat=True,
        tgrow_up=True)
    ok = candidate_ok and ok
    ok = compare_quality("fused vs native TGrow", reference, reference_meta,
                         candidate, candidate_meta) and ok
    del candidate, reference
    return ok


def run_tgrow_up(pipe, lq):
    print(f"\n=== TCDecoder fused TGrow-upsample parity @ {REF_W}x{REF_H} ===")
    short_ok = run_tgrow_up_case(pipe, lq[:, :, :F_SHORT], F_SHORT)
    edge_ok = run_tgrow_up_case(pipe, lq[:, :, :F_EDGE], F_EDGE)
    long_ok = run_tgrow_up_case(pipe, lq, F_LONG)
    set_tgrow_up(False)
    set_concat(False)
    set_lq_packer(False)
    set_upsample(False)
    set_pointwise(False)
    set_direct_output(False)
    set_pointer_state(False)
    set_overlap(False)
    return short_ok and edge_ok and long_ok


def run_cudnn_fused_case(pipe, lq, frame_count):
    print(f"\n-- cuDNN fused decoder convs F={frame_count} --")
    reference, reference_meta, ok = measured_run(
        pipe, lq, frame_count, phase5=True, overlap=True,
        label="separate epilogues", pointer_state=True, direct_output=True,
        pointwise=True, upsample=True, lq_packer=True, concat=True,
        tgrow_up=True, cudnn_fused=False)
    candidate, candidate_meta, candidate_ok = measured_run(
        pipe, lq, frame_count, phase5=True, overlap=True,
        label="cuDNN fused convs", pointer_state=True, direct_output=True,
        pointwise=True, upsample=True, lq_packer=True, concat=True,
        tgrow_up=True, cudnn_fused=True)
    ok = candidate_ok and ok
    ok = compare_quality("cuDNN fused vs separate", reference, reference_meta,
                         candidate, candidate_meta) and ok
    del candidate, reference
    return ok


def run_cudnn_fused(pipe, lq):
    print(f"\n=== TCDecoder cuDNN-fused conv quality gate @ {REF_W}x{REF_H} ===")
    short_ok = run_cudnn_fused_case(pipe, lq[:, :, :F_SHORT], F_SHORT)
    long_ok = run_cudnn_fused_case(pipe, lq, F_LONG)
    set_cudnn_fused(False)
    set_tgrow_up(False)
    set_concat(False)
    set_lq_packer(False)
    set_upsample(False)
    set_pointwise(False)
    set_direct_output(False)
    set_pointer_state(False)
    set_overlap(False)
    return short_ok and long_ok


def run_phase6_case(pipe, lq, frame_count):
    print(f"\n-- Phase-6 cumulative stack F={frame_count} --")
    reference, reference_meta, ok = measured_run(
        pipe, lq, frame_count, phase5=True, overlap=True,
        label="Phase-5 production", pointer_state=False, direct_output=False,
        pointwise=False, upsample=False, lq_packer=False, concat=False)
    candidate, candidate_meta, candidate_ok = measured_run(
        pipe, lq, frame_count, phase5=True, overlap=True,
        label="Phase-6 optimized", pointer_state=True, direct_output=True,
        pointwise=True, upsample=True, lq_packer=True, concat=True)
    ok = candidate_ok and ok
    ok = compare("Phase-6 vs Phase-5", reference, reference_meta,
                  candidate, candidate_meta) and ok
    del candidate, reference
    return ok


def run_phase6(pipe, lq):
    print(f"\n=== Phase-6 cumulative lossless parity @ {REF_W}x{REF_H} ===")
    edge_ok = run_phase6_case(pipe, lq[:, :, :F_EDGE], F_EDGE)
    long_ok = run_phase6_case(pipe, lq, F_LONG)
    set_concat(False)
    set_lq_packer(False)
    set_upsample(False)
    set_pointwise(False)
    set_direct_output(False)
    set_pointer_state(False)
    set_overlap(False)
    return edge_ok and long_ok


def run_overlap(pipe, lq):
    print(f"\n=== Phase-5 decoder-overlap lossless parity @ {REF_W}x{REF_H} ===")
    short_ok = run_overlap_case(pipe, lq[:, :, :F_SHORT], F_SHORT)
    long_ok = run_overlap_case(pipe, lq, F_LONG)
    set_overlap(False)
    return short_ok and long_ok


def parse_mode(argv):
    if not argv:
        return "all"
    if len(argv) == 1 and argv[0].lower() in (
            "e2e", "overlap", "pointer", "direct", "pointwise", "upsample",
            "lqpacker", "concat", "tgrowup", "cudnnfuse", "phase6", "all"):
        return argv[0].lower()
    print(f"usage: {sys.argv[0]} "
          "[e2e|overlap|pointer|direct|pointwise|upsample|lqpacker|concat"
          "|tgrowup|cudnnfuse|phase6|all]")
    return None


def main():
    mode = parse_mode(sys.argv[1:])
    if mode is None:
        return 2
    if not check_import_configuration():
        return 1

    pipe = init_pipeline()
    if len(pipe.dit.blocks) != TINY_BLOCKS:
        print(f"FAIL: expected v1.1 Tiny with {TINY_BLOCKS} blocks, "
              f"got {len(pipe.dit.blocks)}")
        return 1
    lq = build_lq("./inputs/example0.mp4", F_LONG)

    ok = True
    if mode in ("e2e", "all"):
        ok = run_e2e(pipe, lq) and ok
    if mode in ("overlap", "all"):
        ok = run_overlap(pipe, lq) and ok
    if mode in ("pointer", "all"):
        ok = run_pointer(pipe, lq) and ok
    if mode in ("direct", "all"):
        ok = run_direct(pipe, lq) and ok
    if mode in ("pointwise", "all"):
        ok = run_pointwise(pipe, lq) and ok
    if mode in ("upsample", "all"):
        ok = run_upsample(pipe, lq) and ok
    if mode in ("lqpacker", "all"):
        ok = run_lqpacker(pipe, lq) and ok
    if mode in ("concat", "all"):
        ok = run_concat(pipe, lq) and ok
    if mode in ("tgrowup", "all"):
        ok = run_tgrow_up(pipe, lq) and ok
    if mode in ("cudnnfuse", "all"):
        ok = run_cudnn_fused(pipe, lq) and ok
    if mode in ("phase6", "all"):
        ok = run_phase6(pipe, lq) and ok
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
