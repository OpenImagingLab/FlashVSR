#!/usr/bin/env python3
"""Quality gate for Phase-7 DiT LayerNorm/AdaLN row fusion.

Runs the current production stack with ``FLASHVSR_DIT_ROW_FUSION`` disabled and
enabled in one process so both calls share weights and input. The Triton
reduction is accepted only when output PSNR is at least 49 dB.
"""

import importlib.util
import math
import os
import sys
import time

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(os.path.dirname(_here))
os.chdir(_here)
sys.path.insert(0, _root)
sys.path.insert(0, _here)

os.environ.update({
    "FLASHVSR_CONV3D_BACKEND": "gemm",
    "FLASHVSR_CONV3D_PACKER": "triton",
    "FLASHVSR_TCDECODER_CHANNELS_LAST": "1",
    "FLASHVSR_FUSE_NORM": "1",
    "FLASHVSR_ATTN_BACKEND": "triton2",
    "FLASHVSR_ATTN_TMA": "1",
    "FLASHVSR_CACHE_MOD": "1",
    "FLASHVSR_CACHE_MASK_BIAS": "1",
    "FLASHVSR_CACHE_ROPE_FREQS": "0",
    "FLASHVSR_FUSE_ROPE": "1",
    "FLASHVSR_ROPE_KERNEL": "triton",
    "FLASHVSR_KV_RINGBUF": "1",
    "FLASHVSR_ATTN_STRIDED_IO": "1",
    "FLASHVSR_MASKGEN_LEAN": "1",
    "FLASHVSR_LQPROJ_LEAN": "1",
    "FLASHVSR_FUSED_CSR": "1",
    "FLASHVSR_POOLED_K_CACHE": "1",
    "FLASHVSR_ATTN_ZEROCOPY": "1",
    "FLASHVSR_DECODER_OVERLAP": "1",
    "FLASHVSR_FP8_GEMM": "0",
    "FLASHVSR_TCDECODER_POINTER_STATE": "1",
    "FLASHVSR_TCDECODER_DIRECT_OUTPUT": "1",
    "FLASHVSR_TCDECODER_FUSE_POINTWISE": "1",
    "FLASHVSR_TCDECODER_UPSAMPLE": "1",
    "FLASHVSR_TCDECODER_CONCAT": "1",
    "FLASHVSR_TCDECODER_TGROW_UP": "1",
    "FLASHVSR_TCDECODER_CUDNN_FUSED": "1",
    "FLASHVSR_DIT_ROW_FUSION": "1",
    "FLASHVSR_TELEMETRY": "1",
})

import torch  # noqa: E402

from diffsynth import perf_stats  # noqa: E402
import diffsynth.models.wan_video_dit as ditmod  # noqa: E402
from utils import TCDecoder as tcdecmod  # noqa: E402
from profiling.run_pipe_target import build_lq  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "infer_v1_1_tiny", os.path.join(_here, "infer_flashvsr_v1.1_tiny.py"))
_infer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_infer)

W = int(os.environ.get("FLASHVSR_TEST_W", "768"))
H = int(os.environ.get("FLASHVSR_TEST_H", "1408"))
F = int(sys.argv[1]) if len(sys.argv) > 1 else 25
if F < 25 or F % 8 not in (1, 5):
    raise ValueError("frame count must be >=25 and satisfy F % 8 in {1, 5}")
TEST_ROW_FUSION = os.environ.get("FLASHVSR_TEST_DIT_ROWS", "1") != "0"
TEST_THRESHOLD_CACHE = os.environ.get("FLASHVSR_TEST_MASKGEN_THRESHOLD_CACHE", "0") != "0"
TEST_SPLITK_CONV = os.environ.get("FLASHVSR_TEST_SPLITK_CONV", "0") != "0"
TEST_KV_SPARE8 = os.environ.get("FLASHVSR_TEST_KV_SPARE8", "0") != "0"
DEFAULT_KV_SPARE = ditmod._KV_RINGBUF_SPARE


def run(pipe, lq, enabled):
    ditmod._DIT_ROW_FUSION = enabled and TEST_ROW_FUSION
    ditmod._DIT_ROW_FUSION_FAILED = False
    ditmod._MASKGEN_THRESHOLD_CACHE = enabled and TEST_THRESHOLD_CACHE
    tcdecmod._TCDEC_SPLITK_CONV = enabled and TEST_SPLITK_CONV
    tcdecmod._TCDEC_SPLITK_CONV_FAILED = False
    ditmod._KV_RINGBUF_SPARE = 8 if enabled and TEST_KV_SPARE8 else DEFAULT_KV_SPARE
    perf_stats.reset()
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        output = pipe(
            prompt="", negative_prompt="", cfg_scale=1.0,
            num_inference_steps=1, seed=0, LQ_video=lq,
            num_frames=F, height=H, width=W,
            is_full_block=False, if_buffer=True,
            topk_ratio=2.0 * 768 * 1280 / (H * W),
            kv_ratio=3.0, local_range=11, color_fix=True,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    counts = perf_stats.snapshot()
    return output.cpu(), elapsed, counts


def psnr(reference, candidate):
    mse = (reference.float() - candidate.float()).square().mean().item()
    return float("inf") if mse == 0.0 else 10.0 * math.log10(4.0 / mse)


def main():
    pipe = _infer.init_pipeline()
    lq, _, _, _, _ = build_lq("./inputs/example0.mp4", W, H, F)
    reference, ref_time, ref_routes = run(pipe, lq, False)
    candidate, candidate_time, candidate_routes = run(pipe, lq, True)

    chunks = (F - 1) // 8 - 2
    expected_calls = 2 * 30 * chunks
    exact = torch.equal(reference, candidate)
    max_diff = (reference.float() - candidate.float()).abs().max().item()
    quality = float("inf") if exact else psnr(reference, candidate)
    routes_ok = not candidate_routes["errors"]
    if TEST_ROW_FUSION:
        routes_ok = (
            routes_ok
            and candidate_routes["counts"].get("dit_row_ln_modulate", 0) == expected_calls
            and candidate_routes["counts"].get("dit_row_gate", 0) == expected_calls
        )
    if TEST_SPLITK_CONV:
        latent_frames = 2 * (chunks + 2)
        routes_ok = (
            routes_ok
            and candidate_routes["counts"].get("decoder_memblock_splitk", 0)
            == 12 * latent_frames
            and candidate_routes["counts"].get("decoder_concat_triton", 0) == 0
        )
    fps_ref = (F - 4) / ref_time
    fps_candidate = (F - 4) / candidate_time
    parts = []
    if TEST_ROW_FUSION:
        parts.append("A")
    if TEST_THRESHOLD_CACHE:
        parts.append("B")
    if TEST_SPLITK_CONV:
        parts.append("C")
    if TEST_KV_SPARE8:
        parts.append("E")
    label = "P7-" + "+".join(parts)
    print(f"[{label}] F={F} off={fps_ref:.2f} FPS on={fps_candidate:.2f} FPS")
    print(f"[P7-A] exact={exact} max|diff|={max_diff:.4f} psnr={quality:.2f} dB")
    print(f"[P7-A] routes={candidate_routes['counts']} errors={candidate_routes['errors']}")
    if not routes_ok or quality < 49.0:
        raise SystemExit("P7-A quality or routing gate failed")


if __name__ == "__main__":
    main()
