# ROCm inference

FlashVSR can execute its official 128-token Locality-Constrained Sparse
Attention (LCSA) block masks on AMD GPUs using compiled PyTorch FlexAttention.
Unmasked attention uses PyTorch SDPA. NVIDIA retains its existing attention
backend. Model architecture, checkpoint keys and streaming KV-cache ordering
are preserved.

## Installation

Validated on Radeon 8060S (gfx1151, wave32), Linux, Python 3.12,
PyTorch 2.15.0a0+rocm10.1.0a20260902, reported HIP 7.16.26351 and Triton 3.8.0.
Use a matching ROCm PyTorch/torchvision/torchaudio environment. The requirements
below record the inference dependencies used in that environment; they are
not a promise of compatibility with older ROCm/PyTorch releases.

```sh
python -m pip install -r requirements-rocm.txt
python -m pip install -e . --no-deps
python tests/test_rocm_attention.py
```

Do not install the original requirements.txt into this environment: its CUDA
PyTorch pins would replace ROCm PyTorch. Download official v1.1 checkpoints as
explained in README.md. Both v1.1 Full and Tiny examples accept the model path:

```sh
cd examples/WanVSR
FLASHVSR_MODEL_DIR=/path/to/FlashVSR-v1.1 python infer_flashvsr_v1.1_tiny.py
# Or: python infer_flashvsr_v1.1_full.py with the same environment variable.
```

The default remains ./FlashVSR-v1.1. PyTorch uses device="cuda" on ROCm too.
The long-video example retains its existing checkpoint-directory convention.

## Backend controls

- FLASHVSR_ROCM_ATTENTION=flex (default): compiled sparse attention.
- FLASHVSR_ROCM_ATTENTION=reference: diagnostic FP32 SDPA over gathered allowed
  keys; preserves the mask but is slower.
- FLASHVSR_ROCM_PRECISION=fp32: diagnostic FP32 FlexAttention inputs; native
  dtype is the default.

FlexAttention compilation failures are raised; there is no silent dense eager
fallback. Compilation is excluded from warm benchmark results. Many distinct
shapes can exceed PyTorch's compilation limit; the library does not change
process-wide compiler settings. Empty block rows explicitly produce zero.
Large mask scores are computed per head/time group to bound temporary memory,
with unchanged per-group threshold selection. Dense block scores still scale
quadratically.

Tiny's optional TCDecoder.set_rocm_cache_release() installs removable MemBlock
cache-release hooks. It is disabled by default and installs no hooks on CUDA.
Call with False to remove hooks. This trades allocator reuse for lower decoder
cost in the measured memory-constrained workload; it is not universally faster.

## Reproducible validation

Run these commands from the repository root; model/output paths are resolved
relative to the invoking directory. Benchmarks run sequentially on an idle GPU.

```sh
python tests/test_rocm_attention.py
python examples/WanVSR/rocm/benchmark_rocm_video.py --model-dir MODEL_DIR --output-dir RESULTS --backend flex --height 256 --width 512 --frames 65
python examples/WanVSR/rocm/benchmark_rocm_video.py --model-dir MODEL_DIR --output-dir RESULTS --backend reference --height 256 --width 512 --frames 65
python examples/WanVSR/rocm/compare_rocm_video.py RESULTS/flex_native_512x256_65/output_3.pt RESULTS/reference_native_512x256_65/output_3.pt
```

The bundled example0 benchmark uses official v1.1 Tiny checkpoints, BF16,
seed 0, one step, bicubic 4x degradation/reconstruction, local_range=11,
kv_ratio=3 and the official resolution-dependent topk_ratio. Four runs are
recorded; exclude the first. Times cover synchronized inference and decoding,
with loading and saving excluded. Outputs and metadata are saved separately.

Seven regression tests cover FP32/BF16 sparse attention, unequal Q/KV lengths,
partial blocks, empty rows, grouped mask equality, streaming/aspect changes,
compiler failure handling and cache-hook replacement/removal.

The fixed large-video runner requires a 768x1344, 73-frame, 24 FPS input and
produces 1536x2688, 73-frame video. It pads to 81 inference frames and trims
back to the original duration. It uses additional mask-cache release in both
variants. baseline_clean releases decoder cache after each Conv2d; optimized
uses MemBlock hooks. This is a comparison of two ROCm memory-management
settings, not an unmodified NVIDIA baseline. Videos contain no audio.

```sh
PYTORCH_ALLOC_CONF=expandable_segments:True python examples/WanVSR/rocm/run_rocm_orbit.py --input INPUT.mp4 --model-dir MODEL_DIR --output-dir RESULTS --variant baseline_clean
PYTORCH_ALLOC_CONF=expandable_segments:True python examples/WanVSR/rocm/run_rocm_orbit.py --input INPUT.mp4 --model-dir MODEL_DIR --output-dir RESULTS --reference RESULTS/baseline_clean/output.pt --variant optimized
```

## Validation limits

Operator tolerances against gathered FP32 SDPA are atol/rtol=0.001 for FP32
and 0.02 for BF16. Final-output comparisons measure backend agreement, not HR
ground-truth restoration quality. Visual samples check visible corruption and
structure retention; they do not establish universal video quality.

No NVIDIA runtime/numerical comparison, wave64 GPU or Windows validation has
been performed. ROCm automatic dispatch also applies to other components using
the shared Wan attention helper; their end-to-end behavior is unvalidated.
Empty-row equivalence to the NVIDIA sparse backend remains unvalidated.
Full inference has a previous successful tiled-VAE large-video run, but the
repeatable benchmark in this change exercises Tiny. Default example sizes,
1080p/4K throughput and broad ROCm version compatibility are not certified.

## Measured results (2026-10-21)

On the environment above, 512x256 / 65 input frames produced 61 output frames.
Three warm runs after one excluded run gave median 3.2381 s (18.84 FPS) for
FlexAttention and 9.0937 s (6.71 FPS) for the diagnostic reference: 2.81x faster.
Peak allocated memory was 5.088 GB. Final backend agreement was 46.62 dB PSNR
(range 2), RMSE 0.009332 and temporal-delta RMSE 0.006464.

The previously recorded fixed large-video A/B gave 290.210 s for per-Conv2d
cache release and 264.814 s for MemBlock cache release, with exactly identical
output tensors (max absolute difference and RMSE both zero). Allocated peak
was 42.945 GB; reserved memory increased from 43.256 to 44.703 GB. These are
individual large-video runs, not statistical performance claims. Source/output
frame samples showed no obvious structural corruption; this is a limited
visual inspection, not a ground-truth quality score.

Final large-video rerun on the organized code: inference/decode 263.173 s,
decode 60.284 s, total through encoding 274.414 s; exact output equality
(max absolute difference/RMSE 0) and unchanged 42.945 GB allocated peak.
