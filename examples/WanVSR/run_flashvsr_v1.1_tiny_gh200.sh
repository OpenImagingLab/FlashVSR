#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PY="${PYTHON:-$ROOT/venv/bin/python}"

# Phase-5 Hopper stack.
export FLASHVSR_CONV3D_BACKEND=gemm
export FLASHVSR_CONV3D_IM2COL_BUDGET_GB="${FLASHVSR_CONV3D_IM2COL_BUDGET_GB:-2.0}"
export FLASHVSR_TCDECODER_CHANNELS_LAST=1
export FLASHVSR_FUSE_NORM=1
export FLASHVSR_ATTN_BACKEND=triton2
export FLASHVSR_ATTN_TMA=1
export FLASHVSR_CACHE_MOD=1
export FLASHVSR_CACHE_MASK_BIAS=1
export FLASHVSR_CACHE_ROPE_FREQS=0
export FLASHVSR_FUSE_ROPE=1
export FLASHVSR_ROPE_KERNEL=triton
export FLASHVSR_KV_RINGBUF=1
export FLASHVSR_ATTN_STRIDED_IO=1
export FLASHVSR_MASKGEN_LEAN=1
export FLASHVSR_LQPROJ_LEAN=1
export FLASHVSR_FUSED_CSR=1
export FLASHVSR_POOLED_K_CACHE=1
export FLASHVSR_ATTN_ZEROCOPY=1
export FLASHVSR_DECODER_OVERLAP=1
export FLASHVSR_FP8_GEMM=0

# Phase-7 quality-gated DiT / mask-generation paths (combined PSNR >=49 dB).
export FLASHVSR_DIT_ROW_FUSION=1
export FLASHVSR_MASKGEN_THRESHOLD_CACHE=1

# Phase-6 lossless decoder/LQ paths.
export FLASHVSR_CONV3D_PACKER=triton
export FLASHVSR_TCDECODER_POINTER_STATE=1
export FLASHVSR_TCDECODER_DIRECT_OUTPUT=1
export FLASHVSR_TCDECODER_FUSE_POINTWISE=1
export FLASHVSR_TCDECODER_UPSAMPLE=1
export FLASHVSR_TCDECODER_CONCAT=1

# Phase-6b decoder paths.
# TGROW_UP reorders Upsample->TGrow (measured bit-identical E2E).
export FLASHVSR_TCDECODER_TGROW_UP=1
# CUDNN_FUSED is QUALITY-GATED (~55 dB PSNR vs the separate-epilogue path,
# gate >= 49 dB), not bit-exact. Set to 0 for the strictly lossless stack.
export FLASHVSR_TCDECODER_CUDNN_FUSED=1
# Split the recurrent MemBlock input-conv weights instead of materializing
# cat([current, past]); requires CUDNN_FUSED and is quality-gated.
export FLASHVSR_TCDECODER_SPLITK_CONV=1

cd "$HERE"
exec "$PY" infer_flashvsr_v1.1_tiny.py "$@"
