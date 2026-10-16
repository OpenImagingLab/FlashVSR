#!/usr/bin/env bash
# Phase-7 benchmark driver: production knob set + overrides, N runs, median FPS.
# Usage: bench_phase7.sh LABEL [NRUNS] [KEY=VAL ...]
# Logs to profiling/runs/phase7/<LABEL>_run{i}.log
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
WANVSR="$(cd "$HERE/.." && pwd)"
ROOT="$(cd "$WANVSR/../.." && pwd)"
PY="${PYTHON:-$ROOT/venv/bin/python}"

LABEL="${1:?usage: bench_phase7.sh LABEL [NRUNS] [KEY=VAL ...]}"
NRUNS="${2:-3}"
shift 2 2>/dev/null || shift 1

# --- Production recommended set (mirror of run_flashvsr_v1.1_tiny_gh200.sh) ---
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
export FLASHVSR_DIT_ROW_FUSION=1
export FLASHVSR_MASKGEN_THRESHOLD_CACHE=1
export FLASHVSR_CONV3D_PACKER=triton
export FLASHVSR_TCDECODER_POINTER_STATE=1
export FLASHVSR_TCDECODER_DIRECT_OUTPUT=1
export FLASHVSR_TCDECODER_FUSE_POINTWISE=1
export FLASHVSR_TCDECODER_UPSAMPLE=1
export FLASHVSR_TCDECODER_CONCAT=1
export FLASHVSR_TCDECODER_TGROW_UP=1
export FLASHVSR_TCDECODER_CUDNN_FUSED=1
export FLASHVSR_TCDECODER_SPLITK_CONV=1

# --- Bench workload defaults ---
export FLASHVSR_PROF_W="${FLASHVSR_PROF_W:-768}"
export FLASHVSR_PROF_H="${FLASHVSR_PROF_H:-1408}"
export FLASHVSR_PROF_FRAMES="${FLASHVSR_PROF_FRAMES:-81}"
export FLASHVSR_PROF_WARMUP=1
export FLASHVSR_PROF_STEADY=off
if [[ -z "${FLASHVSR_BENCH_STEADY+x}" ]]; then
  N_CHUNKS=$(((FLASHVSR_PROF_FRAMES - 1) / 8 - 2))
  if ((N_CHUNKS <= 3)); then
    export FLASHVSR_BENCH_STEADY="$((N_CHUNKS - 1)):$N_CHUNKS"
  else
    export FLASHVSR_BENCH_STEADY="3:8"
  fi
else
  export FLASHVSR_BENCH_STEADY
fi
export FLASHVSR_TELEMETRY=1
export FLASHVSR_NVTX=0
export FLASHVSR_REQUIRE_FASTPATHS="${FLASHVSR_REQUIRE_FASTPATHS:-0}"

# --- Per-invocation overrides ---
for kv in "$@"; do
  export "${kv?}"
done

OUT="$HERE/runs/phase7"
mkdir -p "$OUT"

declare -a FPS_LIST=()
for i in $(seq 1 "$NRUNS"); do
  LOG="$OUT/${LABEL}_run${i}.log"
  "$PY" "$HERE/run_pipe_target.py" >"$LOG" 2>&1 || { echo "RUN FAILED, tail:"; tail -20 "$LOG"; exit 1; }
  FPS=$(grep -o '"fps": [0-9.]*' "$LOG" | head -1 | cut -d' ' -f2)
  STEADY=$(grep -o '"steady_median_ms": [0-9.]*' "$LOG" | head -1 | cut -d' ' -f2)
  PEAK=$(grep -o '"peak_gib": [0-9.]*' "$LOG" | head -1 | cut -d' ' -f2)
  FB=$(grep -c '\[fallbacks\] {}' "$LOG" || true)
  echo "[$LABEL run$i] fps=$FPS steady_median=$STEADY ms peak=$PEAK GiB fallbacks_clean=$FB"
  FPS_LIST+=("$FPS")
done

echo "${FPS_LIST[@]}" | tr ' ' '\n' | sort -n | awk '
  {a[NR]=$1} END {
    m = (NR%2) ? a[(NR+1)/2] : (a[NR/2]+a[NR/2+1])/2
    printf "[%s] MEDIAN FPS over %d runs: %s\n", "'"$LABEL"'", NR, m
  }'
