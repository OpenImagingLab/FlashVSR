#!/usr/bin/env bash
# nsys profiling driver for FlashVSR v1.1 Tiny.
#
# Usage:
#   ./nsys_run.sh <tag> [extra nsys args...]
#
# Workload/knobs come from the environment (set them before calling), e.g.:
#   FLASHVSR_CONV3D_BACKEND=gemm FLASHVSR_TCDECODER_CHANNELS_LAST=1 \
#   FLASHVSR_FUSE_NORM=1 FLASHVSR_ATTN_BACKEND=triton FLASHVSR_NVTX=1 \
#   FLASHVSR_PROF_W=768 FLASHVSR_PROF_H=1408 \
#   ./nsys_run.sh fullknobs_768 --gpu-metrics-devices=0 --gpu-metrics-frequency=20000
#
# Produces reports/<tag>/{profile.nsys-rep,run.log,dmon.log,env.txt,gpu_state_*.csv}
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PY="/root/FlashVSR/venv/bin/python"
NSYS="${NSYS:-nsys}"

TAG="${1:?usage: nsys_run.sh <tag> [extra nsys args...]}"
shift || true

OUT="$HERE/reports/$TAG"
mkdir -p "$OUT"

env | grep -E '^FLASHVSR' | sort > "$OUT/env.txt" || true
nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.mem,power.limit,temperature.gpu \
  --format=csv > "$OUT/gpu_state_pre.csv"

# Power/clock/util/mem logger (1 Hz) for throttle & residency analysis.
nvidia-smi dmon -s pucm -d 1 -o T > "$OUT/dmon.log" 2>&1 &
DMON=$!
trap 'kill "$DMON" 2>/dev/null || true' EXIT

# Trace set: minimal 'cuda,nvtx' by default (lowest CPU overhead -> least
# distortion for gap analysis); set NSYS_TRACE=cuda,nvtx,osrt,cudnn,cublas
# for the rich API-attribution run.
TRACE="${NSYS_TRACE:-cuda,nvtx}"

set +e
"$NSYS" profile \
  -t "$TRACE" \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --cuda-memory-usage=true \
  --force-overwrite=true \
  -o "$OUT/profile" \
  "$@" \
  "$PY" "$HERE/run_pipe_target.py" 2>&1 | tee "$OUT/run.log"
RC=${PIPESTATUS[0]}
set -e

kill "$DMON" 2>/dev/null || true
nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.mem,power.limit,temperature.gpu \
  --format=csv > "$OUT/gpu_state_post.csv"

echo "[nsys_run] tag=$TAG rc=$RC report=$OUT/profile.nsys-rep"
exit "$RC"
