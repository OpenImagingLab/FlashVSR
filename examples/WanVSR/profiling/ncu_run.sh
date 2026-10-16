#!/usr/bin/env bash
# Nsight Compute driver for FlashVSR kernel deep dives.
#
# Usage:
#   ./ncu_run.sh <tag> [extra ncu args...]
#
# Defaults: short workload (F=45 -> chunks 0,1,2), steady window = chunk2 only,
# --clock-control none (we lock clocks globally), --profile-from-start off
# (cudaProfilerStart fires at the steady chunk). Knobs/workload via env as with
# nsys_run.sh.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PY="/root/FlashVSR/venv/bin/python"
NCU="${NCU:-ncu}"

TAG="${1:?usage: ncu_run.sh <tag> [extra ncu args...]}"
shift || true

OUT="$HERE/reports/ncu"
mkdir -p "$OUT"

export FLASHVSR_PROF_FRAMES="${FLASHVSR_PROF_FRAMES:-45}"
export FLASHVSR_PROF_STEADY="${FLASHVSR_PROF_STEADY:-2:3}"

# Deadlock fix: triton's libcuda discovery spawns `ldconfig` via subprocess;
# ncu child-process injection can deadlock that handshake. Point triton at
# libcuda directly so no subprocess is spawned at all.
export TRITON_LIBCUDA_PATH="${TRITON_LIBCUDA_PATH:-/usr/lib/aarch64-linux-gnu}"

env | grep -E '^FLASHVSR' | sort > "$OUT/$TAG.env.txt" || true

set +e
"$NCU" \
  --clock-control none \
  --profile-from-start off \
  --force-overwrite \
  -o "$OUT/$TAG" \
  "$@" \
  "$PY" "$HERE/run_pipe_target.py" 2>&1 | tee "$OUT/$TAG.log" | tail -5
RC=${PIPESTATUS[0]}
set -e
echo "[ncu_run] tag=$TAG rc=$RC report=$OUT/$TAG.ncu-rep"
exit "$RC"
