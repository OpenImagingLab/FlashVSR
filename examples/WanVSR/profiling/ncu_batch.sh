#!/usr/bin/env bash
# Sequential ncu deep-dive batch with progress heartbeat.
# Progress: profiling/reports/ncu/BATCH_STATUS (one line per state change).
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/.."
ST="$HERE/reports/ncu/BATCH_STATUS"
mkdir -p "$HERE/reports/ncu"
: > "$ST"

log() { echo "$(date +%H:%M:%S) $*" >> "$ST"; }

KNOBS="FLASHVSR_CONV3D_BACKEND=gemm FLASHVSR_TCDECODER_CHANNELS_LAST=1 \
FLASHVSR_FUSE_NORM=1 FLASHVSR_CACHE_MOD=1 FLASHVSR_CACHE_MASK_BIAS=1 \
FLASHVSR_NVTX=1 FLASHVSR_ATTN_BACKEND=triton FLASHVSR_PROF_WARMUP=0"

run() {
  local tag="$1"; shift
  if [ -s "$HERE/reports/ncu/$tag.ncu-rep" ]; then
    log "SKIP  $tag (report exists)"
    return 0
  fi
  log "START $tag"
  if env $KNOBS "$@" "$HERE/ncu_run.sh" "$tag" "${NCU_ARGS[@]}" > /dev/null 2>&1; then
    log "DONE  $tag ($(du -h "$HERE/reports/ncu/$tag.ncu-rep" 2>/dev/null | cut -f1))"
  else
    log "FAIL  $tag (see $tag.log)"
  fi
}

NCU_ARGS=(--set full -k "regex:_bsfa" --launch-skip 6 --launch-count 4)
run attn_bsfa_tma0 FLASHVSR_ATTN_TMA=0

NCU_ARGS=(--set full -k "regex:nvjet" --launch-skip 40 --launch-count 24)
run gemms

NCU_ARGS=(--set detailed -k "regex:elementwise_kernel|vectorized|triton_poi|triton_red|layer_norm|reduce_kernel|CatArrayBatchedCopy" --launch-skip 60 --launch-count 30)
run elemwise

NCU_ARGS=(--set detailed -k "regex:gatherTopK|RadixSort|radixSortKV|softmax_warp|scatter_gather" --launch-skip 16 --launch-count 16)
run masktopk

NCU_ARGS=(--set detailed --nvtx --nvtx-include "decode/" -k "regex:sm90_xmma|elementwise|CatArray|upsample" --launch-skip 20 --launch-count 30)
run decoder FLASHVSR_PROF_STEADY=0:-1

log "BATCH COMPLETE"
