"""Knob-gated NVTX profiling helpers (FLASHVSR_NVTX=1, default OFF).

When FLASHVSR_NVTX is unset/0, ``nvtx_range`` returns a shared null context
manager: zero GPU effect and negligible CPU cost, so the default path is
behaviourally identical to the un-instrumented code.

When FLASHVSR_NVTX=1, ``nvtx_range(name)`` emits an NVTX range visible in
Nsight Systems / Nsight Compute (used for phase attribution and kernel
filtering, e.g. ``ncu --nvtx --nvtx-include``).
"""
import os

import torch

NVTX_ENABLED = os.environ.get("FLASHVSR_NVTX", "0") != "0"


class _NullCtx:
    __slots__ = ()

    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


_NULL = _NullCtx()

if NVTX_ENABLED:
    def nvtx_range(name: str):
        return torch.cuda.nvtx.range(name)
else:
    def nvtx_range(name: str):  # noqa: ARG001 - keep signature identical
        return _NULL
