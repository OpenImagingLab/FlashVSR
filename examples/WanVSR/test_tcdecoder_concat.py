#!/usr/bin/env python3
"""Isolated exactness and speed tests for recurrent channels-last concat."""

import os
import sys
import time

import torch


HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from utils.triton_tcdecoder_ops import concat_channels_last  # noqa: E402


def bench(fn, warmup=5, iterations=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / iterations


def check(shape):
    current = torch.randn(
        shape, device="cuda", dtype=torch.bfloat16).contiguous(
            memory_format=torch.channels_last)
    past = torch.randn_like(current).contiguous(memory_format=torch.channels_last)
    reference = torch.cat([current, past], dim=1)
    candidate = concat_channels_last(current, past)
    exact = torch.equal(reference.view(torch.int16), candidate.view(torch.int16))
    native_ms = bench(lambda: torch.cat([current, past], dim=1))
    triton_ms = bench(lambda: concat_channels_last(current, past))
    print(f"shape={shape} exact={exact} native={native_ms:.3f}ms "
          f"triton={triton_ms:.3f}ms speedup={native_ms / triton_ms:.2f}x")
    return exact


def main():
    ok = True
    ok &= check((1, 512, 176, 96))
    ok &= check((1, 256, 352, 192))
    ok &= check((1, 128, 704, 384))
    print(f"RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
