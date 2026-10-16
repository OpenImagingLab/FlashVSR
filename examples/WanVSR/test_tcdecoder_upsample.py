#!/usr/bin/env python3
"""Isolated exactness and speed tests for the Triton nearest upsampler."""

import os
import sys
import time

import torch
import torch.nn.functional as F


HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from utils.triton_tcdecoder_ops import upsample2x_channels_last  # noqa: E402


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
    source = torch.randn(
        shape, device="cuda", dtype=torch.bfloat16).contiguous(
            memory_format=torch.channels_last)
    reference = F.interpolate(source, scale_factor=2, mode="nearest")
    candidate = upsample2x_channels_last(source)
    exact = torch.equal(reference.view(torch.int16), candidate.view(torch.int16))
    native_ms = bench(lambda: F.interpolate(source, scale_factor=2, mode="nearest"))
    triton_ms = bench(lambda: upsample2x_channels_last(source))
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
