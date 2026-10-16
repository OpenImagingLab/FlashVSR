#!/usr/bin/env python3
"""Exactness and isolated speed tests for the fused TGrow->upsample reorder.

Native reference (production path today):
    Triton nearest-upsample (high res) -> 1x1 TGrow conv (high res)
        -> temporal channel-group split
Fused candidate:
    1x1 TGrow conv (low res) -> fused temporal-unpack + nearest-upsample

Nearest duplication commutes exactly with a pointwise conv, so values are
mathematically identical; only the cuDNN reduction split may differ between
the two spatial sizes. This test reports both bitwise equality and PSNR
(quality budget: >= 49 dB E2E; isolated expectation is far above).
"""

import math
import os
import sys
import time

import torch
import torch.nn.functional as F


HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from utils.triton_tcdecoder_ops import (  # noqa: E402
    tgrow_upsample2x_channels_last,
    upsample2x_channels_last,
)


def bench(fn, warmup=5, iterations=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / iterations


def psnr(reference, candidate):
    ref = reference.double()
    mse = (ref - candidate.double()).pow(2).mean().item()
    if mse == 0.0:
        return float("inf")
    peak = ref.abs().max().item()
    return 10.0 * math.log10(peak * peak / mse)


def native(x, weight, stride):
    up = upsample2x_channels_last(x)
    y = F.conv2d(up, weight)
    frames = [chunk.contiguous(memory_format=torch.channels_last)
              for chunk in y.chunk(stride, 1)]
    return torch.cat(frames, 0)


def fused(x, weight, stride):
    grown = F.conv2d(x, weight)
    return tgrow_upsample2x_channels_last(grown, stride)


def check(channels, stride, height, width):
    torch.manual_seed(0)
    x = torch.randn((1, channels, height, width), device="cuda",
                    dtype=torch.bfloat16).contiguous(
                        memory_format=torch.channels_last)
    weight = (torch.randn((channels * stride, channels, 1, 1), device="cuda",
                          dtype=torch.bfloat16) / math.sqrt(channels))
    weight = weight.contiguous(memory_format=torch.channels_last)

    reference = native(x, weight, stride)
    candidate = fused(x, weight, stride)
    exact = torch.equal(reference, candidate)
    max_diff = (reference.float() - candidate.float()).abs().max().item()
    quality = psnr(reference, candidate)

    native_ms = bench(lambda: native(x, weight, stride))
    fused_ms = bench(lambda: fused(x, weight, stride))
    ok = exact or quality >= 60.0
    print(f"C={channels:3d} S={stride} {height}x{width}: exact={exact} "
          f"max|diff|={max_diff:.3e} psnr={quality:.1f}dB "
          f"native={native_ms:.3f}ms fused={fused_ms:.3f}ms "
          f"speedup={native_ms / fused_ms:.2f}x [{'OK' if ok else 'FAIL'}]")
    return ok


def main():
    ok = check(512, 1, 176, 96)
    ok &= check(256, 2, 352, 192)
    ok &= check(128, 2, 704, 384)
    # Large-resolution spot (1536x2560 output -> stage sizes x2)
    ok &= check(256, 2, 640, 384)
    print(f"RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
