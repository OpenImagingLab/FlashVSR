#!/usr/bin/env python3
"""Exactness and isolated speed tests for the Triton LQ im2col packer."""

import os
import sys
import time

import torch


HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from utils.triton_lq_im2col import im2col3d  # noqa: E402


def eager(x, kernel, stride, h0, rows):
    kt, kh, kw = kernel
    st, sh, sw = stride
    xs = x[:, :, :, h0 * sh:(h0 + rows - 1) * sh + kh, :]
    patches = xs.unfold(2, kt, st).unfold(3, kh, sh).unfold(4, kw, sw)
    n, cin = x.shape[:2]
    to, ho, wo = patches.shape[2:5]
    return patches.permute(0, 2, 3, 4, 1, 5, 6, 7).reshape(
        n * to * ho * wo, cin * kt * kh * kw)


def bench(fn, warmup=3, iterations=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / iterations


def check(shape, kernel=(4, 3, 3), stride=(2, 1, 1)):
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    h_rows = shape[3] - 2
    reference = eager(x, kernel, stride, 0, h_rows)
    candidate = im2col3d(x, kernel, stride, 0, h_rows)
    exact = torch.equal(reference.view(torch.int16), candidate.view(torch.int16))
    tile_h0 = min(17, h_rows // 2)
    tile_rows = min(23, h_rows - tile_h0)
    tile_reference = eager(x, kernel, stride, tile_h0, tile_rows)
    tile_candidate = im2col3d(x, kernel, stride, tile_h0, tile_rows)
    tiled_exact = torch.equal(
        tile_reference.view(torch.int16), tile_candidate.view(torch.int16))
    eager_ms = bench(lambda: eager(x, kernel, stride, 0, h_rows))
    triton_ms = bench(lambda: im2col3d(x, kernel, stride, 0, h_rows))
    print(f"shape={shape} exact={exact} tiled_exact={tiled_exact} "
          f"eager={eager_ms:.3f}ms "
          f"triton={triton_ms:.3f}ms speedup={eager_ms / triton_ms:.2f}x")
    return exact and tiled_exact


def main():
    ok = check((1, 768, 6, 90, 50))
    ok &= check((1, 2048, 4, 90, 50))
    print(f"RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
