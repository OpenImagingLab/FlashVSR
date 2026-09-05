#!/usr/bin/env python3
"""Isolated Phase-7-C quality and speed gate for MemBlock split-K conv."""

import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

os.environ.update({
    "FLASHVSR_TCDECODER_CHANNELS_LAST": "1",
    "FLASHVSR_TCDECODER_CONCAT": "1",
    "FLASHVSR_TCDECODER_CUDNN_FUSED": "1",
    "FLASHVSR_TCDECODER_SPLITK_CONV": "1",
})

from utils import TCDecoder as tcdec  # noqa: E402


def bench(fn, iterations=30):
    for _ in range(6):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / iterations


def psnr(reference, candidate):
    mse = (reference.float() - candidate.float()).square().mean().item()
    return float("inf") if mse == 0.0 else 10.0 * torch.log10(
        torch.tensor(4.0 / mse)).item()


def check(channels, height, width):
    block = tcdec.MemBlock(channels, channels).cuda().to(torch.bfloat16).eval()
    block.to(memory_format=torch.channels_last)
    x = torch.randn((1, channels, height, width), device="cuda", dtype=torch.bfloat16)
    x = x.contiguous(memory_format=torch.channels_last)
    past = torch.randn_like(x).contiguous(memory_format=torch.channels_last)

    tcdec._TCDEC_SPLITK_CONV = False
    reference = block(x, past)
    tcdec._TCDEC_SPLITK_CONV = True
    tcdec._TCDEC_SPLITK_CONV_FAILED = False
    candidate = block(x, past)
    torch.cuda.synchronize()
    tcdec._TCDEC_SPLITK_CONV = False
    native_ms = bench(lambda: block(x, past))
    tcdec._TCDEC_SPLITK_CONV = True
    split_ms = bench(lambda: block(x, past))
    exact = torch.equal(reference, candidate)
    quality = psnr(reference, candidate)
    print(f"shape={(channels, height, width)} exact={exact} psnr={quality:.2f} dB "
          f"merged={native_ms:.3f}ms split={split_ms:.3f}ms "
          f"speedup={native_ms / split_ms:.2f}x")
    return quality >= 49.0 and split_ms < native_ms


def main():
    torch.manual_seed(0)
    ok = all(check(*shape) for shape in ((512, 176, 96), (256, 352, 192), (128, 704, 384)))
    print(f"RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
