#!/usr/bin/env python3
"""Isolated exactness tests for TCDecoder fused pointwise kernels."""

import os
import sys

import torch


HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from utils.triton_tcdecoder_ops import bias_relu, bias_residual_relu  # noqa: E402


def bits(x):
    return x.view(torch.int16)


def reference_bias_relu(x, bias):
    return torch.relu(x + bias.view(1, -1, 1, 1))


def reference_bias_residual_relu(x, bias, residual):
    return torch.relu(x + bias.view(1, -1, 1, 1) + residual)


def check(shape):
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16).contiguous(
        memory_format=torch.channels_last)
    residual = torch.randn_like(x).contiguous(memory_format=torch.channels_last)
    bias = torch.randn(shape[1], device="cuda", dtype=torch.bfloat16)
    ref1 = reference_bias_relu(x, bias)
    got1 = bias_relu(x, bias)
    ref2 = reference_bias_residual_relu(x, bias, residual)
    got2 = bias_residual_relu(x, bias, residual)
    ok1 = torch.equal(bits(ref1), bits(got1))
    ok2 = torch.equal(bits(ref2), bits(got2))
    print(f"shape={shape} bias_relu={ok1} bias_residual_relu={ok2}")
    return ok1 and ok2


def check_special_values():
    values = torch.tensor(
        [-float("inf"), -1.0, -0.0, 0.0, 1.0, float("inf"), float("nan")],
        device="cuda", dtype=torch.bfloat16).view(1, 7, 1, 1)
    values = values.contiguous(memory_format=torch.channels_last)
    bias = torch.zeros(7, device="cuda", dtype=torch.bfloat16)
    residual = torch.zeros_like(values)
    return (
        torch.equal(bits(reference_bias_relu(values, bias)),
                    bits(bias_relu(values, bias)))
        and torch.equal(bits(reference_bias_residual_relu(values, bias, residual)),
                        bits(bias_residual_relu(values, bias, residual)))
    )


def main():
    ok = check_special_values()
    ok &= check((1, 512, 176, 96))
    ok &= check((1, 256, 352, 192))
    ok &= check((1, 128, 704, 384))
    print(f"RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
