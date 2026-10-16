#!/usr/bin/env python3
"""Isolated parity + speed test for the Hopper im2col+GEMM conv3d backend.

Run from examples/WanVSR/ :
    python test_conv3d_gemm_parity.py

Compares CausalConv3d's standard cuDNN path against the FLASHVSR_CONV3D_BACKEND=gemm
path, including:
  - the causal replicate-pad semantics,
  - the streaming cache (cache_x) path used by the LQ projector,
across several shapes and temporal lengths. Requires cos >= 0.999.
"""
import os
import time
import importlib.util

import torch
import torch.nn.functional as F

DEV = "cuda"
DT = torch.bfloat16


def load_utils(backend):
    """Load utils.py fresh with the given FLASHVSR_CONV3D_BACKEND value."""
    os.environ["FLASHVSR_CONV3D_BACKEND"] = backend
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        f"wanvsr_utils_{backend}", os.path.join(here, "utils", "utils.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cos(a, b):
    return F.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0).item()


def max_abs(a, b):
    return (a.float() - b.float()).abs().max().item()


def make_conv(mod, Cin, Cout, kernel, stride, padding, seed=0):
    torch.manual_seed(seed)
    c = mod.CausalConv3d(Cin, Cout, kernel, stride=stride, padding=padding)
    return c.to(DEV, DT).eval().requires_grad_(False)


def copy_weights(dst, src):
    dst.load_state_dict(src.state_dict())


def run_single(label, Cin, Cout, kernel, stride, padding, T, H, W):
    """Single-call (no cache) parity."""
    base = make_conv(BASE, Cin, Cout, kernel, stride, padding)
    gemm = make_conv(GEMM, Cin, Cout, kernel, stride, padding)
    copy_weights(gemm, base)

    x = torch.randn(1, Cin, T, H, W, device=DEV, dtype=DT)
    with torch.no_grad():
        ob = base(x)
        og = gemm(x)
    print(
        f"  [single] {label:32s} T={T} -> {tuple(ob.shape)}  "
        f"cos={cos(ob, og):.6f}  maxabs={max_abs(ob, og):.4f}"
    )
    return cos(ob, og)


def run_streaming(label, Cin, Cout, kernel, stride, padding, H, W, clips, clip_T):
    """Streaming parity: feed clips sequentially with cache_x, like the LQ projector."""
    base = make_conv(BASE, Cin, Cout, kernel, stride, padding)
    gemm = make_conv(GEMM, Cin, Cout, kernel, stride, padding)
    copy_weights(gemm, base)

    cache_b = None
    cache_g = None
    worst = 1.0
    with torch.no_grad():
        for i in range(clips):
            x = torch.randn(1, Cin, clip_T, H, W, device=DEV, dtype=DT)
            cache_x_b = x[:, :, -2:, :, :].clone()
            cache_x_g = x[:, :, -2:, :, :].clone()
            ob = base(x, cache_b)
            og = gemm(x, cache_g)
            cache_b = cache_x_b
            cache_g = cache_x_g
            c = cos(ob, og)
            worst = min(worst, c)
    print(f"  [stream] {label:32s} clips={clips} clipT={clip_T}  worst-cos={worst:.6f}")
    return worst


def bench(fn, it=20, warm=10):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(it):
        out = fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / it, out


def run_speed(label, Cin, Cout, T, H, W):
    base = make_conv(BASE, Cin, Cout, (4, 3, 3), (2, 1, 1), (1, 1, 1))
    gemm = make_conv(GEMM, Cin, Cout, (4, 3, 3), (2, 1, 1), (1, 1, 1))
    copy_weights(gemm, base)
    x = torch.randn(1, Cin, T, H, W, device=DEV, dtype=DT)
    with torch.no_grad():
        db, ob = bench(lambda: base(x))
        dg, og = bench(lambda: gemm(x))
    To, Ho, Wo = ob.shape[2], ob.shape[3], ob.shape[4]
    flops = 2.0 * Cout * To * Ho * Wo * Cin * 4 * 3 * 3
    print(
        f"  {label:28s} base {db*1e3:8.2f} ms ({flops/db/1e12:6.1f} TF)  "
        f"gemm {dg*1e3:7.2f} ms ({flops/dg/1e12:6.1f} TF)  {db/dg:5.2f}x"
    )


if __name__ == "__main__":
    assert torch.cuda.is_available()
    cap = torch.cuda.get_device_capability()
    print(f"Device: {torch.cuda.get_device_name()}  cap={cap}")
    if cap != (9, 0):
        print("WARNING: not sm_90; gemm backend will be a no-op (guard) -> parity trivially holds")

    BASE = load_utils("auto")
    GEMM = load_utils("gemm")

    print("\n== Single-call parity (causal replicate-pad) ==")
    ok = True
    for (Cin, Cout, T) in [(768, 2048, 6), (2048, 3072, 6), (768, 2048, 2), (32, 64, 6)]:
        c = run_single(f"{Cin}->{Cout}", Cin, Cout, (4, 3, 3), (2, 1, 1), (1, 1, 1), T, 32, 32)
        ok &= c >= 0.999

    print("\n== Streaming-cache parity ==")
    for (Cin, Cout) in [(768, 2048), (2048, 3072)]:
        c = run_streaming(f"{Cin}->{Cout}", Cin, Cout, (4, 3, 3), (2, 1, 1), (1, 1, 1), 32, 32, clips=4, clip_T=4)
        ok &= c >= 0.999

    print("\n== Speed (real LQ-projector shapes, 96x96) ==")
    run_speed("conv1 768->2048", 768, 2048, 6, 96, 96)
    run_speed("conv2 2048->3072", 2048, 3072, 6, 96, 96)

    print("\nRESULT:", "PASS (parity cos>=0.999)" if ok else "FAIL")
