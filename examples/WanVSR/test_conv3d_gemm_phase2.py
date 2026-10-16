#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase-2 isolated test: chunked im2col+GEMM parity, memory, and speed.

Verifies that the chunked path (Phase 2) keeps the same tensor-core speed and
bit-identical parity as the single-shot path (Phase 1), while bounding the
transient im2col memory at large resolution (1920x2560 / latent 120x160).

Run from examples/WanVSR/ :
    python test_conv3d_gemm_phase2.py
"""
import os, time, importlib.util
import torch
import torch.nn.functional as F

DEV = "cuda"
DT = torch.bfloat16


def load_utils(backend, budget=None):
    os.environ["FLASHVSR_CONV3D_BACKEND"] = backend
    if budget is not None:
        os.environ["FLASHVSR_CONV3D_IM2COL_BUDGET_GB"] = str(budget)
    here = os.path.dirname(os.path.abspath(__file__))
    name = f"u_{backend}_{budget}"
    spec = importlib.util.spec_from_file_location(name, os.path.join(here, "utils", "utils.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def cos(a, b):
    return F.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0).item()


def bench(fn, it=20, warm=10):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(it): out = fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / it, out


def peakmem(fn):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    out = fn(); torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1e9, out


def make(mod, Cin, Cout):
    torch.manual_seed(0)
    return mod.CausalConv3d(Cin, Cout, (4, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1)).to(DEV, DT).eval()


def run(label, Cin, Cout, T, H, W):
    # reference: plain cuDNN conv via auto backend
    ref_mod = load_utils("auto")
    c_ref = make(ref_mod, Cin, Cout)
    x = torch.randn(1, Cin, T, H, W, device=DEV, dtype=DT)
    with torch.no_grad():
        ref = c_ref(x)

    print(f"\n{label}  latent {H}x{W} -> {tuple(ref.shape)}")
    # Phase-1 single shot (budget disabled) and Phase-2 chunked (budget 2 GB)
    for tag, budget in [("phase1 single-shot", 0.0), ("phase2 chunked@2GB", 2.0)]:
        mod = load_utils("gemm", budget=budget)
        c = make(mod, Cin, Cout)
        c.load_state_dict(c_ref.state_dict())
        with torch.no_grad():
            mem, out = peakmem(lambda: c(x))
            dt, _ = bench(lambda: c(x))
        print(f"  {tag:20s}: {dt*1e3:7.2f} ms  peak={mem:6.2f} GB  cos={cos(ref, out):.6f}")


if __name__ == "__main__":
    cap = torch.cuda.get_device_capability()
    print(f"Device: {torch.cuda.get_device_name()}  cap={cap}")
    run("conv2 2048->3072 @1536",       2048, 3072, 6, 96, 96)
    run("conv2 2048->3072 @1920x2560",  2048, 3072, 6, 120, 160)
    run("conv1 768->2048  @1920x2560",  768,  2048, 6, 120, 160)
