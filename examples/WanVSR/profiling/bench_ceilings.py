#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase-3 ceiling microbenchmarks @768x1408 real shapes (GH200).

H2: dense cuDNN SDPA reference at the exact attention shape -> how fast could
    an "ideal" kernel be (density-scaled), vs the measured _bsfa 2.03 ms.
H4: bf16 vs fp8 (torch._scaled_mm) GEMM at the DiT shapes -> FP8 ceiling.
H3: im2col+GEMM split for the LQ-projector conv2 + cuDNN 9.22 conv3d check.

All shapes match the steady-state 768x1408 run:
  q tokens 8448 (66 blocks x 128), kv 25344 (3 chunks), 12 heads, d=128,
  block-mask density ~0.606; DiT dim 1536, ffn 8960; conv2 2048->3072
  k=(4,3,3) s=(2,1,1) input (1,2048,4,88,48) + cache 2 frames.
"""
import os
import sys
import time

os.environ.setdefault("FLASHVSR_CONV3D_BACKEND", "gemm")
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_here))

import torch
import torch.nn.functional as F

DEV = "cuda"
DT = torch.bfloat16


def bench(fn, it=30, warm=10):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(it):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / it * 1e3  # ms


def sec(title):
    print(f"\n=== {title} ===")


def h2_attention():
    sec("H2: attention reference (q 8448, kv 25344, h12, d128)")
    q = torch.randn(1, 12, 8448, 128, device=DEV, dtype=DT)
    k = torch.randn(1, 12, 25344, 128, device=DEV, dtype=DT)
    v = torch.randn_like(k)
    from torch.nn.attention import sdpa_kernel, SDPBackend

    def dense():
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            return F.scaled_dot_product_attention(q, k, v)

    t_dense = bench(dense)
    flops_dense = 2 * 2 * 12 * 8448 * 25344 * 128  # qk + pv
    print(f"cuDNN dense SDPA : {t_dense:.3f} ms  ({flops_dense/t_dense/1e9:.0f} TFLOP/s)")
    density = 0.606
    print(f"ideal sparse (dense x {density}) : {t_dense*density:.3f} ms")
    print(f"measured _bsfa_tma : 2.03 ms -> efficiency vs ideal-sparse "
          f"{t_dense*density/2.03*100:.0f}%")
    try:
        import flash_attn_interface  # noqa: F401
        has_fa3 = True
    except Exception:
        has_fa3 = False
    print(f"flash_attn_3 available: {has_fa3}")


def h4_gemm_fp8():
    sec("H4: bf16 vs FP8 GEMM at DiT shapes (M=8448)")
    M = 8448
    shapes = [("qkv/o 1536x1536", 1536, 1536),
              ("ffn1 1536x8960", 1536, 8960),
              ("ffn2 8960x1536", 8960, 1536),
              ("lq_lin 3072x1536", 3072, 1536)]
    for name, K, N in shapes:
        a = torch.randn(M, K, device=DEV, dtype=DT)
        b = torch.randn(N, K, device=DEV, dtype=DT)
        t_bf16 = bench(lambda: a @ b.t())
        a8 = a.to(torch.float8_e4m3fn)
        b8 = b.to(torch.float8_e4m3fn)
        sa = torch.ones(M, 1, device=DEV)
        sb = torch.ones(1, N, device=DEV)

        def fp8():
            return torch._scaled_mm(a8, b8.t(), scale_a=sa, scale_b=sb,
                                    out_dtype=DT)
        try:
            t_fp8 = bench(fp8)
            fl = 2 * M * K * N
            print(f"{name:18s} bf16 {t_bf16*1e3:7.1f} us ({fl/t_bf16/1e9:5.0f} TF/s) | "
                  f"fp8 {t_fp8*1e3:7.1f} us ({fl/t_fp8/1e9:5.0f} TF/s) | x{t_bf16/t_fp8:.2f}")
        except Exception as e:
            print(f"{name:18s} bf16 {t_bf16*1e3:7.1f} us | fp8 FAILED: {e}")


def h3_conv():
    sec("H3: LQ conv2 im2col+GEMM split (2048->3072, in (1,2048,4,88,48)+cache2)")
    from utils.utils import CausalConv3d
    conv = CausalConv3d(2048, 3072, (4, 3, 3), stride=(2, 1, 1),
                        padding=(1, 1, 1)).to(DEV, DT).eval()
    x = torch.randn(1, 2048, 4, 88, 48, device=DEV, dtype=DT)
    cache = torch.randn(1, 2048, 2, 88, 48, device=DEV, dtype=DT)

    with torch.no_grad():
        t_gemm_path = bench(lambda: conv(x, cache))

        # split: pad+cat / unfold(im2col) / addmm
        w = conv.weight
        b = conv.bias
        pad = conv._padding

        def prep():
            xx = torch.cat([cache, x], dim=2)
            p = list(pad)
            p[4] = max(0, p[4] - cache.shape[2])
            return F.pad(xx, p, mode="replicate")

        xp = prep()

        def im2col():
            cols = xp.unfold(2, 4, 2).unfold(3, 3, 1).unfold(4, 3, 1)
            return cols.permute(0, 2, 3, 4, 1, 5, 6, 7).reshape(-1, 2048 * 36)

        cols = im2col().contiguous()
        wmat = w.permute(0, 1, 2, 3, 4).reshape(3072, -1)

        # weight layout for addmm: patches are (Cin,kt,kh,kw) flattened
        def gemm_only():
            return torch.addmm(b, cols, wmat.t())

        t_prep = bench(prep)
        t_im2col = bench(lambda: im2col().contiguous())
        t_gemm = bench(gemm_only)
        fl = 2 * cols.shape[0] * cols.shape[1] * 3072
        print(f"full gemm path : {t_gemm_path:.3f} ms")
        print(f"  pad+cat      : {t_prep:.3f} ms")
        print(f"  im2col copy  : {t_im2col:.3f} ms")
        print(f"  addmm only   : {t_gemm:.3f} ms ({fl/t_gemm/1e9:.0f} TFLOP/s)")
        print(f"  im2col share of path: {t_im2col/t_gemm_path*100:.0f}%")

        # cuDNN 9.22 reference (the 'auto' backend path)
        conv_ref = torch.nn.Conv3d(2048, 3072, (4, 3, 3), stride=(2, 1, 1)).to(DEV, DT).eval()
        xx = prep()
        t_cudnn = bench(lambda: conv_ref(xx))
        print(f"cuDNN conv3d   : {t_cudnn:.3f} ms  (gemm path is x{t_cudnn/t_gemm_path:.1f} faster)")


def main():
    torch.manual_seed(0)
    print(f"device: {torch.cuda.get_device_name(0)}, torch {torch.__version__}")
    h2_attention()
    h4_gemm_fp8()
    h3_conv()


if __name__ == "__main__":
    main()
