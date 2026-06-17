#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E2E kernel-level bottleneck profiler @ 768x1408 with the gemm conv3d backend.

Goal: after the conv3d acceleration (now 1.86x A100), find where the remaining
denoise time goes so we can push past A100x3. Runs the v1.1 Tiny pipeline under
torch.profiler and aggregates CUDA self-time into categories (attention, GEMM
[linear/qkv/ffn], conv3d-gemm, norm/elementwise, layout/copy, decoder, other).

Run from examples/WanVSR/ :
    python profile_e2e_bottlenecks.py
"""
import os, time, importlib.util, re
import numpy as np
from PIL import Image
import imageio
import torch

os.environ["FLASHVSR_CONV3D_BACKEND"] = "gemm"
import utils.utils as wanutils
wanutils._CONV3D_BACKEND = "gemm"

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("infer_v1_1_tiny", os.path.join(_here, "infer_flashvsr_v1.1_tiny.py"))
_infer = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_infer)
init_pipeline = _infer.init_pipeline
largest_8n1_leq = _infer.largest_8n1_leq

REF_W, REF_H, SCALE = 768, 1408, 4
SRC_W, SRC_H = REF_W // SCALE, REF_H // SCALE


def build_lq(src, device="cuda", dtype=torch.bfloat16):
    rdr = imageio.get_reader(src); total = rdr.count_frames()
    idx = (list(range(total)) + [total - 1] * 4)
    F = largest_8n1_leq(len(idx)); idx = idx[:F]
    frames = []
    for i in idx:
        img = Image.fromarray(rdr.get_data(i)).convert("RGB").resize((SRC_W, SRC_H), Image.BICUBIC).resize((REF_W, REF_H), Image.BICUBIC)
        t = torch.from_numpy(np.asarray(img, np.uint8)).to(device=device, dtype=torch.float32)
        t = t.permute(2, 0, 1) / 255.0 * 2.0 - 1.0
        frames.append(t.to(dtype))
    rdr.close()
    return torch.stack(frames, 0).permute(1, 0, 2, 3).unsqueeze(0), F


def categorize(name):
    n = name.lower()
    # attention kernels (incl. Triton WGMMA block-sparse kernel _bsfa[_tma]_kernel
    # and the bundled block_sparse_attn CUDA kernel)
    if any(k in n for k in ["fmha", "flash", "attention", "softmax", "scaled_dot", "mha",
                            "block_sparse", "_bsfa", "bsfa_"]):
        return "attention"
    # cuDNN convolution (TCDecoder conv2d, etc.), NOT the gemm-conv path.
    # Note sm90_xmma_fprop_implicit_gemm is cuDNN conv (TCDecoder), not a linear GEMM.
    if ("cudnn_convolution" in n or "implicit_gemm" in n or "xmma_fprop" in n
            or "xmma_wgrad" in n or ("conv" in n and "convolution" in n)):
        return "conv (cudnn, TCDecoder)"
    # explicit GEMM (linear/qkv/ffn + our im2col conv -> addmm/nvjet/cublas)
    if any(k in n for k in ["nvjet", "gemm", "cutlass", "wgmma", "cublas", "addmm", "matmul", "linear"]):
        return "gemm (linear/ffn/im2col-conv)"
    # layout conversions (big cost for conv2d in TCDecoder)
    if any(k in n for k in ["nchwtonhwc", "nhwctonchw", "tonhwc", "tonchw"]):
        return "layout (nchw<->nhwc)"
    # normalization / elementwise / activation
    if any(k in n for k in ["norm", "rms", "silu", "gelu", "relu", "elementwise", "mul", "add", "div", "layer_norm", "reduce", "sigmoid"]):
        return "norm/elementwise/act"
    # copy / cat / pad / transpose
    if any(k in n for k in ["copy", "cat", "memcpy", "pad", "transpose", "permute", "contiguous"]):
        return "copy/cat/pad"
    # upsample / interpolation (decoder)
    if any(k in n for k in ["upsample", "interpolate", "grid_sample"]):
        return "decoder-resample"
    return "other"


def main():
    src = os.environ.get("FLASHVSR_TEST_INPUT", "./inputs/example0.mp4")
    pipe = init_pipeline()
    LQ, F = build_lq(src)
    th, tw = REF_H, REF_W
    kwargs = dict(prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=1, seed=0,
                  LQ_video=LQ, num_frames=F, height=th, width=tw, is_full_block=False, if_buffer=True,
                  topk_ratio=2.0 * 768 * 1280 / (th * tw), kv_ratio=3.0, local_range=11, color_fix=True)

    # warmup
    with torch.no_grad():
        pipe(**kwargs)
    torch.cuda.synchronize()

    from torch.profiler import profile, ProfilerActivity
    with torch.no_grad():
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            t0 = time.perf_counter()
            pipe(**kwargs)
            torch.cuda.synchronize()
            wall = time.perf_counter() - t0

    evs = prof.key_averages()
    cats = {}
    total_cuda = 0.0
    for e in evs:
        st = e.self_device_time_total  # us
        if st <= 0:
            continue
        total_cuda += st
        cats.setdefault(categorize(e.key), 0.0)
        cats[categorize(e.key)] += st

    print(f"\n=== E2E bottleneck profile @ {tw}x{th}, gemm backend ===")
    print(f"wall denoise: {wall*1e3:.0f} ms   total CUDA self-time: {total_cuda/1e3:.0f} ms\n")
    print(f"{'category':28s} {'CUDA ms':>10s} {'% of GPU':>9s}")
    for k in sorted(cats, key=lambda x: -cats[x]):
        print(f"{k:28s} {cats[k]/1e3:10.1f} {100*cats[k]/total_cuda:8.1f}%")

    print("\n=== Top 20 individual CUDA kernels ===")
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))


if __name__ == "__main__":
    main()
