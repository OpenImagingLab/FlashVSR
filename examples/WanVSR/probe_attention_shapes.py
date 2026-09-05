#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Capture real self-attention shapes + sparsity in the v1.1 Tiny DiT @768x1408.

Hooks block_sparse_attn_func / generate_draft_block_mask to record q/k/v shapes,
block counts, and the fraction of KV blocks actually attended (sparsity).

Run from examples/WanVSR/ :
    python probe_attention_shapes.py
"""
import os, importlib.util
import numpy as np
from PIL import Image
import imageio
import torch

os.environ["FLASHVSR_CONV3D_BACKEND"] = "gemm"
import utils.utils as wanutils; wanutils._CONV3D_BACKEND = "gemm"

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("infer_v1_1_tiny", os.path.join(_here, "infer_flashvsr_v1.1_tiny.py"))
_infer = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_infer)
init_pipeline = _infer.init_pipeline; largest_8n1_leq = _infer.largest_8n1_leq

import diffsynth.models.wan_video_dit as dit

REF_W, REF_H, SCALE = 768, 1408, 4
SRC_W, SRC_H = REF_W // SCALE, REF_H // SCALE

records = []
_orig_bsa = dit.block_sparse_attn_func
def bsa_hook(q, k, v, cu_q, cu_k, head_mask_type, streaming_info, base_blockmask, *a, **kw):
    m = base_blockmask
    rec = {
        "q": tuple(q.shape), "k": tuple(k.shape), "v": tuple(v.shape),
        "mask": tuple(m.shape) if hasattr(m, "shape") else None,
        "mask_density": float(m.float().mean().item()) if hasattr(m, "float") else None,
    }
    if len(records) < 6:
        records.append(rec)
    return _orig_bsa(q, k, v, cu_q, cu_k, head_mask_type, streaming_info, base_blockmask, *a, **kw)
dit.block_sparse_attn_func = bsa_hook


def build_lq(src, device="cuda", dtype=torch.bfloat16):
    rdr = imageio.get_reader(src); total = rdr.count_frames()
    idx = (list(range(total)) + [total - 1] * 4); F = largest_8n1_leq(len(idx)); idx = idx[:F]
    frames = []
    for i in idx:
        img = Image.fromarray(rdr.get_data(i)).convert("RGB").resize((SRC_W, SRC_H), Image.BICUBIC).resize((REF_W, REF_H), Image.BICUBIC)
        t = torch.from_numpy(np.asarray(img, np.uint8)).to(device=device, dtype=torch.float32)
        frames.append((t.permute(2, 0, 1) / 255.0 * 2.0 - 1.0).to(dtype))
    rdr.close()
    return torch.stack(frames, 0).permute(1, 0, 2, 3).unsqueeze(0), F


def main():
    pipe = init_pipeline()
    LQ, F = build_lq("./inputs/example0.mp4")
    th, tw = REF_H, REF_W
    with torch.no_grad():
        pipe(prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=1, seed=0,
             LQ_video=LQ, num_frames=F, height=th, width=tw, is_full_block=False, if_buffer=True,
             topk_ratio=2.0 * 768 * 1280 / (th * tw), kv_ratio=3.0, local_range=11, color_fix=True)
    print("\n=== Captured self-attention block_sparse calls @768x1408 ===")
    for i, r in enumerate(records):
        print(f"[{i}] q={r['q']} k={r['k']} mask={r['mask']} density={r['mask_density']}")


if __name__ == "__main__":
    main()
