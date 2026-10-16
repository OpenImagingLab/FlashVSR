#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Isolated parity + speed test for TCDecoder channels_last (NHWC) path.

Builds the v1.1 TCDecoder, runs the same latents through it with channels_last
ON vs OFF, and checks output parity (cos) and decode speed.

Run from examples/WanVSR/ :
    python test_tcdecoder_channels_last.py
"""
import os, time, importlib.util
import torch
import torch.nn.functional as F

DEV = "cuda"
DT = torch.bfloat16


def load_tcdec(enabled):
    os.environ["FLASHVSR_TCDECODER_CHANNELS_LAST"] = "1" if enabled else "0"
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(f"tcdec_{enabled}", os.path.join(here, "utils", "TCDecoder.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def build(mod, ckpt):
    ch = [512, 256, 128, 128]
    dec = mod.build_tcdecoder(new_channels=ch, new_latent_channels=16 + 768)
    sd = torch.load(ckpt, map_location="cpu")
    dec.load_state_dict(sd, strict=False)
    dec.to(DEV, DT).eval()
    return dec


def run(dec, latents, cond):
    dec.clean_mem()
    with torch.no_grad():
        out = dec.decode_video(latents.transpose(1, 2), parallel=False, show_progress_bar=False, cond=cond).transpose(1, 2)
    return out.float().cpu()


def bench(fn, it=5, warm=2):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(it): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / it


def main():
    ckpt = "./FlashVSR-v1.1/TCDecoder.ckpt"
    # Latent for 768x1408 output: /8 spatial roughly -> use latent ~ (16, T, 176, 88)? TCDecoder upsamples x8.
    # Use a modest T to keep it quick; shapes only need to be self-consistent.
    T = 8
    H, W = 768 // 8, 1408 // 8  # 96 x 176
    latents = torch.randn(1, 16, T, H, W, device=DEV, dtype=DT)
    # cond is LQ video at full res, pixel-shuffled inside; provide full-res cond frames
    cond = torch.randn(1, 3, T * 4, 768, 1408, device=DEV, dtype=DT)

    print("Building TCDecoder (channels_last OFF)...")
    m_off = load_tcdec(False)
    dec_off = build(m_off, ckpt)
    print("Building TCDecoder (channels_last ON)...")
    m_on = load_tcdec(True)
    dec_on = build(m_on, ckpt)
    dec_on.load_state_dict(dec_off.state_dict())  # identical weights

    out_off = run(dec_off, latents, cond)
    out_on = run(dec_on, latents, cond)

    a, b = out_off.flatten().double(), out_on.flatten().double()
    c = (a @ b / (a.norm() * b.norm())).item()
    md = (out_off - out_on).abs().max().item()
    print(f"\nparity: cos={c:.6f}  max|diff|={md:.5f}  shapes {tuple(out_off.shape)} / {tuple(out_on.shape)}")

    dt_off = bench(lambda: run(dec_off, latents, cond))
    dt_on = bench(lambda: run(dec_on, latents, cond))
    print(f"decode: OFF {dt_off*1e3:7.1f} ms   ON {dt_on*1e3:7.1f} ms   speedup {dt_off/dt_on:.2f}x")
    print("RESULT:", "PASS" if c >= 0.999 else f"CHECK (cos={c:.4f})")


if __name__ == "__main__":
    main()
