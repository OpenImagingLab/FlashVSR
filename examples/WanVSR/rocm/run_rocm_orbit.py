def main():
    import sys
    import os
    import time
    import json
    import argparse
    import importlib.util
    from pathlib import Path
    import torch
    import imageio
    import numpy as np
    from PIL import Image

    root = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(root), str(root / "examples/WanVSR")]
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["tiny"], default="tiny")
    p.add_argument(
        "--variant", choices=["baseline_clean", "optimized"], default="optimized"
    )
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--reference", type=Path)
    a = p.parse_args()
    a.input = a.input.resolve()
    a.model_dir = a.model_dir.resolve()
    a.output_dir = a.output_dir.resolve()
    if a.variant == "optimized" and a.reference is None:
        p.error("--reference is required for optimized output verification")
    if a.reference is not None:
        a.reference = a.reference.resolve()
    os.chdir(root / "examples/WanVSR")
    spec = importlib.util.spec_from_file_location(
        "infer", root / f"examples/WanVSR/infer_flashvsr_v1.1_{a.mode}.py"
    )
    infer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(infer)
    out = a.output_dir / a.variant
    out.mkdir(parents=True, exist_ok=True)
    total_start = time.perf_counter()
    t = time.perf_counter()
    pipe = infer.init_pipeline(a.model_dir)
    torch.cuda.synchronize()
    load = time.perf_counter() - t
    reader = imageio.get_reader(str(a.input))
    metadata = reader.get_meta_data()
    if abs(metadata.get("fps", 0) - 24) > 0.05:
        raise ValueError("This fixed workload requires a 24 FPS input")
    frames = []
    for frame in reader:
        if frame.shape[:2] != (1344, 768):
            raise ValueError("This fixed workload requires a 768x1344 input")
        frames.append(
            np.array(
                Image.fromarray(frame)
                .convert("RGB")
                .resize((1536, 2688), Image.Resampling.BICUBIC)
            )
        )
    reader.close()
    n = len(frames)
    if n != 73:
        raise ValueError("This fixed workload requires 73 input frames")
    padded = ((n + 4 - 1 + 7) // 8) * 8 + 1
    # Pad to retain every source frame through the streaming decoder's four-frame trim.
    frames.extend([frames[-1]] * (padded - n))
    lq = (
        torch.from_numpy(np.stack(frames))
        .permute(3, 0, 1, 2)
        .unsqueeze(0)
        .cuda()
        .to(torch.bfloat16)
        .div_(127.5)
        .sub_(1)
    )
    del frames
    decode_times = []
    if a.variant == "optimized":
        pipe.TCDecoder.set_rocm_cache_release()
    else:
        for module in pipe.TCDecoder.modules():
            if isinstance(module, torch.nn.Conv2d):
                module.register_forward_hook(lambda m, x, y: torch.cuda.empty_cache())
    original_decode = pipe.TCDecoder.decode_video

    def measured_decode(*args, **kwargs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = original_decode(*args, **kwargs)
        torch.cuda.synchronize()
        decode_times.append(time.perf_counter() - start)
        return result

    pipe.TCDecoder.decode_video = measured_decode
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t = time.perf_counter()
    import diffsynth.models.wan_video_dit as wan

    original_mask = wan.generate_draft_block_mask

    def bounded_mask(*args, **kwargs):
        result = original_mask(*args, **kwargs)
        torch.cuda.empty_cache()
        return result

    wan.generate_draft_block_mask = bounded_mask
    video = pipe(
        prompt="",
        negative_prompt="",
        cfg_scale=1.0,
        num_inference_steps=1,
        seed=0,
        LQ_video=lq,
        num_frames=padded,
        height=2688,
        width=1536,
        tiled=True,
        is_full_block=False,
        if_buffer=True,
        topk_ratio=2.0 * 768 * 1280 / (2688 * 1536),
        kv_ratio=3.0,
        local_range=11,
        color_fix=True,
    )
    torch.cuda.synchronize()
    seconds = time.perf_counter() - t
    video = video[:, :n]
    assert video.shape[1] == n and torch.isfinite(video).all()
    # Downsample agreement to the source is a consistency metric, not HR ground truth.
    squared = absolute = count = 0
    writer = imageio.get_writer(
        str(out / f"orbit_2x_{a.mode}.mp4"),
        fps=24,
        codec="libx264",
        quality=8,
        macro_block_size=None,
    )
    reader = imageio.get_reader(str(a.input))
    for i in range(n):
        x = video[:, i].float().clamp(-1, 1)
        low = torch.nn.functional.interpolate(
            x[None], size=(1344, 768), mode="bicubic", align_corners=False
        )[0]
        src = (
            torch.from_numpy(reader.get_data(i).copy())
            .permute(2, 0, 1)
            .cuda()
            .float()
            .div(127.5)
            .sub(1)
        )
        d = low - src
        squared += d.square().sum().item()
        absolute += d.abs().sum().item()
        count += d.numel()
        writer.append_data(
            x.add(1).mul(127.5).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
        )
    writer.close()
    reader.close()
    metrics = {
        "mode": a.mode,
        "variant": a.variant,
        "decode_seconds": decode_times,
        "input_frames": n,
        "padded_frames": padded,
        "output_frames": video.shape[1],
        "width": 1536,
        "height": 2688,
        "load_seconds": load,
        "inference_seconds": seconds,
        "fps": n / seconds,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "downsample_source_rmse": (squared / count) ** 0.5,
        "downsample_source_psnr": 10 * np.log10(4 / (squared / count)),
        "downsample_source_mae": absolute / count,
        "total_seconds": time.perf_counter() - total_start,
    }
    output = video.cpu()
    torch.save(output, out / "output.pt")
    if a.reference is not None:
        ref = torch.load(a.reference, weights_only=True)
        diff = video.cpu().float() - ref.float()
        metrics.update(
            output_max_abs=diff.abs().max().item(),
            output_rmse=diff.square().mean().sqrt().item(),
        )
        torch.testing.assert_close(video.cpu(), ref, atol=0, rtol=0)
    (out / f"{a.mode}_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(metrics, flush=True)


if __name__ == "__main__":
    main()
