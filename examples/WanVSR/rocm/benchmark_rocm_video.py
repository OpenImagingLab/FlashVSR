def main():
    """Measure official v1.1 Tiny inference with synchronized GPU timings."""

    import argparse
    import importlib.util
    import json
    import os
    import sys
    import time
    from pathlib import Path
    import numpy as np
    import imageio
    import torch
    from PIL import Image

    root = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(root), str(root / "examples/WanVSR")]
    spec = importlib.util.spec_from_file_location(
        "infer", root / "examples/WanVSR/infer_flashvsr_v1.1_tiny.py"
    )
    infer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(infer)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/rocm-benchmark")
    )
    parser.add_argument("--backend", choices=("flex", "reference"), default="flex")
    parser.add_argument("--precision", choices=("native", "fp32"), default="native")
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--frames", type=int, default=33)
    parser.add_argument("--runs", type=int, default=4)
    parser.add_argument("--capture-attention", action="store_true")
    args = parser.parse_args()
    args.model_dir = args.model_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    os.chdir(root / "examples/WanVSR")
    if (
        args.height % 128
        or args.width % 128
        or args.frames < 25
        or args.frames % 8 != 1
        or args.runs < 2
    ):
        parser.error("Use 128-multiple dimensions, frames=8n+1 >=25 and runs>=2")
    os.environ["FLASHVSR_ROCM_ATTENTION"] = args.backend
    os.environ["FLASHVSR_ROCM_PRECISION"] = args.precision
    out = (
        Path(args.output_dir)
        / f"{args.backend}_{args.precision}_{args.width}x{args.height}_{args.frames}"
    )
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    pipe = infer.init_pipeline(args.model_dir)
    torch.cuda.synchronize()
    load = time.perf_counter() - t0
    reader = imageio.get_reader(str(root / "examples/WanVSR/inputs/example0.mp4"))
    frames = []
    for i in range(args.frames):
        img = (
            Image.fromarray(reader.get_data(i))
            .convert("RGB")
            .resize((args.width // 4, args.height // 4), Image.Resampling.BICUBIC)
            .resize((args.width, args.height), Image.Resampling.BICUBIC)
        )
        frames.append(torch.from_numpy(np.array(img)).permute(2, 0, 1))
    reader.close()
    lq = (
        torch.stack(frames, dim=1)
        .unsqueeze(0)
        .cuda()
        .float()
        .div(127.5)
        .sub(1)
        .to(torch.bfloat16)
    )
    # Same source frames and device-generated fixed noise for both backends.
    from diffsynth.models import rocm_attention

    original_attention = rocm_attention.sparse_attention
    captured = []

    def capture_attention(q, k, v, mask):
        if args.capture_attention and not captured:
            torch.save(
                {"q": q.cpu(), "k": k.cpu(), "v": v.cpu(), "mask": mask.cpu()},
                out / "attention_input.pt",
            )
            captured.append(True)
        return original_attention(q, k, v, mask)

    # Resolve through the symbol imported by the official model.
    import diffsynth.models.wan_video_dit as wan

    wan.sparse_attention = capture_attention
    results = []
    for run in range(args.runs):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        video = pipe(
            prompt="",
            negative_prompt="",
            cfg_scale=1.0,
            num_inference_steps=1,
            seed=0,
            LQ_video=lq,
            num_frames=args.frames,
            height=args.height,
            width=args.width,
            is_full_block=False,
            if_buffer=True,
            topk_ratio=2.0 * 768 * 1280 / (args.height * args.width),
            kv_ratio=3.0,
            local_range=11,
            color_fix=True,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        assert torch.isfinite(video).all()
        results.append(
            {
                "seconds": elapsed,
                "frames": video.shape[1],
                "fps": video.shape[1] / elapsed,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
        )
        print(results[-1], flush=True)
        torch.save(video.cpu(), out / f"output_{run}.pt")
        if run == args.runs - 1:
            infer.save_video(
                infer.tensor2video(video), str(out / "output.mp4"), fps=24, quality=8
            )
    (out / "metrics.json").write_text(
        json.dumps(
            {
                "torch": torch.__version__,
                "hip": torch.version.hip,
                "gpu": torch.cuda.get_device_name(),
                "backend": args.backend,
                "precision": args.precision,
                "load_seconds": load,
                "height": args.height,
                "width": args.width,
                "input_frames": args.frames,
                "runs": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
