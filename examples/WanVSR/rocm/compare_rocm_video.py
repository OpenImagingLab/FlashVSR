def main():
    """Compare lossless output tensors from benchmark_rocm_video.py."""

    import argparse
    import json
    import math
    from pathlib import Path

    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("actual", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    a = torch.load(args.actual, map_location="cpu", weights_only=True).float()
    b = torch.load(args.reference, map_location="cpu", weights_only=True).float()
    if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("Outputs must have matching shapes and finite values")
    d = a - b
    mse = d.square().mean().item()
    result = {
        "max_abs": d.abs().max().item(),
        "mae": d.abs().mean().item(),
        "rmse": math.sqrt(mse),
        "psnr_range_2": 10 * math.log10(4 / mse) if mse else None,
        "temporal_delta_rmse": (d[:, 1:] - d[:, :-1]).square().mean().sqrt().item(),
    }
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
