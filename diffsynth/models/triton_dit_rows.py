"""Row-wise DiT elementwise kernels used by the optional Phase-7 fast path."""

import torch

try:
    import triton
    import triton.language as tl
    _TRITON_OK = True
except Exception:  # pragma: no cover
    _TRITON_OK = False


if _TRITON_OK:

    @triton.jit
    def _layer_norm_modulate_kernel(
        X, SHIFT, SCALE, OUT,
        N: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(x, axis=0) / N
        centered = x - mean
        variance = tl.sum(centered * centered, axis=0) / N
        normalized = centered * tl.rsqrt(variance + EPS)
        shift = tl.load(SHIFT + cols, mask=mask, other=0.0).to(tl.float32)
        scale = tl.load(SCALE + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(OUT + row * N + cols, normalized * (1.0 + scale) + shift, mask=mask)


    @triton.jit
    def _gated_residual_kernel(
        X, GATE, RESIDUAL, OUT, TOTAL,
        N: tl.constexpr, BLOCK: tl.constexpr,
    ):
        block = tl.program_id(0)
        offsets = block * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < TOTAL
        cols = offsets % N
        x = tl.load(X + offsets, mask=mask, other=0.0).to(tl.float32)
        residual = tl.load(RESIDUAL + offsets, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(GATE + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(OUT + offsets, x + gate * residual, mask=mask)


def _check_bf16_rows(x, *vectors):
    if not _TRITON_OK:
        raise RuntimeError("Triton is unavailable")
    if x.device.type != "cuda" or x.dtype != torch.bfloat16 or not x.is_contiguous():
        raise ValueError("expected contiguous CUDA bf16 rows")
    if x.ndim < 2:
        raise ValueError("expected a tensor with a row dimension")
    width = x.shape[-1]
    if width > 4096:
        raise ValueError(f"unsupported row width {width}")
    for vector in vectors:
        if vector.device != x.device or vector.dtype != x.dtype or vector.numel() != width:
            raise ValueError("modulation vector does not match the input rows")
    return width


def layer_norm_modulate_triton(x, shift, scale, eps):
    """Fused affine-free LayerNorm followed by AdaLN modulation."""
    width = _check_bf16_rows(x, shift, scale)
    rows = x.numel() // width
    out = torch.empty_like(x)
    block = triton.next_power_of_2(width)
    _layer_norm_modulate_kernel[(rows,)](
        x, shift.reshape(-1), scale.reshape(-1), out,
        N=width, EPS=eps, BLOCK=block, num_warps=8,
    )
    return out


def gated_residual_triton(x, gate, residual):
    """Fused ``x + gate * residual`` for row-broadcast AdaLN gates."""
    width = _check_bf16_rows(x, gate)
    if residual.shape != x.shape or residual.dtype != x.dtype or not residual.is_contiguous():
        raise ValueError("residual does not match the input rows")
    out = torch.empty_like(x)
    total = x.numel()
    block = 2048
    _gated_residual_kernel[(triton.cdiv(total, block),)](
        x, gate.reshape(-1), residual, out, total,
        N=width, BLOCK=block, num_warps=8,
    )
    return out
