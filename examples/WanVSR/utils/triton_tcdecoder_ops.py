"""Forward-only Triton pointwise kernels for the Tiny decoder."""

import torch


_TRITON_OK = False
_TRITON_ERR = None
try:
    import triton
    import triton.language as tl
    _TRITON_OK = True
except Exception as exc:  # pragma: no cover - optional fast path
    _TRITON_ERR = exc


if _TRITON_OK:
    @triton.jit
    def _bias_relu_kernel(x, bias, out, total, channels,
                          BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = offsets < total
        channel = offsets % channels
        value = tl.load(x + offsets, mask=valid, other=0.0).to(tl.float32)
        value += tl.load(bias + channel, mask=valid, other=0.0).to(tl.float32)
        # Preserve eager's BF16 materialization between bias-add and ReLU.
        rounded = value.to(tl.bfloat16).to(tl.float32)
        relu = tl.where(rounded > 0.0, rounded, 0.0)
        relu = tl.where(rounded != rounded, rounded, relu)  # preserve NaNs
        tl.store(out + offsets, relu, mask=valid)


    @triton.jit
    def _bias_residual_relu_kernel(x, bias, residual, out, total, channels,
                                   BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = offsets < total
        channel = offsets % channels
        value = tl.load(x + offsets, mask=valid, other=0.0).to(tl.float32)
        value += tl.load(bias + channel, mask=valid, other=0.0).to(tl.float32)
        # Eager writes BF16 after bias-add and again after residual-add.
        value = value.to(tl.bfloat16).to(tl.float32)
        value += tl.load(residual + offsets, mask=valid, other=0.0).to(tl.float32)
        rounded = value.to(tl.bfloat16).to(tl.float32)
        relu = tl.where(rounded > 0.0, rounded, 0.0)
        relu = tl.where(rounded != rounded, rounded, relu)
        tl.store(out + offsets, relu, mask=valid)


    @triton.jit
    def _upsample2x_channels_last_kernel(x, out, total, channels, height, width,
                                         BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = offsets < total
        frame_elems = height * width * channels
        frame = offsets // frame_elems
        within_frame = offsets % frame_elems
        pixel = within_frame // channels
        channel = within_frame % channels
        row = pixel // width
        column = pixel % width

        out_width = width * 2
        out_frame_elems = frame_elems * 4
        out_base = frame * out_frame_elems
        row0 = row * 2
        column0 = column * 2
        dst00 = out_base + (row0 * out_width + column0) * channels + channel
        dst01 = dst00 + channels
        dst10 = dst00 + out_width * channels
        dst11 = dst10 + channels
        value = tl.load(x + offsets, mask=valid)
        tl.store(out + dst00, value, mask=valid)
        tl.store(out + dst01, value, mask=valid)
        tl.store(out + dst10, value, mask=valid)
        tl.store(out + dst11, value, mask=valid)


    @triton.jit
    def _tgrow_upsample2x_channels_last_kernel(x, out, total, channels,
                                               wide_channels, height, width,
                                               BLOCK: tl.constexpr):
        # x: (1, C*S, H, W) channels-last = [pixel, cglob] row-major with the
        # C*S channel axis innermost. out: (S, C, 2H, 2W) channels-last.
        # TGrow channel-group semantics: group g = channels [g*C, (g+1)*C)
        # becomes temporal frame g. Each input element is read once and
        # written to its 2x2 nearest-neighbor footprint inside frame g.
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = offsets < total
        pixel = offsets // wide_channels
        cglob = offsets % wide_channels
        group = cglob // channels
        channel = cglob % channels
        row = pixel // width
        column = pixel % width
        out_width = width * 2
        frame_base = group * (channels * height * width * 4)
        dst00 = frame_base \
            + ((row * 2) * out_width + column * 2) * channels + channel
        dst01 = dst00 + channels
        dst10 = dst00 + out_width * channels
        dst11 = dst10 + channels
        value = tl.load(x + offsets, mask=valid)
        tl.store(out + dst00, value, mask=valid)
        tl.store(out + dst01, value, mask=valid)
        tl.store(out + dst10, value, mask=valid)
        tl.store(out + dst11, value, mask=valid)


    @triton.jit
    def _concat_channels_last_kernel(x, past, out, total, channels,
                                     frame_elems,
                                     BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = offsets < total
        frame = offsets // frame_elems
        within_frame = offsets % frame_elems
        pixel = within_frame // channels
        channel = within_frame % channels
        output_frame_elems = frame_elems * 2
        output_base = frame * output_frame_elems + pixel * channels * 2
        current_value = tl.load(x + offsets, mask=valid)
        past_value = tl.load(past + offsets, mask=valid)
        tl.store(out + output_base + channel, current_value, mask=valid)
        tl.store(out + output_base + channels + channel, past_value, mask=valid)


def _validate(x, bias, residual=None):
    if not _TRITON_OK:
        raise RuntimeError(f"Triton unavailable: {_TRITON_ERR}")
    if x.ndim != 4 or not x.is_cuda or x.dtype != torch.bfloat16:
        raise ValueError("TCDecoder fusion requires CUDA BF16 NCHW")
    if not x.is_contiguous(memory_format=torch.channels_last):
        raise ValueError("TCDecoder fusion requires channels-last input")
    if bias is None or bias.ndim != 1 or bias.shape[0] != x.shape[1]:
        raise ValueError("bias shape mismatch")
    if residual is not None:
        if residual.shape != x.shape or residual.stride() != x.stride():
            raise ValueError("residual shape/stride mismatch")


def bias_relu(x, bias):
    _validate(x, bias)
    out = torch.empty_like(x, memory_format=torch.preserve_format)
    block = 1024
    _bias_relu_kernel[(triton.cdiv(x.numel(), block),)](
        x, bias, out, x.numel(), x.shape[1], BLOCK=block, num_warps=8)
    return out


def bias_residual_relu(x, bias, residual):
    _validate(x, bias, residual)
    out = torch.empty_like(x, memory_format=torch.preserve_format)
    block = 1024
    _bias_residual_relu_kernel[(triton.cdiv(x.numel(), block),)](
        x, bias, residual, out, x.numel(), x.shape[1], BLOCK=block,
        num_warps=8)
    return out


def upsample2x_channels_last(x):
    if not _TRITON_OK:
        raise RuntimeError(f"Triton unavailable: {_TRITON_ERR}")
    if x.ndim != 4 or not x.is_cuda or x.dtype != torch.bfloat16:
        raise ValueError("upsample requires CUDA BF16 NCHW")
    if not x.is_contiguous(memory_format=torch.channels_last):
        raise ValueError("upsample requires channels-last input")
    n, channels, height, width = x.shape
    out = torch.empty(
        (n, channels, height * 2, width * 2), dtype=x.dtype, device=x.device,
        memory_format=torch.channels_last)
    block = 1024
    _upsample2x_channels_last_kernel[(triton.cdiv(x.numel(), block),)](
        x, out, x.numel(), channels, height, width, BLOCK=block, num_warps=8)
    return out


def tgrow_upsample2x_channels_last(x, stride):
    """Fused TGrow temporal unpack + nearest 2x upsample.

    x: (1, C*stride, H, W) channels-last BF16 (the low-resolution TGrow 1x1
    conv output). Returns (stride, C, 2H, 2W) channels-last BF16 where frame g
    is the nearest-upsampled channel group [g*C, (g+1)*C) — exactly the frames
    the eager `Upsample -> TGrow -> view/chunk` sequence produces, already
    materialized contiguously in temporal order.
    """
    if not _TRITON_OK:
        raise RuntimeError(f"Triton unavailable: {_TRITON_ERR}")
    if x.ndim != 4 or not x.is_cuda or x.dtype != torch.bfloat16:
        raise ValueError("tgrow upsample requires CUDA BF16 NCHW")
    if x.shape[0] != 1:
        raise ValueError("tgrow upsample requires N == 1")
    if not x.is_contiguous(memory_format=torch.channels_last):
        raise ValueError("tgrow upsample requires channels-last input")
    if stride < 1 or x.shape[1] % stride != 0:
        raise ValueError("channel count must be divisible by TGrow stride")
    _, wide_channels, height, width = x.shape
    channels = wide_channels // stride
    out = torch.empty(
        (stride, channels, height * 2, width * 2), dtype=x.dtype,
        device=x.device, memory_format=torch.channels_last)
    block = 1024
    _tgrow_upsample2x_channels_last_kernel[(triton.cdiv(x.numel(), block),)](
        x, out, x.numel(), channels, wide_channels, height, width,
        BLOCK=block, num_warps=8)
    return out


def concat_channels_last(x, past):
    if not _TRITON_OK:
        raise RuntimeError(f"Triton unavailable: {_TRITON_ERR}")
    if x.shape != past.shape or x.stride() != past.stride():
        raise ValueError("concat inputs must have identical shape and stride")
    if x.ndim != 4 or not x.is_cuda or x.dtype != torch.bfloat16:
        raise ValueError("concat requires CUDA BF16 NCHW")
    if not x.is_contiguous(memory_format=torch.channels_last):
        raise ValueError("concat requires channels-last inputs")
    n, channels, height, width = x.shape
    out = torch.empty(
        (n, channels * 2, height, width), dtype=x.dtype, device=x.device,
        memory_format=torch.channels_last)
    block = 1024
    _concat_channels_last_kernel[(triton.cdiv(x.numel(), block),)](
        x, past, out, x.numel(), channels, channels * height * width,
        BLOCK=block, num_warps=8)
    return out
