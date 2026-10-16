"""Exact BF16 im2col packing for the FlashVSR LQ Conv3D layers."""

import torch


_TRITON_OK = False
_TRITON_ERR = None
try:
    import triton
    import triton.language as tl
    _TRITON_OK = True
except Exception as exc:  # pragma: no cover
    _TRITON_ERR = exc


if _TRITON_OK:
    @triton.jit
    def _im2col3d_kernel(x, patches, h_start,
                         N: tl.constexpr, CIN: tl.constexpr,
                         T: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
                         TO: tl.constexpr, HO: tl.constexpr, WO: tl.constexpr,
                         KT: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
                         ST: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
                         K: tl.constexpr,
                         BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
        m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
        k = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)[None, :]
        valid = (m < N * TO * HO * WO) & (k < K)

        wo = m % WO
        m0 = m // WO
        ho = m0 % HO
        m0 = m0 // HO
        to = m0 % TO
        batch = m0 // TO

        kernel_w = k % KW
        k0 = k // KW
        kernel_h = k0 % KH
        k0 = k0 // KH
        kernel_t = k0 % KT
        channel = k0 // KT

        input_t = to * ST + kernel_t
        input_h = (ho + h_start) * SH + kernel_h
        input_w = wo * SW + kernel_w
        source = ((((batch * CIN + channel) * T + input_t) * H + input_h)
                  * W + input_w)
        value = tl.load(x + source, mask=valid)
        tl.store(patches + m * K + k, value, mask=valid)


def im2col3d(x, kernel_size, stride, h_start, h_rows):
    """Return the exact eager patch matrix for output rows in this H slice."""
    if not _TRITON_OK:
        raise RuntimeError(f"Triton unavailable: {_TRITON_ERR}")
    if x.ndim != 5 or not x.is_cuda or x.dtype != torch.bfloat16:
        raise ValueError("im2col3d requires CUDA BF16 NCTHW")
    if not x.is_contiguous():
        raise ValueError("im2col3d requires contiguous NCTHW input")
    n, cin, t, height, width = x.shape
    kt, kh, kw = kernel_size
    st, sh, sw = stride
    to = (t - kt) // st + 1
    wo = (width - kw) // sw + 1
    m = n * to * h_rows * wo
    k = cin * kt * kh * kw
    patches = torch.empty((m, k), dtype=x.dtype, device=x.device)
    block_m, block_k = 4, 256
    grid = (triton.cdiv(m, block_m), triton.cdiv(k, block_k))
    _im2col3d_kernel[grid](
        x, patches, h_start,
        N=n, CIN=cin, T=t, H=height, W=width,
        TO=to, HO=h_rows, WO=wo,
        KT=kt, KH=kh, KW=kw, ST=st, SH=sh, SW=sw, K=k,
        BLOCK_M=block_m, BLOCK_K=block_k, num_warps=8)
    return patches
