"""Phase 2B-2: FP8 GEMM infrastructure (FLASHVSR_FP8_GEMM, default OFF).

Routes the DiT hot-path GEMMs (self-attn q/k/v/o, ffn1/ffn2, LQ-projector
per-layer linears) through `torch._scaled_mm` with float8_e4m3fn operands:

  * Weights are pre-cast ONCE per module (lazy, cached on the module) with
    per-output-channel scales -> (N, K) e4m3 + (1, N) fp32 scales.
  * Activations are quantized dynamically per call with per-row scales via a
    single Triton kernel (row amax + scale + cast in one launch; the eager
    equivalent chain of abs/amax/div/clamp/cast kernels measured ~5x slower
    than bandwidth and would erase the GEMM win).
  * ffn2's input quantization is fused with the GELU(tanh) activation in the
    same kernel (fp32 gelu -> e4m3), replacing the eager bf16 GELU pass
    entirely; on the widest activation in the network this is the difference
    between FP8 being a net win and a net loss.
  * Bias is applied inside `_scaled_mm`'s epilogue (bf16 bias, bf16 out).

Numerics: this is NOT a lossless path (e4m3 mantissa = 3 bits). It ships
permanently default-OFF; the enable decision belongs to the Phase-4 quality
protocol (PSNR/SSIM vs flag-OFF). Rollback: FLASHVSR_FP8_GEMM=0 (default)
leaves every call site on the original eager path, bit-for-bit.

Scope control (comma list, default all): FLASHVSR_FP8_GEMM_SCOPE=qkv,o,ffn,lq
— used for per-site attribution if the Phase-4 quality audit needs to bisect.

Any failure inside the FP8 path (unsupported build, shape, dtype) disables it
for the rest of the process (warn once) and falls back to eager — same silent
degradation pattern as the attention backends.
"""
import os

import torch

_FP8_GEMM = os.environ.get("FLASHVSR_FP8_GEMM", "0") != "0"
_FP8_SCOPE = frozenset(
    s.strip() for s in
    os.environ.get("FLASHVSR_FP8_GEMM_SCOPE", "qkv,o,ffn,lq").split(",")
    if s.strip()
)
# Phase 4B: rowwise (2B-2) vs blockwise (DeepSeek-style 1x128 activation +
# 1x128 weight scales via torch._scaled_mm_v2). Blockwise captures per-128-K
# activation outliers (the post-GELU spikes that pinned rowwise at 40.7 dB),
# targeting the >=49 dB enable gate. Requires cublasLt >=12.9 + sm_90.
_FP8_MODE = os.environ.get("FLASHVSR_FP8_GEMM_MODE", "rowwise").lower()
# Quality-probe switch: eager group-quant (correct but slow) so the >=49 dB
# gate can be measured before investing in the fused blockwise quant kernel.
_FP8_BLOCKWISE_EAGER = os.environ.get("FLASHVSR_FP8_BLOCKWISE_EAGER", "1") != "0"

try:
    from torch.nn.functional import ScalingType as _ScalingType, SwizzleType as _SwizzleType
    _HAS_V2 = hasattr(torch, "_scaled_mm_v2")
except Exception:
    _HAS_V2 = False

_FAILED = False  # sticky: set on first unexpected error, forces eager fallback

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:  # pragma: no cover
    _HAS_TRITON = False


def _capability_ok():
    if not (_HAS_TRITON and hasattr(torch, "_scaled_mm")):
        return False
    try:
        if not torch.cuda.is_available():
            return False
        major, minor = torch.cuda.get_device_capability()
        return (major, minor) >= (8, 9)  # e4m3 matmul: sm_89+
    except Exception:
        return False


_CAP_OK = None


def enabled(site: str) -> bool:
    """Per-call-site gate. Reads the module flag at call time so the parity
    harness can toggle `fp8_gemm._FP8_GEMM` on a live session."""
    global _CAP_OK
    if not _FP8_GEMM or _FAILED or site not in _FP8_SCOPE:
        return False
    if _CAP_OK is None:
        _CAP_OK = _capability_ok()
    return _CAP_OK


if _HAS_TRITON:

    @triton.jit
    def _rowquant_kernel(X, Y, S, K,
                         stride_xm, stride_ym,
                         APPLY_GELU: tl.constexpr, BLOCK_K: tl.constexpr):
        """Per-row dynamic e4m3 quantization (optionally fused with GELU-tanh).

        One program per row. Two passes over the row tiles (amax, then cast);
        the second pass re-reads through L2. All math in fp32.
        Y[row, :] = clamp(x * 448/amax, +-448) as e4m3;  S[row] = amax/448.
        """
        row = tl.program_id(0).to(tl.int64)
        xrow = X + row * stride_xm
        yrow = Y + row * stride_ym

        amax = 0.0
        for k0 in range(0, K, BLOCK_K):
            offs = k0 + tl.arange(0, BLOCK_K)
            m = offs < K
            v = tl.load(xrow + offs, mask=m, other=0.0).to(tl.float32)
            if APPLY_GELU:
                # torch GELU(approximate='tanh'): 0.5*v*(1+tanh(c0*(v+c1*v^3)))
                # tanh via exp(-2|t|) (never overflows), sign restored after.
                t = 0.7978845608028654 * (v + 0.044715 * v * v * v)
                e = tl.exp(-2.0 * tl.abs(t))
                th = (1.0 - e) / (1.0 + e)
                th = tl.where(t < 0.0, -th, th)
                v = 0.5 * v * (1.0 + th)
            amax = tl.maximum(amax, tl.max(tl.abs(v), axis=0))

        amax = tl.maximum(amax, 1e-12)
        inv = 448.0 / amax
        for k0 in range(0, K, BLOCK_K):
            offs = k0 + tl.arange(0, BLOCK_K)
            m = offs < K
            v = tl.load(xrow + offs, mask=m, other=0.0).to(tl.float32)
            if APPLY_GELU:
                t = 0.7978845608028654 * (v + 0.044715 * v * v * v)
                e = tl.exp(-2.0 * tl.abs(t))
                th = (1.0 - e) / (1.0 + e)
                th = tl.where(t < 0.0, -th, th)
                v = 0.5 * v * (1.0 + th)
            q = tl.minimum(tl.maximum(v * inv, -448.0), 448.0)
            tl.store(yrow + offs, q.to(tl.float8e4nv), mask=m)
        tl.store(S + row, amax / 448.0)


def quant(x2d: torch.Tensor, apply_gelu: bool = False):
    """Shared activation quant. Rowwise (2B-2) or blockwise (4B) per mode."""
    if _FP8_MODE == "blockwise":
        if x2d.stride(1) != 1:
            x2d = x2d.contiguous()
        return (_blockquant_eager_gelu(x2d) if apply_gelu
                else _blockquant_eager(x2d))
    assert x2d.dim() == 2, "2D input required"
    if x2d.stride(1) != 1:
        # e.g. the LQ projector's rearrange returns a transposed view; one
        # contiguous copy here is amortized over all GEMMs sharing the quant.
        x2d = x2d.contiguous()
    M, K = x2d.shape
    y = torch.empty(M, K, dtype=torch.float8_e4m3fn, device=x2d.device)
    s = torch.empty(M, 1, dtype=torch.float32, device=x2d.device)
    _rowquant_kernel[(M,)](
        x2d, y, s, K, x2d.stride(0), y.stride(0),
        APPLY_GELU=apply_gelu, BLOCK_K=1024, num_warps=4,
    )
    return y, s


def _weight_fp8(lin):
    """Lazy per-module weight cast: (N,K) e4m3 + (1,N) fp32 per-out-channel
    scales, cached on the module. Weights are inference-static; the cache key
    guards against reload/device moves. Works for AutoWrappedLinear too
    (.weight/.bias exposed)."""
    w = lin.weight
    key = (w.data_ptr(), w.device, w.dtype, w.shape)
    cache = getattr(lin, "_fp8_wcache", None)
    if cache is not None and cache[0] == key:
        return cache[1], cache[2]
    with torch.no_grad():
        wf = w.detach().to(torch.float32)
        s = wf.abs().amax(dim=1, keepdim=True).clamp_(min=1e-12) / 448.0  # (N,1)
        w8 = (wf / s).clamp_(-448.0, 448.0).to(torch.float8_e4m3fn)       # (N,K)
    sb = s.reshape(1, -1)  # (1,N) fp32, contiguous view of (N,1)
    lin._fp8_wcache = (key, w8, sb)
    return w8, sb


def _blockquant_eager(x2d):
    """Eager 1x128 group quant (quality probe). (M,K)->(e4m3 (M,K),
    scale (M,K//128) laid out as .t() of contiguous (K//128,M))."""
    M, K = x2d.shape
    G = K // 128
    xf = x2d.to(torch.float32).reshape(M, G, 128)
    amax = xf.abs().amax(-1).clamp_(min=1e-12)          # (M,G)
    s = amax / 448.0
    x8 = (xf / s[..., None]).clamp_(-448.0, 448.0).to(
        torch.float8_e4m3fn).reshape(M, K)
    sa = s.t().contiguous().t()                          # (M,G) col-major
    return x8, sa


def _blockquant_eager_gelu(h2d):
    """GELU(tanh) fused into 1x128 group quant (ffn2 input)."""
    v = h2d.to(torch.float32)
    v = torch.nn.functional.gelu(v, approximate="tanh")
    return _blockquant_eager(v)


def _weight_fp8_blockwise(lin):
    """Lazy 1x128-along-K weight cast: b operand (K,N) col-major e4m3 +
    scale (N,K//128) laid out as .t() of contiguous (K//128,N)."""
    w = lin.weight
    key = (w.data_ptr(), w.device, w.dtype, w.shape)
    cache = getattr(lin, "_fp8_wcache_bw", None)
    if cache is not None and cache[0] == key:
        return cache[1], cache[2]
    N, K = w.shape
    G = K // 128
    with torch.no_grad():
        wf = w.detach().to(torch.float32).reshape(N, G, 128)
        wamax = wf.abs().amax(-1).clamp_(min=1e-12)      # (N,G)
        ws = wamax / 448.0
        w8 = (wf / ws[..., None]).clamp_(-448.0, 448.0).to(
            torch.float8_e4m3fn).reshape(N, K)
    b8 = w8.t()                                          # (K,N) col-major
    sb = ws.t().contiguous().t()                         # (N,G) col-major
    lin._fp8_wcache_bw = (key, b8, sb)
    return b8, sb


def _mm_blockwise(x8, sa, b8, sb, bias, out_dtype):
    return torch._scaled_mm_v2(
        x8, b8,
        [sa], [_ScalingType.BlockWise1x128], [_SwizzleType.NO_SWIZZLE],
        [sb], [_ScalingType.BlockWise1x128], [_SwizzleType.NO_SWIZZLE],
        bias, out_dtype)


def _linear_blockwise(lin, x, pre=None):
    K = x.shape[-1]
    x2 = x.reshape(-1, K)
    if x2.stride(1) != 1:
        x2 = x2.contiguous()
    x8, sa = _blockquant_eager(x2) if pre is None else pre
    b8, sb = _weight_fp8_blockwise(lin)
    out = _mm_blockwise(x8, sa, b8, sb, lin.bias, x.dtype)
    return out.reshape(*x.shape[:-1], b8.shape[1])


def _ffn_blockwise(seq, x):
    h = _linear_blockwise(seq[0], x)
    h2 = h.reshape(-1, h.shape[-1])
    g8, gs = _blockquant_eager_gelu(h2)
    b8, sb = _weight_fp8_blockwise(seq[2])
    out = _mm_blockwise(g8, gs, b8, sb, seq[2].bias, x.dtype)
    return out.reshape(*x.shape[:-1], b8.shape[1])


def _warn_and_disable(e):
    global _FAILED
    if not _FAILED:
        _FAILED = True
        print(f"[fp8_gemm] disabled after error, falling back to eager: {e!r}")


def linear(lin, x: torch.Tensor, pre=None) -> torch.Tensor:
    """FP8 replacement for lin(x). `pre` = optional shared (x8, scales) from
    `quant()` when several linears consume the same activation (qkv, LQ).
    Falls back to eager on any error (sticky)."""
    try:
        if _FP8_MODE == "blockwise":
            return _linear_blockwise(lin, x, pre=pre)
        K = x.shape[-1]
        x2 = x.reshape(-1, K)
        x8, sa = quant(x2) if pre is None else pre
        w8, sb = _weight_fp8(lin)
        out = torch._scaled_mm(x8, w8.t(), scale_a=sa, scale_b=sb,
                               bias=lin.bias, out_dtype=x.dtype)
        return out.reshape(*x.shape[:-1], w8.shape[0])
    except Exception as e:  # pragma: no cover
        _warn_and_disable(e)
        return torch.nn.functional.linear(x, lin.weight, lin.bias)


def ffn(seq, x: torch.Tensor) -> torch.Tensor:
    """FP8 replacement for Sequential(Linear, GELU(tanh), Linear)(x).
    ffn1 out stays bf16; the GELU is fused into ffn2's input quantization
    (fp32 gelu -> e4m3 directly, replacing the eager bf16 GELU pass)."""
    try:
        if _FP8_MODE == "blockwise":
            return _ffn_blockwise(seq, x)
        h = linear(seq[0], x)
        h2 = h.reshape(-1, h.shape[-1])
        g8, gs = quant(h2, apply_gelu=True)
        w8, sb = _weight_fp8(seq[2])
        out = torch._scaled_mm(g8, w8.t(), scale_a=gs, scale_b=sb,
                               bias=seq[2].bias, out_dtype=x.dtype)
        return out.reshape(*x.shape[:-1], w8.shape[0])
    except Exception as e:  # pragma: no cover
        _warn_and_disable(e)
        return seq(x)


def ffn_partial(seq, x: torch.Tensor) -> torch.Tensor:
    """Scope 'ffn1' (not in the default set): e4m3 ffn1 only, eager bf16
    GELU + ffn2. Quality/speed bisection point — ffn2's post-GELU activation
    is outlier-heavy and is the dominant fp8 quality cost."""
    try:
        h = linear(seq[0], x)
        return seq[2](seq[1](h))
    except Exception as e:  # pragma: no cover
        _warn_and_disable(e)
        return seq(x)
