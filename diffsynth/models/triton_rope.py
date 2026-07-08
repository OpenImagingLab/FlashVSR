"""Hand-written coalesced RoPE apply kernel (FLASHVSR_ROPE_KERNEL=triton).

The Phase-2A-1b fused RoPE (torch.compile) is bit-exact vs eager but reaches
only ~484 GB/s (143 us/call at the 8448-token steady shape): it reads
`freqs.real/.imag` as stride-2 fp64 views of the complex128 tensor (each 8 B
element pulls a 16 B line) and inductor's codegen for the interleaved
(..., dc, 2) stack is scalar-ish. RoPE apply is a pure elementwise complex
multiply — no reductions — so a hand kernel can reproduce the EXACT same fp64
operations per element (o_r = xr*fr - xi*fi; o_i = xr*fi + xi*fr, then a
single fp64->bf16 round, the same cast the inductor kernel performs) while
moving all data as packed 32-bit words:

  * x / out: bf16 pairs loaded/stored as ONE u32 per complex pair (fully
    coalesced; bf16->fp32 by bit-shift is exact),
  * freqs: read directly from the contiguous complex128 storage as adjacent
    fp64 (re, im) pairs — no strided .real/.imag views.

Ideal traffic ~61 MB/call -> ~20-30 us at HBM3 rates (vs 143 us).
Bit-exactness is gated (kernel + E2E max|diff| == 0 vs the FUSE_ROPE path).
Default OFF; any failure falls back to the existing fused/eager paths.
"""
import torch

try:
    import triton
    import triton.language as tl
    _TRITON_OK = True
except Exception:  # pragma: no cover
    _TRITON_OK = False


if _TRITON_OK:

    @triton.jit
    def _rope_u32_kernel(XU, F, OU, TOTAL, S,
                         ND: tl.constexpr, DC: tl.constexpr,
                         BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        p = pid * BLOCK + tl.arange(0, BLOCK)
        m = p < TOTAL
        row = p // ND              # token row (b*S + s)
        s = row % S                # freqs row
        c = p % DC                 # complex lane within a head
        # x pair: one u32 = (bf16 imag << 16) | bf16 real
        xv = tl.load(XU + p, mask=m, other=0).to(tl.uint32, bitcast=True)
        xr = ((xv & 0xFFFF) << 16).to(tl.float32, bitcast=True).to(tl.float64)
        xi = ((xv >> 16) << 16).to(tl.float32, bitcast=True).to(tl.float64)
        # freqs pair: adjacent fp64 (re, im) in the complex128 storage
        fo = (s * DC + c) * 2
        fr = tl.load(F + fo, mask=m, other=0.0)
        fi = tl.load(F + fo + 1, mask=m, other=0.0)
        # same fp64 expressions as the fused impl; torch casts fp64->bf16
        # THROUGH fp32 (c10::BFloat16 is constructed from float), so match
        # that double-rounding chain exactly for bit-equality
        o_r = (xr * fr - xi * fi).to(tl.float32).to(tl.bfloat16)
        o_i = (xr * fi + xi * fr).to(tl.float32).to(tl.bfloat16)
        r16 = o_r.to(tl.uint16, bitcast=True).to(tl.uint32)
        i16 = o_i.to(tl.uint16, bitcast=True).to(tl.uint32)
        tl.store(OU + p, (r16 | (i16 << 16)).to(tl.int32, bitcast=True), mask=m)


def rope_apply_triton(x, freqs, num_heads):
    """Drop-in for rope_apply's fused path. x: (B,S,n*d) bf16 contiguous,
    freqs: (S,1,d//2) complex128 contiguous. Raises on unsupported input;
    the caller falls back to the fused/eager paths."""
    assert _TRITON_OK
    B, S, D = x.shape
    n = num_heads
    dc = (D // n) // 2
    assert x.dtype == torch.bfloat16 and x.is_contiguous()
    assert freqs.dtype == torch.complex128 and freqs.is_contiguous()
    assert freqs.shape[0] == S and freqs.shape[-1] == dc
    o = torch.empty_like(x)
    xu = x.view(torch.int32)          # (B,S,D/2) u32 pairs, same storage
    ou = o.view(torch.int32)
    fv = torch.view_as_real(freqs)    # (S,1,dc,2) fp64 view of the storage
    total = B * S * n * dc
    BLOCK = 2048
    _rope_u32_kernel[(triton.cdiv(total, BLOCK),)](
        xu, fv, ou, total, S, ND=n * dc, DC=dc, BLOCK=BLOCK, num_warps=8)
    return o
