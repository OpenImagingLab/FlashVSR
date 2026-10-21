"""ROCm execution of the official 128-token LCSA block mask."""

import os
from functools import lru_cache

import torch
import torch.nn.functional as F


def reference_attention(q, k, v, mask):
    """Gather all allowed keys per query block before a single softmax."""
    out = torch.zeros_like(q)
    for b in range(q.shape[0]):
        for h in range(q.shape[1]):
            for row in range(mask.shape[2]):
                blocks = mask[b, h, row].nonzero().flatten()
                if blocks.numel() == 0:
                    continue
                indices = (
                    blocks[:, None] * 128 + torch.arange(128, device=q.device)
                ).flatten()
                indices = indices[indices < k.shape[2]]
                start, end = row * 128, min((row + 1) * 128, q.shape[2])
                out[b, h, start:end] = F.scaled_dot_product_attention(
                    q[b, h, start:end].float(),
                    k[b, h, indices].float(),
                    v[b, h, indices].float(),
                    dropout_p=0.0,
                ).to(q.dtype)
    return out


@lru_cache(maxsize=1)
def _compiled_attention():
    from torch.nn.attention.flex_attention import flex_attention

    return torch.compile(flex_attention, dynamic=False, fullgraph=True)


def sparse_attention(q, k, v, mask):
    expected = (
        q.shape[0],
        q.shape[1],
        (q.shape[2] + 127) // 128,
        (k.shape[2] + 127) // 128,
    )
    if mask.shape != expected or mask.dtype != torch.bool or mask.device != q.device:
        raise ValueError(f"Expected boolean block mask {expected} on {q.device}")
    backend = os.environ.get("FLASHVSR_ROCM_ATTENTION", "flex")
    if backend == "reference":
        return reference_attention(q, k, v, mask)
    if backend != "flex":
        raise ValueError("FLASHVSR_ROCM_ATTENTION must be flex or reference")
    from torch.nn.attention.flex_attention import BlockMask

    counts = mask.sum(-1).to(torch.int32)
    indices = torch.argsort(
        mask.to(torch.int32), dim=-1, descending=True, stable=True
    ).to(torch.int32)
    # Logical blocks are completely allowed; tail tokens remain length-bounded.
    block_mask = BlockMask.from_kv_blocks(
        torch.zeros_like(counts),
        indices,
        counts,
        indices,
        BLOCK_SIZE=128,
        seq_lengths=(q.shape[2], k.shape[2]),
    )
    precision = os.environ.get("FLASHVSR_ROCM_PRECISION", "native")
    if precision not in ("native", "fp32"):
        raise ValueError("FLASHVSR_ROCM_PRECISION must be native or fp32")
    if precision == "fp32":
        out = _compiled_attention()(
            q.float(), k.float(), v.float(), block_mask=block_mask
        ).to(q.dtype)
    else:
        out = _compiled_attention()(q, k, v, block_mask=block_mask)
    # Empty rows have no allowed keys and return zero, rather than NaN.
    valid = (counts > 0).repeat_interleave(128, dim=-1)[..., : q.shape[2]]
    return torch.where(valid[..., None], out, 0)
