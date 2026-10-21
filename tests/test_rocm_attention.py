import importlib.util
from pathlib import Path
import unittest
import torch

spec = importlib.util.spec_from_file_location(
    "rocm_attention",
    Path(__file__).resolve().parents[1] / "diffsynth/models/rocm_attention.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@unittest.skipUnless(
    torch.cuda.is_available() and torch.version.hip, "Requires ROCm GPU"
)
class SparseAttentionTests(unittest.TestCase):
    def test_compile_limit_does_not_fall_back(self):
        from torch.nn.attention.flex_attention import flex_attention, BlockMask
        with torch._dynamo.config.patch(recompile_limit=0):
            compiled = torch.compile(flex_attention, dynamic=False, fullgraph=True)
            q = torch.randn(1, 1, 128, 128, device="cuda")
            counts = torch.ones(1, 1, 1, device="cuda", dtype=torch.int32)
            indices = torch.zeros(1, 1, 1, 1, device="cuda", dtype=torch.int32)
            mask = BlockMask.from_kv_blocks(torch.zeros_like(counts), indices, counts, indices, BLOCK_SIZE=128, seq_lengths=(128, 128))
            with self.assertRaises(torch._dynamo.exc.FailOnRecompileLimitHit):
                compiled(q, q, q, block_mask=mask)

    def test_decoder_cache_hook_lifecycle(self):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples/WanVSR"))
        from utils.TCDecoder import TAEHV, MemBlock
        decoder = TAEHV.__new__(TAEHV)
        torch.nn.Module.__init__(decoder)
        decoder.decoder = torch.nn.Sequential(MemBlock(4, 4), MemBlock(4, 4))
        decoder.set_rocm_cache_release()
        self.assertEqual(len(decoder._rocm_cache_handles), 2)
        decoder.set_rocm_cache_release()
        self.assertEqual(sum(len(m._forward_hooks) for m in decoder.decoder), 2)
        decoder.set_rocm_cache_release(False)
        self.assertEqual(sum(len(m._forward_hooks) for m in decoder.decoder), 0)

    def test_masks_and_streaming_lengths(self):
        torch.manual_seed(123)
        for dtype in (torch.float32, torch.bfloat16):
            for nq, nk in ((256, 384), (384, 384), (129, 257)):
                with self.subTest(dtype=dtype, nq=nq, nk=nk):
                    q = torch.randn(1, 2, nq, 128, device="cuda", dtype=dtype)
                    k = torch.randn(1, 2, nk, 128, device="cuda", dtype=dtype)
                    v = torch.randn_like(k)
                    mask = (
                        torch.rand(
                            1, 2, (nq + 127) // 128, (nk + 127) // 128, device="cuda"
                        )
                        > 0.4
                    )
                    mask[:, 0, 0] = False
                    actual = module.sparse_attention(q, k, v, mask)
                    expected = module.reference_attention(q, k, v, mask)
                    tol = 0.02 if dtype == torch.bfloat16 else 0.001
                    torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)
                    self.assertTrue(torch.isfinite(actual).all())
                    self.assertEqual(actual[:, 0, :128].abs().max().item(), 0)

    def test_official_streaming_and_aspect_switch(self):
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from diffsynth.models.wan_video_dit import SelfAttention

        model = SelfAttention(256, 2).cuda().to(torch.bfloat16).eval()
        cache_k = cache_v = None
        with torch.no_grad():
            for f, h, w in (
                (6, 16, 24),
                (2, 16, 24),
                (2, 16, 24),
                (6, 24, 16),
                (6, 16, 24),
            ):
                if f == 6:
                    cache_k = cache_v = None
                x = torch.randn(1, f * h * w, 256, device="cuda", dtype=torch.bfloat16)
                freqs = torch.ones(
                    f * h * w, 1, 64, device="cuda", dtype=torch.complex128
                )
                y, cache_k, cache_v = model(
                    x,
                    freqs,
                    f=f,
                    h=h,
                    w=w,
                    topk=10,
                    kv_len=3,
                    is_stream=True,
                    pre_cache_k=cache_k,
                    pre_cache_v=cache_v,
                    local_range=3,
                )
                self.assertEqual(y.shape, x.shape)
                self.assertTrue(torch.isfinite(y).all())
                self.assertEqual(cache_k.shape[0], 18)
                self.assertEqual(model.local_attn_mask_h, h // 8)
                self.assertEqual(model.local_attn_mask_w, w // 8)


class OfficialMaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import ast
        import math
        from einops import rearrange

        path = Path(__file__).resolve().parents[1] / "diffsynth/models/wan_video_dit.py"
        tree = ast.parse(path.read_text())
        names = {
            "generate_draft_block_mask",
            "build_local_block_mask_shifted_vec_normal_slide",
        }
        functions = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in names
        ]
        cls.scope = {"torch": torch, "math": math, "rearrange": rearrange}
        exec(
            compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"),
            cls.scope,
        )

    def test_grouped_mask_matches_original(self):
        generate = self.scope["generate_draft_block_mask"]
        for device in ("cpu", "cuda"):
            if device == "cuda" and not torch.cuda.is_available():
                continue
            for dtype in (torch.float32, torch.bfloat16):
                for ties in (False, True):
                    torch.manual_seed(42)
                    q = torch.randn(12, 128, 256, device=device, dtype=dtype)
                    k = torch.randn(18, 128, 256, device=device, dtype=dtype)
                    if ties:
                        q.fill_(1)
                        k.fill_(1)
                    locality = torch.tensor([[True, False, True], [False, True, True]], device=device)
                    expected = generate(1, 2, 3, q, k, topk=3, local_attn_mask=locality)
                    actual = generate(1, 2, 3, q, k, topk=3, local_attn_mask=locality, _force_grouped=True)
                    self.assertTrue(torch.equal(actual, expected))

    def test_clipped_locality(self):
        mask = self.scope["build_local_block_mask_shifted_vec_normal_slide"](4, 5, 3, 3)
        self.assertEqual(mask[0].sum().item(), 4)
        self.assertEqual(mask[6].sum().item(), 9)
        self.assertTrue(mask.diagonal().all())

    def test_strict_threshold_ties_and_global_selection(self):
        generate = self.scope["generate_draft_block_mask"]
        q = torch.ones(4, 128, 128)
        k = torch.ones(6, 128, 128)
        locality = torch.ones(2, 2, dtype=torch.bool)
        mask = generate(1, 1, 2, q, k, topk=3, local_attn_mask=locality)
        self.assertEqual(mask.shape, (1, 1, 4, 6))
        self.assertEqual(mask.sum().item(), 0)
        torch.manual_seed(42)
        q, k = torch.randn_like(q), torch.randn_like(k)
        mask = generate(1, 1, 2, q, k, topk=3, local_attn_mask=locality)
        self.assertEqual(mask[:, :, :2].sum().item(), 3)
        self.assertEqual(mask[:, :, 2:].sum().item(), 3)


if __name__ == "__main__":
    unittest.main()
