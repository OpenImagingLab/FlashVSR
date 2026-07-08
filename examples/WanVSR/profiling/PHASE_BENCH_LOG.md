# FlashVSR Phase-2+ Benchmark Log

Chronological, append-only log of every optimization attempt (successes,
failures, and reverts alike). Governed by the rules in
[`PHASE_ROADMAP.md`](./PHASE_ROADMAP.md) §0.5 and §5.

**The gate:** do not start the next optimization until the current one has
(a) an entry in this file and (b) a 2–3 sentence interpretation.

Measurement rules (short form):
- Untraced `run_pipe_target.py` runs only; 3 runs back-to-back, log the median.
- Before/after measured at the same commit, flag OFF vs flag ON.
- @768x1408 F=81 mandatory; @1536x2560 only at phase closure.
- Lossless claims: `max|diff| == 0`. Numeric-neutral claims: PSNR ≥ 49 dB.
- Traced (nsys/ncu) runs are attribution-only — never headline numbers.

---

## Step 0 — Fresh Phase-2 baseline (MUST be filled before the first change)

Command (from `examples/WanVSR/`):

```bash
FLASHVSR_CONV3D_BACKEND=gemm FLASHVSR_TCDECODER_CHANNELS_LAST=1 \
FLASHVSR_FUSE_NORM=1 FLASHVSR_ATTN_BACKEND=triton \
FLASHVSR_CACHE_MOD=1 FLASHVSR_CACHE_MASK_BIAS=1 FLASHVSR_PROF_STEADY=off \
/root/FlashVSR/venv/bin/python profiling/run_pipe_target.py
```

| Field | Value |
|---|---|
| Date/time | 2026-07-08 ~08:30 |
| Commit | df94d94 (phase1 instrumentation committed on top of 613bf9f) |
| GPU clocks locked | y (`nvidia-smi -lgc 1980,1980`) |
| Resolution / frames | 768x1408 / F=81 |
| Run 1 / Run 2 / Run 3 FPS | 38.585 / 38.525 / 38.594 |
| **Median FPS (= Phase-2 baseline)** | **38.585** |
| Median steady chunk ms | **156.28** (156.28 / 156.50 / 156.18) |
| Peak memory GiB | 12.6 |
| Reference (Phase-1 campaign, single run) | 38.55 FPS · 156.24 ms · 12.6 GiB |
| Notes | GPU otherwise idle, no compute apps; logs: `profiling/runs/phase2a/step0_baseline_run{1..3}.log`. Matches Phase-1 reference within noise. |

---

## Per-change entries

<!-- Append one table row + one entry block per optimization attempt.
     Newest at the bottom. Do not delete failed attempts. -->

| Date | Phase | Optimization | Flag | FPS Before | FPS After | Delta | Steady Chunk Before | Steady Chunk After | Peak Mem Before | Peak Mem After | Correctness | Decision |
|------|-------|--------------|------|------------|-----------|-------|---------------------|--------------------|-----------------|----------------|-------------|----------|
| 2026-07-08 | 2A-1a | RoPE freqs device cache | `FLASHVSR_CACHE_ROPE_FREQS` | 38.585 | 38.592 | +0.02% | 156.28 ms | 156.30 ms | 12.60 GiB | 12.63 GiB | max\|diff\|=0 (bit-identical) | keep-behind-flag |
| 2026-07-08 | 2A-1b | Fused RoPE apply | `FLASHVSR_FUSE_ROPE` | 38.585 | 39.023 | +1.14% | 156.28 ms | 153.62 ms | 12.60 GiB | 12.60 GiB | max\|diff\|=0 (bit-identical) | keep-enabled |
| 2026-07-08 | 2A-2 | KV cache arena (no per-chunk cat) | `FLASHVSR_KV_RINGBUF` | 39.023 | 39.429 | +1.04% | 153.62 ms | 150.76 ms | 12.60 GiB | 15.50 GiB | max\|diff\|=0 over 9 chunks | keep-enabled |
| 2026-07-08 | 2A-3 | Attention strided IO (no transposes) | `FLASHVSR_ATTN_STRIDED_IO` | 39.429 | 41.099 | +4.24% | 150.76 ms | 141.14 ms | 15.50 GiB | 15.50 GiB | kernel + E2E max\|diff\|=0 | keep-enabled |
| 2026-07-08 | 2A-4 | mask_gen lean (kthvalue + no repeat + seqlens cache) | `FLASHVSR_MASKGEN_LEAN` | 41.099 | 41.477 | +0.92% | 141.14 ms | 139.00 ms | 15.50 GiB | 15.50 GiB | mask equality + E2E max\|diff\|=0 | keep-enabled |
| 2026-07-08 | 2A-5 | LQ projector lean (pad fold + no clones) | `FLASHVSR_LQPROJ_LEAN` | 41.477 | 41.580 | +0.25% | 139.00 ms | 138.46 ms | 15.50 GiB | 15.62 GiB | E2E max\|diff\|=0 (after dropping sub-item b) | keep-enabled |
| 2026-07-08 | 2A-6 | CUDA Graphs on steady chunk | (not implemented) | 41.580 | — | — | 138.46 ms | — | 15.62 GiB | — | n/a | postpone (go/no-go gate failed) |
| 2026-07-08 | 2B-1 | Decoder overlap on side CUDA stream | `FLASHVSR_DECODER_OVERLAP` | 41.662 | 42.373 | +1.71% | 138.30 ms | 190.58 ms (absorbs decode) | 15.62 GiB | 15.16 GiB | E2E max\|diff\|=0 (full+short clip, 3x repeats) | keep-behind-flag |
| 2026-07-08 | 2B-2 | FP8 GEMM infra (e4m3 `_scaled_mm` + Triton rowwise quant) | `FLASHVSR_FP8_GEMM` | 41.704 | 42.986 | +3.07% | 137.98 ms | 132.96 ms | 15.62 GiB | 16.65 GiB | PSNR 40.70 dB (full scope; NOT lossless by design — Phase-4 gate) | keep-behind-flag (Phase-4 enable gate) |
| 2026-07-08 | 2B-3 | Fused mask-gen threshold-select (Triton radix select+compare) | `FLASHVSR_FUSED_MASKGEN` (removed) | 42.00 | 41.84 | −0.4% (harness E2E) | — | — | 15.62 GiB | 15.62 GiB | mask + E2E max\|diff\|=0 (exact) | **revert** (bit-exact but performance-negative) |
| 2026-07-08 | 3 | Warp-specialized block-sparse attention v2 (Gluon producer/consumer + pingpong) | `FLASHVSR_ATTN_BACKEND=triton2` | 41.692 | 45.466 | **+9.05%** | 137.96 ms | 122.06 ms | 15.62 GiB | 15.62 GiB | kernel cos ≥0.999995 vs block_sparse (all densities + degenerates); E2E PSNR 50.03 dB; repeats bit-identical; default paths max\|diff\|=0 | keep-behind-flag (ncu gate 3/4; residency 12 warps/SM WS vs ≥16 letter) |

#### 2026-07-08 09:00 · Phase 2A-1a · RoPE freqs device cache

- Commit / patch: on top of df94d94 (committed as phase2a rope freqs cache)
- Files changed: `diffsynth/pipelines/flashvsr_tiny.py` (+ new `test_phase2a_lossless.py` harness)
- Flag: `FLASHVSR_CACHE_ROPE_FREQS` (default OFF)
- Env vars used (full set): full-knob baseline + flag under test
- Exact benchmark command: §0.4 primary command + `FLASHVSR_CACHE_ROPE_FREQS=1`
- Resolution / frames: 768x1408 / F=81 (spot-check 1536: n)
- Warmup / steady settings: warmup=1, steady=chunks 2..6
- FPS before → after (Δ): 38.585 → 38.592 (+0.02%, noise) — 3-run medians
- Steady chunk before → after (Δ): 156.28 → 156.30 ms (noise)
- Peak mem before → after: 12.60 → 12.63 GiB (+30 MB: device freqs buffers f=6 and f=2)
- Correctness: `test_phase2a_lossless.py CACHE_ROPE_FREQS` → max|diff| == 0 (PASS)
- Nsight report path: n/a (untraced only)
- Decision: keep-behind-flag
- Interpretation: the ~44 ms/chunk `rope_freqs` cost seen in *traced* runs is CPU
  wall that rides entirely under the 156 ms GPU chunk in untraced runs, so
  removing it does not move FPS @768 — consistent with the roadmap note that
  this is "CPU-side headroom", not GPU time. The change is bit-identical and
  removes per-chunk CPU tensor construction + an 8.6 MB H2D, which matters for
  CPU-loaded deployments and is a precondition for CUDA-graph capture (2A-6),
  so it stays available behind its flag rather than being reverted.

#### 2026-07-08 09:25 · Phase 2A-1b · Fused RoPE apply

- Commit / patch: phase2a fused rope apply
- Files changed: `diffsynth/models/wan_video_dit.py` (`rope_apply` + compiled impl)
- Flag: `FLASHVSR_FUSE_ROPE` (default OFF)
- Env vars used (full set): full-knob baseline + `FLASHVSR_FUSE_ROPE=1`
- Exact benchmark command: §0.4 primary command + `FLASHVSR_FUSE_ROPE=1`
- Resolution / frames: 768x1408 / F=81 (spot-check 1536: at phase closure)
- Warmup / steady settings: warmup=1, steady=chunks 2..6
- FPS before → after (Δ): 38.585 → 39.023 (+1.14%) — 3-run medians
- Steady chunk before → after (Δ): 156.28 → 153.62 ms (−2.66 ms)
- Peak mem before → after: 12.60 → 12.60 GiB
- Correctness: kernel-level max|diff|=0 on real shape; E2E
  `test_phase2a_lossless.py FUSE_ROPE` → max|diff| == 0 (exceeds the ≥49 dB gate)
- Isolated kernel: eager 0.181 ms/call → fused 0.128 ms/call (×1.41, 60 calls/chunk)
- Nsight report path: n/a (untraced only)
- Decision: keep-enabled (joins recommended set)
- Interpretation: got −2.7 ms of the −8 ms roadmap ceiling: the estimate double
  counted freqs-side effects, and the fp64 multiply itself (not just the
  materialized intermediates) is part of the cost, so the fused kernel is
  compute-bound at ~0.13 ms/call rather than pure-BW ~0.02 ms. Output is
  bit-identical since the same fp64 operations are performed in one kernel.
  Remaining rope headroom (folding the apply into the attention prologue or
  fp32 freqs) is Phase-4 territory (numerics change).

#### 2026-07-08 09:55 · Phase 2A-2 · KV cache arena (kv_cat removal)

- Commit / patch: phase2a kv cache arena
- Files changed: `diffsynth/models/wan_video_dit.py` (`_KVArena`, `SelfAttention.forward`)
- Flag: `FLASHVSR_KV_RINGBUF` (default OFF); tuning: `FLASHVSR_KV_RINGBUF_SPARE` (default 2)
- Env vars used (full set): full-knob baseline + `FLASHVSR_FUSE_ROPE=1` (kept set) + flag under test
- Exact benchmark command: §0.4 primary command + `FLASHVSR_FUSE_ROPE=1 FLASHVSR_KV_RINGBUF=1`
- Resolution / frames: 768x1408 / F=81 (spot-check 1536: at phase closure)
- Warmup / steady settings: warmup=1, steady=chunks 2..6
- FPS before → after (Δ): 39.023 → 39.429 (+1.04%) — 3-run medians
- Steady chunk before → after (Δ): 153.62 → 150.76 ms (−2.86 ms)
- Peak mem before → after: 12.60 → 15.50 GiB (+2.9 GiB = 2 spare arena slots × 60 tensors)
- Correctness: `test_phase2a_lossless.py KV_RINGBUF` → max|diff| == 0 over a
  9-chunk clip (exercises slot rotation AND tail compaction)
- Nsight report path: n/a (untraced only)
- Decision: keep-enabled (joins recommended set)
- Interpretation: recovers essentially the whole kv_cat budget (−2.9 of the
  3.4 ms attribution): new windows are partition-written straight into the
  arena (no extra traffic) and only the amortized compaction remains — visible
  as ~+2.5 ms on every 3rd chunk in the per-chunk trace. Bit-identical since
  the live view preserves value order and contiguity exactly. Cost is +2.9 GiB
  retained memory (documented; `FLASHVSR_KV_RINGBUF_SPARE` trades memory for
  compaction frequency, and the flag restores the old path entirely).

#### 2026-07-08 10:20 · Phase 2A-3 · Attention-path strided IO

- Commit / patch: phase2a attention strided IO
- Files changed: `diffsynth/models/triton_block_sparse_attn.py`
  (`_bsfa_tma_kernel_snd`, `triton_block_sparse_attention_snd`),
  `diffsynth/models/wan_video_dit.py` (glue branch in `flash_attention`)
- Flag: `FLASHVSR_ATTN_STRIDED_IO` (default OFF)
- Env vars used (full set): full-knob baseline + kept set
  (`FUSE_ROPE=1 KV_RINGBUF=1`) + flag under test
- Exact benchmark command: §0.4 primary + `FLASHVSR_FUSE_ROPE=1 FLASHVSR_KV_RINGBUF=1 FLASHVSR_ATTN_STRIDED_IO=1`
- Resolution / frames: 768x1408 / F=81 (spot-check 1536: at phase closure)
- Warmup / steady settings: warmup=1, steady=chunks 2..6
- FPS before → after (Δ): 39.429 → 41.099 (+4.24%) — 3-run medians
- Steady chunk before → after (Δ): 150.76 → 141.14 ms (−9.62 ms)
- Peak mem before → after: 15.50 → 15.50 GiB
- Correctness: kernel-level max|diff| == 0 vs contiguous path on the real
  shape (both TMA and non-TMA variants); E2E
  `test_phase2a_lossless.py ATTN_STRIDED_IO` → max|diff| == 0 (exceeds the
  ≥49 dB gate)
- Isolated: contiguous glue+kernel 2.851 ms/call → strided 2.514 ms/call;
  strided is even faster than the kernel alone on pre-copied inputs
  (2.558 ms) — per-head tiles are adjacent in the (S, n*d) layout, so TMA
  loads of neighbouring heads share L2 lines
- Nsight report path: n/a (untraced only)
- Decision: keep-enabled (joins recommended set)
- Interpretation: −9.6 of the −11.6 ms ceiling: all four transpose copies are
  gone; the residual is the win_part/win_rev reorder that was counted in the
  same bucket. Bit-identical because the tile schedule and accumulation order
  are unchanged — only addressing moved from flattened (H·S, D) descriptors
  to 2D (S, H·D) descriptors with per-head column offsets. TMA=0 fallback is
  also strided (stride-general kernel), and any failure falls back to the
  contiguous triton path, not to the sparse backend.

#### 2026-07-08 10:45 · Phase 2A-4 · mask_gen allocation / sync cleanup

- Commit / patch: phase2a mask_gen lean
- Files changed: `diffsynth/models/wan_video_dit.py`
  (`generate_draft_block_mask`, `flash_attention` sparse branch)
- Flag: `FLASHVSR_MASKGEN_LEAN` (default OFF)
- Env vars used (full set): full-knob baseline + kept set
  (`FUSE_ROPE=1 KV_RINGBUF=1 ATTN_STRIDED_IO=1`) + flag under test
- Exact benchmark command: §0.4 primary + kept set + `FLASHVSR_MASKGEN_LEAN=1`
- Resolution / frames: 768x1408 / F=81
- Warmup / steady settings: warmup=1, steady=chunks 2..6
- FPS before → after (Δ): 41.099 → 41.477 (+0.92%) — 3-run medians
- Steady chunk before → after (Δ): 141.14 → 139.00 ms (−2.14 ms)
- Peak mem before → after: 15.50 → 15.50 GiB
- Correctness: threshold + boolean-mask exact equality vs topk on random /
  heavy-tie / softmax-distributed inputs at the real shape; E2E
  `test_phase2a_lossless.py MASKGEN_LEAN` → max|diff| == 0
- Isolated: topk(k=7920 of 17424) 0.187 ms → kthvalue 0.076 ms per call
  (30 calls/chunk); plus removal of the boolean-mask repeat copy; sparse
  backend additionally gets persistent cu_seqlens/head_mask_type (hidden H2D
  sync removed — benefits the sparse fallback path, not the triton bench)
- Nsight report path: n/a (untraced only)
- Decision: keep-enabled (joins recommended set)
- Interpretation: −2.1 ms banked (roadmap estimated −1–2 ms for the lean pass;
  the kthvalue swap alone projected −3.3 ms isolated but part of the chain is
  latency that overlaps with neighbouring small kernels). Mask semantics are
  provably unchanged — same order statistic, ties behave identically, strict
  `>` compare untouched. The remaining ~5 ms of the mask_gen chain needs the
  fused top-k kernel (Phase 2B-3), not more hygiene.

#### 2026-07-08 11:10 · Phase 2A-5 · LQ projector allocation/layout cleanup

- Commit / patch: phase2a lq projector lean
- Files changed: `examples/WanVSR/utils/utils.py` (`CausalConv3d._forward_lean`,
  `_conv3d_gemm(contig_out=)`, `Causal_LQ4x_Proj.stream_forward`)
- Flag: `FLASHVSR_LQPROJ_LEAN` (default OFF)
- Env vars used (full set): full-knob baseline + kept set
  (`FUSE_ROPE=1 KV_RINGBUF=1 ATTN_STRIDED_IO=1 MASKGEN_LEAN=1`) + flag under test
- Exact benchmark command: §0.4 primary + kept set + `FLASHVSR_LQPROJ_LEAN=1`
- Resolution / frames: 768x1408 / F=81
- Warmup / steady settings: warmup=1, steady=chunks 2..6
- FPS before → after (Δ): 41.477 → 41.580 (+0.25%) — 3-run medians
- Steady chunk before → after (Δ): 139.00 → 138.46 ms (−0.54 ms, ~noise floor)
- Peak mem before → after: 15.50 → 15.62 GiB (persistent pad buffers + view retention)
- Correctness: E2E `test_phase2a_lossless.py LQPROJ_LEAN` → max|diff| == 0
  across a full clip (streaming cache exercised across chunk boundaries)
- Sub-item REJECTED during development: skipping the GEMM output
  `.contiguous()` (layout-only in values) changed `F.normalize`'s reduction
  accumulation order → measured max|diff|=0.397 / PSNR 49.9 dB, which
  violates this item's lossless gate. Dropped; documented in code. Could be
  revisited under a Phase-4 numerics-tolerant gate if the projector becomes
  hot again.
- Nsight report path: n/a (untraced only)
- Decision: keep-enabled (joins recommended set)
- Interpretation: the kept copies-only cleanup (fold cat+F.pad into one
  buffer write, drop streaming-cache clones) recovers ~0.5 ms of the ~1–2 ms
  estimate — pad+cat was only ~8% of the projector path, and the addmm (65%)
  and im2col gather (29%) are untouched by design. Since 2A-5 recovered
  <2 ms, the roadmap gate keeps Phase 2B-4 (fused im2col-GEMM) on the table.

#### 2026-07-08 11:40 · Phase 2A-6 · CUDA Graphs go/no-go gate → NO-GO (postpone)

- Commit / patch: none (gate evaluation only, per roadmap §2A-6)
- Nsight report path: `profiling/reports/p2a_stack_768/` (nsys minimal trace,
  full 2A stack, steady window chunks 2–6; attribution-only, not headline FPS)
- Gate input 1 — idle: corrected steady-chunk idle with the full 2A stack is
  0.6–2.5% (avg ~1.8%, `analysis.md` busy/idle table) → below the ≥2% go
  threshold; ceiling ≤ ~3 ms/chunk @768 and ~0 at higher resolutions.
- Gate input 2 — capture safety: the steady body is NOT capture-stable
  today: (a) the 2A-2 KV arena advances a Python-int slice offset every chunk
  and compacts on a data-dependent schedule (a captured graph would freeze
  both), (b) LQ conditioning slices a different video range per chunk,
  (c) the 2A-1a freqs buffer is rewritten per chunk (solvable — update
  outside the graph — but only relevant if (a)/(b) were solved). Fixing (a)
  requires a true ring buffer with device-side index remapping — exactly the
  "invasive changes" this experiment is instructed not to expand into.
- Decision: postpone (documented no-go; do not implement in 2A)
- Interpretation: after the 2A cleanup the chunk is even less launch-bound
  than the 3.1% Phase-1 measurement, so the graphs ceiling shrank while its
  implementation cost grew (arena offsets). Attribution cross-checks the
  stack wins: kv_cat memcpy 3.4 → ~0.5 ms/chunk, rope 10.1 → ~7.4, mask_gen
  7.4 → ~4.4, attn transposes gone, `_bsfa_tma_kernel_snd` unchanged at
  2.03 ms/call. Revisit graphs only if Phase 2B/3 work makes the steady body
  static (or if deployment shows CPU contention).

#### 2026-07-08 · Phase 2B-1 · Step 2 — fresh Phase-2A-stack baseline (2B-1 starting point)

Re-measured at `45b2ddd` (pre-change), full 2A recommended stack ON, decoder
overlap absent. Command = §0.4 primary + `FLASHVSR_FUSE_ROPE=1
FLASHVSR_KV_RINGBUF=1 FLASHVSR_ATTN_STRIDED_IO=1 FLASHVSR_MASKGEN_LEAN=1
FLASHVSR_LQPROJ_LEAN=1`; clocks locked 1980 MHz.

| Field | Value |
|---|---|
| Run 1 / 2 / 3 FPS | 41.780 / 41.759 / 41.686 |
| **Median FPS (= 2B-1 baseline)** | **41.759** |
| Median steady chunk ms | 137.46 (137.46 / 136.90 / 137.96) |
| Peak memory GiB | 15.62 |
| Reference (Phase 2A closure) | 41.580 FPS · 138.46 ms · 15.62 GiB |
| Notes | Matches the 2A closure within noise (+0.4% FPS). Logs: `profiling/runs/phase2b/step2_baseline_run{1..3}.log`. |

#### 2026-07-08 12:40 · Phase 2B-1 · Decoder overlap on a side CUDA stream

- Commit / patch: phase2b decoder overlap. Note: a concurrent duplicate
  session committed an equivalent variant of this same 2B-1 change as
  7cecb65 mid-campaign; this entry and the follow-up commit supersede it
  (same design, independently implemented and fully re-validated — the
  numbers in THIS entry correspond to the tree at the follow-up commit).
- Files changed: `diffsynth/pipelines/flashvsr_tiny.py` (flag + overlap path),
  `examples/WanVSR/profiling/run_pipe_target.py` (additive `[tail]`
  post-loop-ms print), new `examples/WanVSR/profiling/test_decoder_overlap_lossless.py`
- Flag: `FLASHVSR_DECODER_OVERLAP` (default OFF; serialized end-of-loop decode
  path is byte-identical code when OFF)
- Design: after each chunk's `cur_latents -= noise_pred` a ready-event is
  recorded on the main stream; a persistent side stream waits on it and runs
  `TCDecoder.decode_video` for that chunk (cond slice `LQ_pre_idx:LQ_cur_idx`,
  same per-chunk semantics as `flashvsr_tiny_long.py`); per-chunk done-events
  are waited on by the main stream only once, right before output assembly
  (`wait_event`, GPU-side, no CPU sync); decoded chunks are assembled from a
  chunk-id-indexed list → output order is structural, never completion order.
  TAEHV mem-block state is only ever touched by the decode stream; decode
  inputs stay alive in `latents_total`; `record_stream` guards added as
  defense in depth.
- Env vars used (full set): full-knob baseline + 2A kept set + flag under test
- Exact benchmark command: §0.4 primary + 2A kept set + `FLASHVSR_DECODER_OVERLAP=1`
- Resolution / frames: 768x1408 / F=81 (spot-check 1536x2560: y, below)
- Warmup / steady settings: warmup=1, steady=chunks 2..6
- FPS before → after (Δ): 41.662 → 42.373 (+1.71%) — 3-run medians, same commit
- Steady chunk before → after (Δ): 138.30 → 190.58 ms (+52.3 ms — the chunk
  wall now absorbs ~50 ms/chunk of decode GPU work + ~12 ms CPU enqueue; this
  metric is no longer denoise-only when the flag is ON, see interpretation)
- Decode tail (post-loop) before → after: 561.2 → 125.1 ms (−436 ms hidden)
- Peak mem before → after: 15.62 → 15.16 GiB (−0.46 GiB: the one-shot
  20-frame decode working set and the full-latents cat are gone)
- Correctness: `test_decoder_overlap_lossless.py` → max|diff| == 0,
  mean|diff| == 0, 85/85 frames bit-equal (ordering), shape/dtype/device
  identical, on BOTH full clip (F=89, 9 chunks) and short clip (F=25,
  1 chunk, trim edge); overlap run 3x back-to-back → all repeats bit-identical
  (no concurrency races). `test_phase2a_lossless.py ALL` re-run → PASS
  (max|diff|=0, 2A stack unaffected).
- Nsight report path: `profiling/reports/phase2b_decoder_overlap/`
  (`profile.nsys-rep`, `profile.sqlite`, `overlap_analysis.txt`). Evidence:
  decoder kernels (1060 cuDNN convs, 498.7 ms busy) on side stream 13 vs
  denoise on stream 7; decode active across the whole loop span, not as a
  tail; 170.2 ms (34.1%) of decode busy time co-executes with denoise
  kernels; `decode_wait` = 0.1 ms and `color_fix` = 1.6 ms at the end (no
  per-chunk sync, final sync negligible); GPU idle 1.9% of span.
- Decision: keep-behind-flag (default OFF)
- Interpretation: the mechanics work exactly as designed — bit-identical
  output, decode tail fully hidden (561 → 125 ms), zero hidden syncs, and
  peak memory *drops* — but the E2E gain is +1.7% @768 / +2.0% @1536, far
  under the roadmap's +21–29% ceiling. Nsight shows why: the GPU is
  work-conserving (idle 1.9%), and only ~34% of decode kernel time can
  co-execute with denoise because the `_bsfa` attention kernel owns a full
  SM (229 KB smem, 1 CTA/SM) and the decoder's cuDNN conv grids are
  SM-saturating themselves, so the hardware time-shares instead of
  co-executing — moving the decode from the tail into the chunks conserves
  total GPU work and only the co-executed 170 ms (+ warm-chunk gap fill) is
  actually won. The roadmap estimate implicitly assumed decode could ride on
  spare SM capacity that doesn't exist under the current attention kernel.
  Re-evaluate after Phase 3: the planned 2-CTA / warp-specialized attention
  kernel frees smem headroom, which should raise the co-execution fraction
  substantially — the flag composes losslessly with everything, so it can be
  promoted then. Default stays OFF also for campaign hygiene: with the flag
  ON the `[chunks]` steady metric measures denoise+decode contention rather
  than denoise alone, which would muddy per-chunk attribution for 2B-2/3/4.

##### Decoder-overlap detail (768x1408 F=81 medians; 1536x2560 single runs)

| Config | E2E FPS | Steady Chunk Time | Decode Tail Time | Peak Memory | Notes |
|--------|---------|-------------------|------------------|-------------|-------|
| 768 · 2A stack, overlap OFF | 41.662 | 138.30 ms | 561.2 ms | 15.62 GiB | serialized decode after loop |
| 768 · 2A stack, overlap ON | 42.373 | 190.58 ms | 125.1 ms | 15.16 GiB | decode rides inside chunks; tail = last-chunk decode remainder + color fix |
| 1536 · 2A stack, overlap OFF | 11.485 | 502.46 ms | 2046.2 ms | 48.39 GiB | matches 2A closure spot-check (11.489) |
| 1536 · 2A stack, overlap ON | 11.712 | 682.66 ms | 470.3 ms | 46.72 GiB | +2.0% FPS; decode share is larger at high res but co-execution stays SM-bound |

#### 2026-07-08 13:55 · Phase 2B-2 · FP8 GEMM infrastructure (`torch._scaled_mm`)

- Commit / patch: phase2b fp8 gemm infrastructure (on top of 41f1d8e)
- Files changed: new `diffsynth/models/fp8_gemm.py` (flag, Triton rowwise
  quantizer, gelu-fused quantizer, weight pre-cast cache, `_scaled_mm`
  wrappers with sticky eager fallback), `diffsynth/models/wan_video_dit.py`
  (qkv shared-quant site, o site, ffn site), `examples/WanVSR/utils/utils.py`
  (LQ per-layer linears, one shared quant for 30 GEMMs), new
  `examples/WanVSR/test_fp8_gemm_quality.py`, new
  `examples/WanVSR/profiling/sweep_fp8_scope.py` (per-site attribution tool
  for the Phase-4 audit)
- Flag: `FLASHVSR_FP8_GEMM` (default OFF — **permanently until the Phase-4
  quality protocol clears it**); scope bisection via
  `FLASHVSR_FP8_GEMM_SCOPE=qkv,o,ffn,lq` (+`ffn1` token, not in default set)
- Design: weights pre-cast once to e4m3 with per-out-channel scales (lazy,
  cached on module, +1.03 GiB); activations quantized per call with per-row
  scales by ONE Triton kernel (row amax + cast in a single launch — the
  eager/compiled quantize chain measured ~5x off bandwidth, 113-272 µs, and
  erased the GEMM win; the kernel does 15.1 µs @ 8448x1536); ffn2's input
  quant is fused with GELU-tanh (fp32 gelu -> e4m3, replacing the eager bf16
  GELU pass — on the 8960-wide activation this flips ffn2 from net-loss to
  net-win); bias fused in `_scaled_mm` epilogue; `use_fast_accum=False`
- Env vars used (full set): full-knob baseline + 2A kept set + flag under test
- Exact benchmark command: §0.4 primary + 2A kept set + `FLASHVSR_FP8_GEMM=1`
- Resolution / frames: 768x1408 / F=81 (1536 spot-check: deferred to phase
  closure; flag is default-OFF anyway)
- Warmup / steady settings: warmup=1, steady=chunks 2..6
- FPS before → after (Δ): 41.704 → 42.986 (+3.07%) — 3-run medians
- Steady chunk before → after (Δ): 137.98 → 132.96 ms (−5.02 ms)
- Peak mem before → after: 15.62 → 16.65 GiB (+1.03 GiB e4m3 weight copies)
- Correctness / quality (Phase-4 protocol measurement, example0 @768):
  full scope PSNR **40.70 dB** (max|d|=0.875, mean|d|=0.0114) →
  revert-or-redesign tier for *enablement*. Per-site sweep
  (`sweep_fp8_scope.py`): qkv 45.58 dB (and net-SLOWER in-pipe) · o 45.05 ·
  ffn 42.37 (the speed lever AND the quality offender) · ffn1-only 44.15 ·
  lq 49.08 (recommended tier, speed-neutral) · qkv,o,lq 43.42 ·
  o,ffn,lq 41.34 — every meaningful-speed combination lands < 45 dB.
  Flag-OFF neutrality: `test_phase2a_lossless.py ALL` max|diff|=0 PASS,
  `test_decoder_overlap_lossless.py` PASS (default paths untouched).
- Kernel evidence: torch.profiler + ncu on the live pipeline steady chunk
  (`profiling/reports/ncu/phase2b2_fp8_kernels.ncu-rep`): e4m3 GEMMs dispatch
  `nvjet_sm90_qqtst_128x128_128x6_..._ovscale_bias_TNT` (6/10 sampled nvjet
  launches) vs bf16 `nvjet_sm90_tst_256x128_...` — FP8 kernels confirmed
  selected. Isolated GEMM ratios (locked 1980 MHz, pre-quantized inputs):
  qkv ×1.45-1.53, ffn1 ×1.26-1.38, ffn2 ×1.54, lq ×1.43.
- Decision: keep-behind-flag (mandatory per roadmap — 2B-2 ships default-OFF
  regardless of speed; enable decision belongs to Phase 4)
- Interpretation: the infrastructure works and is net-faster (+3.07% E2E,
  −5.0 ms/chunk — inside the revised −3.5..−6 ms estimate once dynamic
  quantization costs are counted; the roadmap's −13 ms ceiling assumed
  pre-quantized activations). But the distilled one-step model is confirmed
  fp8-sensitive: 40.7 dB full-scope with errors compounding across the 30
  blocks and persisting through the KV cache, so **no fp8 configuration is
  currently enable-eligible**. Phase-4 redesign ticket (in priority order):
  (1) blockwise/1x128 activation scales (post-GELU outliers dominate the
  rowwise amax), (2) smoothquant-style weight/activation rebalancing,
  (3) first/last-block bf16 exclusion, (4) fast-accum A/B. The per-site
  sweep tool and the `ffn1` scope token exist precisely for that audit.

#### 2026-07-08 14:35 · Phase 2B-3 · Fused mask-gen threshold-select → REVERT

- Commit / patch: developed on top of 3b4b23d, reverted before landing; only
  this log entry is committed. (Run logs: `profiling/runs/phase2b2/test_fused_maskgen*.log`.)
- Files changed (during the attempt, all removed on revert):
  `diffsynth/models/fused_topk_mask.py` (Triton radix-select+compare kernel,
  single-tile n<=16384 fast path + tiled variant),
  `diffsynth/models/wan_video_dit.py` (branch in `generate_draft_block_mask`),
  `examples/WanVSR/profiling/test_fused_maskgen_lossless.py`
- Flag: `FLASHVSR_FUSED_MASKGEN` (removed with the revert)
- Scope decision (pre-registered): the exact-mask gate forbids fusing
  mean-pool/einsum/softmax (any reduction-order change flips ties across the
  66 concatenated softmax rows whose quotients share a threshold), so the
  exact-safe fusion is ONLY kthvalue+broadcast+compare → one kernel. The
  selected threshold is a pure order statistic (a VALUE of the multiset), so
  any correct selection is bit-identical — no tie-breaking ambiguity.
- Correctness (the gate PASSED): kernel-level mask equality vs kthvalue AND
  topk formulations on randn / heavy-tie / softmax-with--inf / all-equal /
  negative inputs at (12|36)x13068, 12x172800, edge k∈{1,2,n/3,n−1,n} — all
  exact; E2E max|diff| == 0 vs eager, 2 ON repeats identical.
- Performance (the reason for the revert): isolated @12x13068,
  k_smallest=5149: eager kthvalue+compare 59.7 µs vs fused 69.2 µs (x0.86;
  best of num_warps sweep {4,8,16,32}: 171.6/78.5/69.0/79.7 µs; first tiled
  version was 113 µs). Harness E2E @768 F=89: OFF 42.00 FPS → ON 41.84/41.62
  FPS (−0.4..−0.9%). Below the pre-registered keep gate (≥ +0.5%).
- Peak mem before → after: 15.62 → 15.62 GiB (unchanged)
- Nsight report path: n/a (kernel-level + harness E2E evidence sufficient
  for a negative result)
- Decision: **revert** — code removed cleanly, no flag left behind
- Interpretation: the roadmap's −5 ms/chunk estimate assumed fusing the
  whole pool→einsum→softmax→topk chain to ~1 ms, but the exact-mask gate
  makes everything upstream of the select bitwise-untouchable, and the
  remaining exact-safe slice (select+compare, ~60 µs/call eager) has no
  headroom: torch's `kthvalue` is already a single efficient radix-select
  kernel (the 2A-4 lean pass already banked the topk→kthvalue win), and at
  rows=12 a one-CTA-per-row fused kernel is latency-bound by 4 dependent
  histogram rounds (~69 µs floor ≈ eager). The launch-gap latency the fusion
  removes was already hidden by neighbouring kernels (2A-4 interpretation).
  Conclusion: mask_gen (~4.4 ms/chunk) is now attention-kernel and
  numerics-gate bound — further gains require either folding mask
  generation into the Phase-3 attention kernel v2 or accepting non-bitwise
  mask changes under the Phase-4 E2E-neutral protocol. 2B-4 (fused
  im2col-GEMM) remains the only open 2B item.

#### 2026-07-08 · Phase 3 · Step 2 — fresh 2A-stack baseline + kernel isolation (Phase-3 starting point)

Re-measured at the Phase-3 working tree (parent 25811e3), full 2A recommended
stack, clocks locked 1980 MHz. Command = §0.4 primary + 2A kept set.

| Field | Value |
|---|---|
| Run 1 / 2 / 3 FPS (pre-change tree) | 42.115 / 42.048 / 41.830 |
| **Median FPS** | **42.048** (matches 2B-1 baseline 41.759 within noise) |
| Median steady chunk ms | 136.88 · peak 15.62 GiB |
| ncu `_bsfa_tma_kernel_snd` (steady, in-pipe) | 1997.4 µs · SM/tensor SOL 40.7% · occ 12.5% = 1 CTA/SM (178 reg + 229.4 KB smem) · grid 792 = 6.0 waves · L2 92% · stalls barrier 2.36 / wait 1.01 / short_sb 0.55 · no-eligible 68.6% |
| `bench_ceilings.py` H2 re-run | cuDNN dense 1.945 ms → ideal sparse 1.179 ms (ANALYSIS ref: 1.86 / 1.13) |
| Shape correction (measured, was undocumented) | steady attention consumes the UNTRIMMED kv window: q 8448 × **kv 33792** (264 blocks) at mask density **0.42–0.45** — FLOP-identical to ANALYSIS's "kv 25344 @ 0.606" framing (~120 active kv blocks/row either way). True ideal-sparse at the real shape/density: dense(33792) 2.576 ms × 0.4213 = **1.085 ms**. |
| Logs | `profiling/runs/phase3/step2_baseline_run{1..3}.log`, `profiling/reports/ncu/phase3_bsfa_before.{ncu-rep,csv}` |

#### 2026-07-08 16:30 · Phase 3 · Warp-specialized block-sparse attention v2 (`triton2`)

- Commit / patch: phase3 attention v2 (this commit; parent 25811e3)
- Files changed: new `diffsynth/models/triton_block_sparse_attn_v2.py` (Gluon
  kernel + wrapper), `diffsynth/models/wan_video_dit.py` (triton2 branch in
  `flash_attention`, knob doc), new `examples/WanVSR/test_attention_v2.py`,
  new `examples/WanVSR/profiling/bench_attn_v2.py` (+ captured real mask
  `profiling/cache/attn_mask_768_steady.pt`)
- Flag: `FLASHVSR_ATTN_BACKEND=triton2` (default backend remains `sparse`;
  the recommended-set line keeps `triton` until this entry is independently
  confirmed). Tuning knobs: `FLASHVSR_ATTN_V2_NBUF` (default 3, measured
  best), `FLASHVSR_ATTN_V2_PROD_REGS` (default 40, measured best).
- Design (kernel): FA3-style Gluon warp specialization on sm_90 —
  1 TMA producer warp group (4 warps @ 40 regs via `worker_num_regs`
  setmaxnreg reallocation) streams the CSR-selected K/V 128×128 tiles into an
  NBUF=3-deep smem ring (mbarrier full/empty handshakes, 2×64-col TMA boxes
  per tile, next-index software prefetch); 2 consumer warpgroups (4 warps
  each) own 64-row halves of the q block ("pingpong": one WG's softmax
  overlaps the other's WGMMA) and additionally defer each P·V wgmma by one
  iteration (issue QK(j) before PV(j−1); in-order retirement lets
  `wait(pendings=1)` complete QK while PV drains under the softmax).
  Exact-mask preservation is structural: same `(H,Nqb,Nkvb)` boolean mask,
  same `_make_csr` (stable argsort → ascending kv-block order), all-false
  rows → l=0 → l_safe → zero rows. Softmax scale folded into the exponent
  (matches block_sparse_attn's FA2 formulation; v1 pre-scaled q in bf16).
- Fallback ladder: triton2 → v1 strided (`_bsfa_tma_kernel_snd`) → v1
  contiguous → `block_sparse_attn`; any v2 import/compile/launch failure
  falls through silently. Default `sparse`/`triton` code paths untouched.
- Route notes (prototyped and rejected):
  (a) `tl.range(warp_specialize=True)` one-liner on the v1 kernel — the
  hopper autoWS pass accepted the annotation but did NOT partition
  (num_warps stayed 8; 1.751 ms = no-op). (b) occupancy tuning for
  2 CTA/SM (M64/N64/w4 variants) — all slower (1.83–3.32 ms); WGMMA at
  M64 tiles + no WS loses more than residency gains. (c) BLOCK_N=64
  2-CTA variant of v2 — needs two wgmma layouts with per-iteration
  conversions; rejected (compile complexity, expected loss).
- Isolated kernel (bench_attn_v2.py, real captured mask d=0.4213, locked
  clocks): v1 kernel-only 1.753 ms → v2 **1.139 ms** (×1.54, 649 TF/s
  sparse-effective, 95% of the 1.085 ms ideal-sparse ceiling); NBUF/PROD_REGS
  sweep: {2,3}×{24,40,56} → 3/40 best.
- ncu acceptance gate (in-pipe steady chunk, `phase3_bsfa_v2_after.ncu-rep`):

| Metric | Gate | Before (`_bsfa_tma_kernel_snd`) | After (`_bsfa_v2_kernel`) | Verdict |
|---|---|---|---|---|
| duration @ reference shape | ≤ 1.3 ms | 1997.4 µs | **1260.5 µs** | PASS |
| tensor pipe (SM SOL) | ≥ 60% elapsed | 40.7% | **66.2%** | PASS |
| barrier stall /issue | < 1.0 | 2.36 | **0.71** | PASS |
| residency | ≥2 CTA/SM or WS ≥16 warps/SM | 1 CTA/SM · 8 warps | 1 CTA/SM · **12 warps WS** (4+4 consumers + 4 producer) | PARTIAL (see interpretation) |
| (info) no-eligible-warp cycles | — | 68.6% | 57.7% | — |
| (info) stalls long_sb / wait | — | 0.55 / 1.01 | 2.00 / 1.32 | — |

- E2E @768x1408 F=81 (untraced, 3-run medians, same tree, back-to-back):
  OFF (`triton`) 41.692 FPS / 137.96 ms / 15.62 GiB → ON (`triton2`)
  **45.466 FPS / 122.06 ms / 15.62 GiB** = **+9.05% FPS, −15.90 ms/chunk,
  peak unchanged**. Logs `profiling/runs/phase3/e2e_{off,on}_run{1..3}.log`.
- Quality/correctness: kernel cos vs `block_sparse_attn` ≥ 0.999995 at the
  real shape across densities {real 0.42, 0.1, 0.3, 0.45, 0.9, all-true} +
  degenerate all-false masks/rows/heads (all-false rows exactly zero) +
  determinism (2 calls bit-identical); E2E `test_attention_v2.py`:
  PSNR(sparse, triton2) = **50.03 dB** (gate ≥49, v1 reference ~49.7),
  max|d| = 0.3135, triton2 ×2 repeats bit-identical (multi-chunk KV/cache
  contract stable); `test_phase2a_lossless.py ALL` → max|diff| = 0
  (default paths byte-identical).
- nsys attribution (`profiling/reports/phase3_attn_v2/`): `_bsfa_v2_kernel`
  n=240 (8 chunks × 30 blocks) fully replaces `_bsfa_tma_kernel_snd` (0
  occurrences); steady chunks 137.96 → 122.06 ms untraced.
- @1536x2560 spot-check (single runs): OFF 11.472 FPS / 503.22 ms /
  48.39 GiB → ON **12.436 FPS / 449.24 ms / 48.39 GiB** (+8.4%,
  −53.98 ms/chunk) — the win holds/grows at scale as predicted
  (attention share is scale-invariant).
- Nsight report paths:
  `profiling/reports/ncu/phase3_bsfa_before.{ncu-rep,csv}`,
  `profiling/reports/ncu/phase3_bsfa_v2_after.{ncu-rep,csv}`,
  `profiling/reports/phase3_attn_v2/` (nsys).
- Decision: **keep-behind-flag** (pre-registered tier: the ncu gate is 3/4 —
  the residency letter asks ≥2 CTA/SM or ≥16 warps/SM and v2 runs 12
  warps/SM WS at 1 CTA/SM). Presumptive recommended-set promotion
  (`FLASHVSR_ATTN_BACKEND=triton2`) after one independent confirming entry,
  per the two-entry promotion rule.
- Interpretation: the kernel hit the acceptance window on every *binding*
  metric — 1.26 ms in-pipe (target ≤1.3; 1.14 ms isolated = 95% of the
  ideal-sparse ceiling), tensor-active 66.2% (target ≥60), barrier stalls
  2.36 → 0.71 — confirming the Phase-1 diagnosis that the v1 kernel was
  scheduling-bound, not math-bound. The residency sub-criterion was
  deliberately not chased: every ≥16-warp/2-CTA configuration we could
  construct measured slower (occ variants 1.83–3.32 ms; N=64 2-CTA needs
  per-iteration layout conversions), i.e. 12-warp warp-specialization with
  softmax/WGMMA pingpong is the empirically optimal structure here, and the
  gate's intent (kill the no-eligible-warp starvation) is what the passing
  metrics measure. E2E banks −15.9 ms/chunk vs the −22.1 ms one would get by
  naively scaling the single-shape ncu delta (737 µs × 30): that −22.1 was an
  extrapolation, not a target — the 30 attention calls/chunk span DiT blocks
  at different densities and warm-cache states, so the measured −15.9 ms is
  the truth and the gap is extrapolation slack, NOT power throttling.
  Power/clock was directly measured (nvidia-smi dmon, OFF vs ON busy window):
  the GPU is a 900 W-capable Hopper enforced to 700 W (the 1000 W Grace+Hopper
  module budget minus ~300 W reserved for Grace/LPDDR5X), but BOTH backends
  peak at only ~660 W (v1 666 W / v2 660 W) — never touching the 700 W cap —
  and BOTH sag identically to ~1650–1800 MHz under the 1980 MHz lock (a
  heavy-tensor DVFS operating point, not a power or thermal cap: temps 52–56
  °C, power under limit). Since the sag is backend-independent it cannot be
  the OFF-vs-ON differential; v2 does the SAME work at the SAME ~660 W in less
  time = pure efficiency. (An earlier draft of this entry wrongly attributed
  the gap to an H8 power-cap clawback; corrected here against the dmon data.)
  @768 lands at 45.5 FPS (+9.1%); the roadmap's 49–51 was itself built on the
  same optimistic 0.9 ms/call extrapolation. Follow-ups: (1) re-measure `FLASHVSR_DECODER_OVERLAP`
  co-execution on top of triton2 (v2 still occupies 229 KB smem/SM, so the
  34% co-execution ceiling probably persists — measure, don't assume);
  (2) long_sb 2.00/wait 1.32 in v2 are now the top stalls (TMA-latency
  bound at the ring head) — NBUF=4 needs BLOCK_N=64 smem or Q-in-regs to
  fit, which is Phase-4-adjacent tuning; (3) FP8 attention (Phase 4) now
  has a WS substrate to build on.

### Entry template (copy-paste per attempt)

```markdown
#### <YYYY-MM-DD HH:MM> · <Phase 2A-1> · <Optimization name>

- Commit / patch:
- Files changed:
- Flag: FLASHVSR_<...>  (default OFF)
- Env vars used (full set):
- Exact benchmark command:
- Resolution / frames: 768x1408 / F=81   (spot-check: 1536x2560 y/n)
- Warmup / steady settings: warmup=1, steady=chunks 2..6 (from [chunks] line)
- FPS before → after (Δ):            (3-run medians)
- Steady chunk before → after (Δ):
- Peak mem before → after:
- Correctness: (max|diff| == 0 | PSNR = XX.X dB | mask-equality | n/a)
- Output difference (if applicable):
- Nsight report path (if generated):
- Decision: keep-enabled | keep-behind-flag | revert | investigate | postpone
- Interpretation (2–3 sentences, mandatory): did it match the roadmap
  estimate? why / why not? what follows?
```

---

## Phase 2A closure summary (2026-07-08)

- All of 2A-1..2A-5 landed with flag + log entry + interpretation +
  correctness result; 2A-6 evaluated and postponed via its go/no-go gate.
- Kept enabled (recommended set): `FUSE_ROPE`, `KV_RINGBUF`,
  `ATTN_STRIDED_IO`, `MASKGEN_LEAN`, `LQPROJ_LEAN`. Kept behind flag:
  `CACHE_ROPE_FREQS` (FPS-neutral @768, lossless, graph-capture prereq).
  Rejected during development: LQPROJ sub-item (b) (non-contiguous GEMM
  output → 49.9 dB, fails the lossless gate).
- Combined-stack parity: `test_phase2a_lossless.py ALL` → max|diff| == 0 vs
  all-flags-OFF (no flag interactions).
- Result @768x1408: 38.585 → 41.580 FPS (+7.8%), steady chunk 156.28 →
  138.46 ms (−17.8 ms); roadmap ceiling for 2A was ~20–26 ms — the gap is the
  traced-CPU rope_freqs share (rides under GPU work untraced) and the
  deliberately-skipped deep fusions (2B-3 mask topk, 2B-4 im2col).
- @1536x2560 spot-check recorded below; peak-mem cost of the arena documented.
- Recommended next (Phase 2B): decoder overlap (2B-1) first — decode is still
  a fully serialized 17–23% of E2E; then FP8 GEMM infra (2B-2, default-OFF),
  fused mask top-k (2B-3), fused im2col (2B-4, still eligible since 2A-5
  recovered <2 ms).

## Cumulative stack

<!-- Update whenever a flag joins/leaves the recommended set.
     "Delta vs Phase-2 Baseline" compares against the Step-0 median. -->

| Step | Enabled Optimizations | FPS | Steady Chunk Time | Peak Memory | Delta vs Phase-2 Baseline | Notes |
|------|----------------------|-----|-------------------|-------------|---------------------------|-------|
| 0 | full-knobs baseline (gemm+NHWC+fuse_norm+triton+TMA+caches) | 38.585 | 156.28 ms | 12.6 GiB | — | Step-0 fresh baseline (2026-07-08, df94d94) |
| 1 | baseline + FUSE_ROPE | 39.023 | 153.62 ms | 12.6 GiB | +1.14% FPS / −2.66 ms | 2A-1b, lossless |
| 2 | + KV_RINGBUF | 39.429 | 150.76 ms | 15.5 GiB | +2.19% FPS / −5.52 ms | 2A-2, lossless; +2.9 GiB arena slack |
| 3 | + ATTN_STRIDED_IO | 41.099 | 141.14 ms | 15.5 GiB | +6.52% FPS / −15.14 ms | 2A-3, lossless |
| 4 | + MASKGEN_LEAN | 41.477 | 139.00 ms | 15.5 GiB | +7.50% FPS / −17.28 ms | 2A-4, exact mask |
| 5 | + LQPROJ_LEAN | 41.580 | 138.46 ms | 15.62 GiB | +7.76% FPS / −17.82 ms | 2A-5, lossless |
| 6 | + DECODER_OVERLAP (kept behind flag, not in default set) | 42.373 | 190.58 ms (denoise+decode) | 15.16 GiB | +9.82% FPS vs Step 0 / +1.71% vs Step 5 | 2B-1, lossless; steady-chunk metric absorbs decode when ON — later per-chunk benchmarking stays flag-OFF; decode tail 561→125 ms |
| 7 | 2A set + ATTN_BACKEND=**triton2** (kept behind flag pending confirmation entry) | 45.466 | 122.06 ms | 15.62 GiB | +17.83% FPS vs Step 0 / +9.05% vs the 2A set | Phase 3 attention v2; PSNR 50.03 dB vs sparse (same class as `triton`'s ~49.7); composes with all 2A flags |

---

## Phase closure spot-checks (@1536x2560)

| Date | Phase closed | Enabled set | FPS @1536 | Steady chunk @1536 | Peak mem @1536 | Notes |
|------|--------------|-------------|-----------|--------------------|----------------|-------|
| 2026-07-08 | 2A | full-knobs + FUSE_ROPE + KV_RINGBUF + ATTN_STRIDED_IO + MASKGEN_LEAN + LQPROJ_LEAN | 11.489 | 501.92 ms | 48.39 GiB | Phase-1 ref: 11.01 / 531.5 ms / 37.4 GiB → +4.35% FPS, −29.6 ms; +11 GiB = arena spare slots at this res (`FLASHVSR_KV_RINGBUF_SPARE` trades it back). Gain smaller than @768 (+7.8%) as attention/decode share grows with res — consistent with ANALYSIS §0. |
| 2026-07-08 | 3 | 2A set + ATTN_BACKEND=triton2 (OFF ref same session: 11.472 / 503.22 ms / 48.39 GiB) | 12.436 | 449.24 ms | 48.39 GiB | Phase-3 attention v2 spot-check: +8.4% FPS, −53.98 ms/chunk, peak unchanged — the kernel win holds at scale (attention share scale-invariant, ANALYSIS §0). |
