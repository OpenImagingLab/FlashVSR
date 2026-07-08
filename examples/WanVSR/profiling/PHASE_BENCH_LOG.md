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
| 2026-07-08 | 2B-1 | Decoder overlap on side CUDA stream | `FLASHVSR_DECODER_OVERLAP` | 41.708 | 42.375 | +1.60% | 138.02 ms | 190.94 ms † | 15.62 GiB | 15.16 GiB | max\|diff\|=0 (768×{1,8}-chunk clips + 1536 1-chunk, incl. 3× ON race-repeat) | keep-behind-flag |

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

#### 2026-07-08 · Phase 2B-1 · Decoder overlap on a side CUDA stream

- Commit / patch: phase2b decoder overlap (this entry)
- Files changed: `diffsynth/pipelines/flashvsr_tiny.py` (side-stream decode
  path, event-gated, guarded by `use_decoder_overlap`), `profiling/run_pipe_target.py`
  (`[tail]` post-loop timing print — additive, no behaviour change), new
  `profiling/test_decoder_overlap_lossless.py`
- Flag: `FLASHVSR_DECODER_OVERLAP` (default OFF; serialized end-of-loop
  decode is byte-for-byte unchanged when OFF — verified via
  `test_phase2a_lossless.py ALL` still passing at this commit)
- Env vars used (full set): full-knob baseline + kept 2A set
  (`FUSE_ROPE=1 KV_RINGBUF=1 ATTN_STRIDED_IO=1 MASKGEN_LEAN=1 LQPROJ_LEAN=1`)
  + `FLASHVSR_DECODER_OVERLAP=0|1` under test — i.e. the **final Phase 2A
  cumulative configuration**, per this task's baseline requirement, not the
  bare full-knobs Step-0 baseline
- Exact benchmark command: §0.4 primary + kept set + `FLASHVSR_DECODER_OVERLAP=0|1`
- Resolution / frames: 768x1408 / F=81 (8 chunks); spot-check 1536x2560: y
- Warmup / steady settings: warmup=1, steady=chunks 2..6; **new** `[tail]`
  metric = wall time from the last denoise-chunk yield to `pipe()` return
  (decode + color_fix, whichever path is active)
- FPS before → after (Δ), 768, median of 3: 41.708 → 42.375 ms (+1.60%)
- Steady chunk before → after (Δ), 768: 138.02 → 190.94 ms († see below —
  this is NOT a regression; see interpretation)
- Decode tail before → after (Δ), 768: 559.4 → 125.7 ms (−433.7 ms, −77.5%)
- Peak mem before → after, 768: 15.62 → 15.16 GiB (**decrease**, −0.46 GiB)
- FPS before → after (Δ), 1536 (avg of 2 runs each): 11.4815 → 11.6315
  (+1.31%); steady chunk 501.26 → 686.57 ms; decode tail 2051.3 → 473.45 ms
  (−1577.85 ms, −76.9%); peak mem 48.39 → 46.72 GiB (**decrease**, −1.67 GiB)
- Correctness: `profiling/test_decoder_overlap_lossless.py` — OFF vs ON and
  ON vs ON (×3 repeat, race-condition check) at @768 for both a 1-chunk
  clip (F=25, exercises the degenerate "nothing to overlap with" case) and
  the 8-chunk profiling-default clip (F=81), PLUS a @1536x2560 1-chunk
  check: **every comparison `max|diff| == 0`, `mean|diff| == 0`**, frame
  count / shape / dtype (`bfloat16`) / device (`cuda:0`) identical, and
  explicit per-frame max|diff| == 0 (output ordering proof). Combined-stack
  regression check `test_phase2a_lossless.py ALL` still `max|diff| == 0`
  (flag-OFF path is untouched).
- Nsight report path: `profiling/reports/phase2b_decoder_overlap_on_768/`
  (ON) and `profiling/reports/phase2b_decoder_overlap_off_768/` (OFF,
  control), both `profile.nsys-rep` + exported `profile.sqlite`
  (`FLASHVSR_NVTX=1 FLASHVSR_PROF_STEADY=0:-1`, minimal `cuda,nvtx` trace,
  whole-run capture per this task's guidance — traced numbers are
  attribution-only, not headline FPS, per roadmap §0.5 rule 1)
- Nsight findings (sqlite queried directly, `CUPTI_ACTIVITY_KIND_KERNEL` /
  `NVTX_EVENTS` / `CUPTI_ACTIVITY_KIND_SYNCHRONIZATION`):
  - OFF trace: **one** CUDA stream carries all kernels (streamId 7, 15308
    kernels). ON trace: **two** streams — streamId 7 (main, 11490 kernels,
    1466.5 ms busy) and streamId 13 (the new decode side stream, 3860
    kernels, 500.1 ms busy). Confirms the side stream exists and does real
    work only when the flag is ON.
  - Per-chunk NVTX-window kernel-busy attribution (ON trace) proves genuine
    concurrent GPU execution, not just concurrent issue: e.g. `chunk1`
    window = 123.70 ms wall, but contains 84.42 ms of main-stream busy
    **and** 67.02 ms of decode-stream busy (sum 151.44 ms > 123.70 ms
    window ⇒ ≥27.7 ms is provably running on both streams at once).
    Steady chunks 3–7 each show ~148–154 ms main-busy + ~50–51 ms
    decode-busy (386 decode kernels/chunk) inside ~182–189 ms windows ⇒
    ~15–18 ms of each chunk's ~50 ms decode cost is genuinely hidden by
    concurrency (~30–36%, roughly a third); the remaining ~64–70% still
    lands on the wall clock.
  - No new hidden synchronization: the only sync records at the
    `decode_enqueue{i}` / `decode_wait` call sites are
    `CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_STREAM_WAIT_EVENT` (non-blocking
    GPU-side queue waits, 0.3–3.5 µs each — exactly `wait_event(...)`,
    never a blocking sync). A pre-existing `STREAM_SYNCHRONIZE` (~3–5 µs,
    once/chunk) shows up **identically** in the OFF-mode control trace —
    confirmed unrelated to this change (lives elsewhere in the already-landed
    2A stack; out of scope here). The only `CONTEXT_SYNCHRONIZE` in either
    trace is the benchmark harness's own post-`pipe()` call — the expected,
    allowed "benchmark boundary" sync, present in both modes.
- Decision: **keep-behind-flag** (default OFF in source, matching every
  other Phase 2A/2B flag; opt in with `FLASHVSR_DECODER_OVERLAP=1`)
- Interpretation: Decode genuinely overlaps denoise — confirmed at the
  kernel level (two real streams, measurable concurrent busy time, no
  hidden sync) — and the serialized decode **tail** collapses by ~77% at
  both resolutions (768: −433.7 ms; 1536: −1577.85 ms), which is the exact
  mechanism the roadmap targeted. However, the resulting E2E FPS gain
  (+1.60% @768, +1.31% @1536) is far below the roadmap's optimistic
  +21–29% ceiling, because only ~30% of decode's own GPU kernel time is
  truly free-riding on denoise's spare SM capacity in steady state — the
  other ~70% of what used to be tail time is simply *relocated* into the
  per-chunk loop (hence steady-chunk-time rising, as this task's own
  instructions warned it might) rather than hidden. The likely reason:
  TCDecoder is an eager, per-timestep, many-small-kernel Python loop
  (386 kernels for a 2-timestep chunk decode); at that granularity its CPU
  dispatch cost is comparable to its own GPU execution cost, so part of its
  cost is CPU-dispatch-bound rather than purely GPU-idle-time-bound, which
  the ceiling estimate did not account for. An unexpected, genuine bonus:
  peak memory *decreased* at both resolutions (768 −2.9%, 1536 −3.45%),
  because decoding 2–6 timesteps/chunk bounds TAEHV's internal
  `apply_model_with_memblocks` work-queue depth lower than one 20+-timestep
  one-shot call — this is not a workaround, it falls out of the existing
  streaming-decoder design once invoked more granularly. Correctness is
  bit-identical at both resolutions, for both a 1-chunk and an 8-chunk
  clip, and stable across 3 repeated ON runs (no race condition detected);
  output ordering, shape, dtype, device, and frame count are all unchanged.
  Given the gain, while real, reproducible, and complication-free from a
  correctness/memory standpoint, is modest relative to the genuine added
  stream/event/lifetime complexity, this lands as keep-behind-flag rather
  than joining the always-on recommended set outright — a strong candidate
  for promotion once it has seen more soak testing (varied clip lengths and
  content beyond this harness's synthetic clips).

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

## Phase 2B-1 — Decoder overlap on a side CUDA stream (2026-07-08)

Scope: overlap TCDecoder per-chunk decode with the denoise/chunk loop via a
side CUDA stream + explicit event synchronization, gated by
`FLASHVSR_DECODER_OVERLAP` (default OFF). Baseline = the final Phase 2A
cumulative stack (Step 5 of the cumulative table below: 41.580 FPS /
138.46 ms / 15.62 GiB @768), freshly re-measured at the start of this task
(see decoder table, "baseline (fresh)" row) before any code changed.
Full detail entry, correctness methodology, and Nsight findings are in the
per-change entry above; this section is the required decoder-specific
summary table + closure interpretation.

| Config | E2E FPS | Steady Chunk Time | Decode Tail Time | Peak Memory | Notes |
|--------|---------|--------------------|-------------------|-------------|-------|
| 768x1408 baseline (fresh, pre-change, 3-run median) | 41.699 | 137.58 ms | n/a (no `[tail]` metric yet) | 15.62 GiB | reproduces Phase 2A final log (41.580/138.46/15.62) within noise |
| 768x1408 OFF (post-change, 3-run median) | 41.708 | 138.02 ms | 559.4 ms | 15.62 GiB | flag-OFF path confirmed neutral vs fresh baseline |
| 768x1408 ON (3-run median) | 42.375 | 190.94 ms † | 125.7 ms | 15.16 GiB | † includes inline decode-enqueue, not comparable to denoise-only steady-chunk numbers elsewhere in this log |
| 768x1408 Δ (ON − OFF) | +1.60% | +52.92 ms † | −433.7 ms (−77.5%) | −0.46 GiB | tail collapse is the real story; steady-chunk rise is expected relocation, not regression |
| 1536x2560 OFF (avg of 2 runs) | 11.4815 | 501.26 ms | 2051.3 ms | 48.39 GiB | matches Phase 2A spot-check ref (11.489/501.9/48.39) |
| 1536x2560 ON (avg of 2 runs) | 11.6315 | 686.57 ms † | 473.45 ms | 46.72 GiB | same qualitative pattern as @768, slightly smaller % FPS gain |
| 1536x2560 Δ (ON − OFF) | +1.31% | +185.31 ms † | −1577.85 ms (−76.9%) | −1.67 GiB | decode's larger absolute share at 1536 shows up as a larger absolute tail cut, not a larger FPS %  |

Written interpretation (required questions):

- **Did decode actually overlap with denoise?** Yes — verified at the CUDA
  stream/kernel level via Nsight (not inferred from wall time alone): the
  ON trace shows two distinct active streams (main + a new decode stream,
  the OFF trace has only one), and decode-stream kernels measurably execute
  *during* later chunks' NVTX windows, with per-chunk window duration less
  than the sum of main-stream-busy + decode-stream-busy time inside that
  window — that inequality is only possible if the two streams' kernels
  ran concurrently on the GPU.
- **How much of the decode tail was hidden?** The serialized tail shrank
  ~77% at both resolutions (768: 559→126 ms; 1536: 2051→473 ms). But
  kernel-level attribution shows only ~30–36% of decode's own GPU busy time
  in steady state is truly concurrent with denoise (~15–18 ms hidden out of
  ~50 ms decode cost per chunk @768) — most of the tail's disappearance is
  because that work moved earlier (into the loop), not because it became
  free. The residual, non-hidden portion is why E2E FPS gain (+1.3–1.6%) is
  much smaller than the tail-collapse percentage.
- **Did steady chunk time change?** Yes, it rose (768: 138→191 ms; 1536:
  501→687 ms). This is expected, not a regression: per this task's own
  instructions, that metric now measures denoise **plus** inline
  decode-enqueue wall time, and must not be used alone to judge the
  optimization (see the tail and E2E FPS rows instead).
- **Did E2E FPS improve?** Yes, modestly and reproducibly: +1.60% @768
  (3/3 runs improved), +1.31% @1536 (2/2 runs improved).
- **Did peak memory increase?** No — it *decreased* at both resolutions
  (768 −2.9%, 1536 −3.45%). Per-chunk decode (2–6 timesteps) bounds
  TAEHV's `apply_model_with_memblocks` work-queue depth lower than one
  single 20+-timestep one-shot call; this falls out of invoking the
  existing streaming decoder more granularly, not a new allocation
  strategy.
- **Is the memory change acceptable?** Trivially yes — there is no
  increase to justify.
- **Is the implementation safe enough to enable by default?** Mechanically,
  yes: bit-identical output (`max|diff| == 0`) at both resolutions, for a
  1-chunk edge case and the 8-chunk default clip, stable across 3 repeated
  ON runs (no race condition), explicit ordering/shape/dtype/device checks
  all pass, and Nsight confirms no new hidden synchronization was
  introduced (the one pre-existing per-chunk sync is identical in the OFF
  control trace). The flag-OFF path is verified byte-for-byte unchanged
  (`test_phase2a_lossless.py ALL` still `max|diff| == 0`).
- **Should the flag remain default-off or default-on?** Default-OFF in
  source, consistent with every other Phase 2A/2B flag in this repo (opt-in
  via `FLASHVSR_DECODER_OVERLAP=1`). Given the gain is real but modest
  relative to the genuine stream/event/tensor-lifetime complexity added,
  this is logged as **keep-behind-flag** rather than joining the always-on
  recommended set — a good promotion candidate once it has run against more
  varied clip lengths/content than this harness's synthetic clips.

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
| 6 | + DECODER_OVERLAP (behind flag, not in default recommended set) | 42.375 | 190.94 ms † | 15.16 GiB | +9.82% FPS vs Step-0 | 2B-1, bit-identical; primary comparison is same-session same-commit fresh OFF re-measurement (41.708→42.375, **+1.60%**, this task's methodologically correct Δ) — vs the older Step-5 log entry (41.580, different session) it's +1.91%, a looser cross-check only; † steady-chunk no longer denoise-only (see Phase 2B-1 section); decode tail −77.5% (559→126 ms); peak mem **−0.46 GiB** vs Step 5 |

---

## Phase closure spot-checks (@1536x2560)

| Date | Phase closed | Enabled set | FPS @1536 | Steady chunk @1536 | Peak mem @1536 | Notes |
|------|--------------|-------------|-----------|--------------------|----------------|-------|
| 2026-07-08 | 2A | full-knobs + FUSE_ROPE + KV_RINGBUF + ATTN_STRIDED_IO + MASKGEN_LEAN + LQPROJ_LEAN | 11.489 | 501.92 ms | 48.39 GiB | Phase-1 ref: 11.01 / 531.5 ms / 37.4 GiB → +4.35% FPS, −29.6 ms; +11 GiB = arena spare slots at this res (`FLASHVSR_KV_RINGBUF_SPARE` trades it back). Gain smaller than @768 (+7.8%) as attention/decode share grows with res — consistent with ANALYSIS §0. |
| 2026-07-08 | 2B-1 (spot-check, flag behind-flag) | 2A set + `FLASHVSR_DECODER_OVERLAP=1` | 11.6315 (avg of 2; OFF control avg 11.4815) | 686.57 ms † | 46.72 GiB | +1.31% FPS vs the OFF control at this same res (not vs the 2A row above, which predates this task's code); decode tail 2051.3→473.45 ms (−76.9%, larger absolute cut than @768 since decode's share grows with resolution); peak mem **decreased** −1.67 GiB; bit-identical (1-chunk clip, OFF vs ON×3). † steady-chunk includes inline decode-enqueue, see Phase 2B-1 section. |
