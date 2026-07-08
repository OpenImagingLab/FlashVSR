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

## Cumulative stack

<!-- Update whenever a flag joins/leaves the recommended set.
     "Delta vs Phase-2 Baseline" compares against the Step-0 median. -->

| Step | Enabled Optimizations | FPS | Steady Chunk Time | Peak Memory | Delta vs Phase-2 Baseline | Notes |
|------|----------------------|-----|-------------------|-------------|---------------------------|-------|
| 0 | full-knobs baseline (gemm+NHWC+fuse_norm+triton+TMA+caches) | 38.585 | 156.28 ms | 12.6 GiB | — | Step-0 fresh baseline (2026-07-08, df94d94) |
| 1 | baseline + FUSE_ROPE | 39.023 | 153.62 ms | 12.6 GiB | +1.14% FPS / −2.66 ms | 2A-1b, lossless |
| 2 | + KV_RINGBUF | 39.429 | 150.76 ms | 15.5 GiB | +2.19% FPS / −5.52 ms | 2A-2, lossless; +2.9 GiB arena slack |

---

## Phase closure spot-checks (@1536x2560)

| Date | Phase closed | Enabled set | FPS @1536 | Steady chunk @1536 | Peak mem @1536 | Notes |
|------|--------------|-------------|-----------|--------------------|----------------|-------|
|      |              |             |           |                    |                |       |
