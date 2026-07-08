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
| Date/time | _(fill)_ |
| Commit | _(fill)_ |
| GPU clocks locked | _(y/n, `nvidia-smi -lgc 1980,1980`)_ |
| Resolution / frames | 768x1408 / F=81 |
| Run 1 / Run 2 / Run 3 FPS | _(fill)_ / _(fill)_ / _(fill)_ |
| **Median FPS (= Phase-2 baseline)** | _(fill)_ |
| Median steady chunk ms | _(fill)_ |
| Peak memory GiB | _(fill)_ |
| Reference (Phase-1 campaign, single run) | 38.55 FPS · 156.24 ms · 12.6 GiB |
| Notes | _(dmon anomalies, background processes, etc.)_ |

---

## Per-change entries

<!-- Append one table row + one entry block per optimization attempt.
     Newest at the bottom. Do not delete failed attempts. -->

| Date | Phase | Optimization | Flag | FPS Before | FPS After | Delta | Steady Chunk Before | Steady Chunk After | Peak Mem Before | Peak Mem After | Correctness | Decision |
|------|-------|--------------|------|------------|-----------|-------|---------------------|--------------------|-----------------|----------------|-------------|----------|
|      |       |              |      |            |           |       |                     |                    |                 |                |             |          |

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
| 0 | full-knobs baseline (gemm+NHWC+fuse_norm+triton+TMA+caches) | _(fill)_ | _(fill)_ | _(fill)_ | — | Step-0 fresh baseline |
|   |                      |     |                   |             |                           |       |

---

## Phase closure spot-checks (@1536x2560)

| Date | Phase closed | Enabled set | FPS @1536 | Steady chunk @1536 | Peak mem @1536 | Notes |
|------|--------------|-------------|-----------|--------------------|----------------|-------|
|      |              |             |           |                    |                |       |
