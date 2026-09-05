# FlashVSR Phase-2+ Optimization Roadmap

Derived from the Phase-1 profiling campaign in [`ANALYSIS.md`](./ANALYSIS.md)
(nsys + ncu + ceiling microbenches, GH200 / sm_90, 2026-07-08).

This document converts the profiling findings into a staged engineering plan.
**No optimization in this document is implemented yet.** All gain figures are
*estimates derived from profiling attribution and microbenchmarks* — they are
ceilings, not promises, and every one of them must be re-validated by a fresh
benchmark entry in [`PHASE_BENCH_LOG.md`](./PHASE_BENCH_LOG.md) when the
optimization lands.

---

## 0. Baseline and Rules

### 0.1 Phase-2 baseline (from the Phase-1 campaign, untraced single runs)

| Metric | Value | Source |
|---|---|---|
| E2E FPS @768x1408, F=81 | **38.55** | `run_pipe_target.py`, all knobs ON |
| Steady chunk time @768 | **156.24 ms** (8 new frames/chunk) | `[chunks]` line, chunks 2–6 avg |
| Peak GPU memory @768 | **12.6 GiB** | `torch.cuda.max_memory_allocated` |
| E2E FPS @1024x1920 / @1536x2560 | 21.77 / 11.01 | resolution shift runs |
| Steady chunk @1024 / @1536 | 274.1 ms / 531.5 ms | |
| GPU idle within steady chunk | 3.1% @768, ~0% @1024+ | not launch-bound |
| Decode share of E2E | 17–23% (serialized tail) | all resolutions |

Before the first Phase-2A change lands, this baseline MUST be re-measured
fresh (3 runs, median) and recorded as **Step 0** in `PHASE_BENCH_LOG.md`.
All subsequent deltas are computed against that Step-0 entry, not against the
numbers above.

### 0.2 Environment

- NVIDIA GH200 480GB (sm_90, 132 SM, 96 GB HBM3), driver 595.58.03
- CUDA 13.2 · torch 2.12.0a0 (NGC 26.05) · triton 3.7.0 · cuDNN 9.22
- Python: `/root/FlashVSR/venv/bin/python` (system python lacks imageio)
- GPU clocks locked: `nvidia-smi -lgc 1980,1980` (release: `nvidia-smi -rgc`)
- **Power constraint:** platform enforces a 700 W cap (900 W requested and
  denied). Under sustained load the GPU draws 649–687 W and DVFS sags clocks
  to 1635–1815 MHz (84–92% of max). Consequence: *removing work* compounds
  (saves power → higher sustained clocks); *adding raw math throughput*
  partially self-defeats. This is why the roadmap is efficiency-first.

### 0.3 Baseline environment variables (the "full knobs" config)

```bash
FLASHVSR_CONV3D_BACKEND=gemm
FLASHVSR_TCDECODER_CHANNELS_LAST=1
FLASHVSR_FUSE_NORM=1
FLASHVSR_ATTN_BACKEND=triton        # implies FLASHVSR_ATTN_TMA=1 (default)
FLASHVSR_CACHE_MOD=1
FLASHVSR_CACHE_MASK_BIAS=1
```

### 0.4 Benchmark command template (PRIMARY, untraced)

Run from `examples/WanVSR/`:

```bash
FLASHVSR_CONV3D_BACKEND=gemm \
FLASHVSR_TCDECODER_CHANNELS_LAST=1 \
FLASHVSR_FUSE_NORM=1 \
FLASHVSR_ATTN_BACKEND=triton \
FLASHVSR_CACHE_MOD=1 \
FLASHVSR_CACHE_MASK_BIAS=1 \
FLASHVSR_PROF_STEADY=off \
/root/FlashVSR/venv/bin/python profiling/run_pipe_target.py
```

Resolution / length variants via `FLASHVSR_PROF_W`, `FLASHVSR_PROF_H`
(multiples of 128), `FLASHVSR_PROF_FRAMES` (8n+1; default 85 → F=81,
8 chunks). The script prints:

- `[result] {... fps, wall_s, peak_gib, steady_chunk_ms ...}` — the numbers
  that go into the bench log,
- `[chunks] per-chunk ms: [...]` — steady chunk = mean of chunks 2..6.

### 0.5 Mandatory discipline (non-negotiable rules)

1. **Untraced runs are the only source of truth for FPS / chunk time / peak
   memory.** nsys/ncu runs distort wall time (stop-flush lands inside the
   timed window; CPU-side tracing overhead inflates gaps).
2. **Nsight Systems / Nsight Compute are used exclusively for attribution**
   (where did the time go, why is a kernel slow) — never for headline numbers.
3. **One optimization at a time.** Every optimization is benchmarked
   independently (flag ON vs flag OFF on the same commit) before the next one
   is started.
4. **Every optimization is behind a `FLASHVSR_*` environment flag, default
   OFF**, or (if a flag is genuinely impossible) trivially revertible in a
   single commit. This preserves the PR philosophy: default output is
   bit-for-bit the current behaviour.
5. **No performance claim without a fresh `PHASE_BENCH_LOG.md` entry.**
   Numbers quoted from memory, from ANALYSIS.md, or from a stale run do not
   count.
6. **Do not proceed to the next optimization until the current one has a log
   entry AND a short written interpretation** (2–3 sentences: did it match the
   estimate, why/why not, decision).
7. Every change lands with a correctness check appropriate to its class:
   *lossless* claims require `max|diff| == 0` (pattern:
   `test_cache_lossless.py`); *numerically-neutral* claims require PSNR vs
   flag-OFF ≥ 49 dB on the standard clip (pattern: `test_fuse_norm.py`).
8. @768x1408 is mandatory for every change. @1536x2560 spot-check is required
   only at phase closure (bottleneck structure is scale-invariant per
   ANALYSIS §0, so per-change high-res runs are wasted GPU time).

---

## 1. Phase 2A — Low-effort / high-confidence cleanup

Goal: bank the safe wins first. Everything here is (a) small surface area,
(b) exactly-preserving or trivially PSNR-gated, (c) independently flag-gated.
Estimated combined ceiling from ANALYSIS: **~20–26 ms/chunk**
(156 → ~130–136 ms steady, i.e. roughly +12–17% denoise throughput) — to be
verified item by item.

**Explicitly excluded from 2A:** attention kernel rewrite (Phase 3), FP8
anything (Phase 2B infra + Phase 4 gate), decoder overlap (Phase 2B),
any change to the sparse mask semantics (Phase 4).

### 2A-1 RoPE frequency caching + RoPE apply cleanup

| | |
|---|---|
| Profiling says | `rope` phase = 10.1 ms/chunk GPU (6.8%); additionally `rope_freqs` construction showed ~44 ms/chunk of CPU-side wall in traced runs and is step-invariant (depends only on `(cur_process_idx, f, h, w)`) |
| Why low effort | freqs caching is a dict keyed on `(idx, f, h, w)` around an existing pure function; apply-fusion reuses the existing `_maybe_compile` pattern from FUSE_NORM |
| Estimated gain | −8 ms/chunk GPU (estimate, unverified) + CPU-side headroom that currently rides under GPU work |
| Confidence | high |
| Risk | low (freqs cache is bit-identical; fused apply is elementwise-only) |
| Files/functions | `diffsynth/pipelines/flashvsr_tiny.py::model_fn_wan_video` (the `rope_freqs` block), `diffsynth/models/wan_video_dit.py::rope_apply` |
| Flags | `FLASHVSR_CACHE_ROPE_FREQS` (lossless), `FLASHVSR_FUSE_ROPE` (fusion) — two separate flags, two separate log entries |
| Benchmark | primary @768; log both flags individually and combined |
| Correctness | freqs cache: `max\|diff\|==0` vs OFF. Fused apply: PSNR ≥ 49 dB vs OFF |
| Default | OFF until log entry; lossless part is a candidate for default-ON later |

### 2A-2 KV cache ring buffer (remove `kv_cat`)

| | |
|---|---|
| Profiling says | `kv_cat` = 3.4 ms/chunk (2.3%), `CatArrayBatchedCopy` at 3235 GB/s = 81% HBM3 — at BW limit, so the only fix is not copying; sliding-window trim already drops the oldest window each chunk |
| Why low effort | replace `torch.cat([pre_cache_k, k_w])` + trim-slice with a preallocated `(kv_len+1)`-slot buffer and rolling writes; contained in one forward function + cache init |
| Estimated gain | −3 ms/chunk (estimate, unverified) |
| Confidence | high |
| Risk | low-medium (indexing/rotation bugs would corrupt temporal context — caught by bit-diff) |
| Files/functions | `diffsynth/models/wan_video_dit.py::SelfAttention.forward` (kv concat + `cache_trim`), `diffsynth/pipelines/flashvsr_tiny.py::__call__` (`pre_cache_k/v` lifecycle) |
| Flag | `FLASHVSR_KV_RINGBUF` |
| Benchmark | primary @768 |
| Correctness | `max\|diff\|==0` vs OFF over a full multi-chunk clip (exercises rotation) |
| Default | OFF until proven, then candidate for default-ON (lossless) |

### 2A-3 Attention-path transpose / contiguous cleanup

| | |
|---|---|
| Profiling says | `attn_core` − kernel = 11.6 ms/chunk (7.7% of GPU): three `.contiguous()` transposes `(S,n,d)→(n,S,d)` for q/k/v plus the output transpose back, all introduced by the triton backend glue; ncu shows them as SM-bound strided copies at only 980 GB/s |
| Why low-medium effort | the Triton kernel's addressing already goes through strides; passing real (non-contiguous) strides for q/k/v/out removes the copies without touching kernel math. Glue-side change + kernel signature audit |
| Estimated gain | −11.6 ms/chunk ceiling (estimate, unverified; some reorder cost may remain) |
| Confidence | high |
| Risk | medium (wrong stride handling = garbage attention — caught immediately by cos/PSNR checks) |
| Files/functions | `diffsynth/models/wan_video_dit.py::flash_attention` (triton branch glue), `diffsynth/models/triton_block_sparse_attn.py` (`triton_block_sparse_attention` wrapper + `_bsfa*_kernel` stride params; note TMA descriptors constrain layouts — if TMA requires contiguity, fix the TMA=0 path first and measure) |
| Flag | `FLASHVSR_ATTN_STRIDED_IO` |
| Benchmark | primary @768; also compare `FLASHVSR_ATTN_TMA=0/1` interaction |
| Correctness | kernel-level cosine ≥ 0.9999 vs current triton path; E2E PSNR ≥ 49 dB vs sparse backend (same gate the PR used) |
| Default | OFF until log entry |

### 2A-4 mask_gen allocation / sync cleanup (NOT the fused kernel)

| | |
|---|---|
| Profiling says | `mask_gen` = 7.4 ms/chunk (4.9%) through a chain of tiny kernels; separately, the *sparse* backend allocates `cu_seqlens_q/k` + `head_mask_type` via `torch.tensor(..., device=...)` **per call** — a hidden H2D sync that explains its 8.2% idle |
| Why low effort | buffer reuse and hoisting constant tensors out of the hot path; no algorithm change (the fused top-k kernel is Phase 2B-3) |
| Estimated gain | −1–2 ms/chunk @768 on the triton path (estimate); larger effect on the sparse fallback path (idle reduction) |
| Confidence | medium-high |
| Risk | low |
| Files/functions | `diffsynth/models/wan_video_dit.py::generate_draft_block_mask` (intermediate allocs), `flash_attention` sparse branch (persistent cu_seqlens/head_mask_type keyed on shape/device) |
| Flag | `FLASHVSR_MASKGEN_LEAN` |
| Benchmark | primary @768 (triton) + one sparse-backend run to capture the sync win |
| Correctness | `max\|diff\|==0` (allocation/hoisting only) |
| Default | OFF until log entry |

### 2A-5 LQ projector allocation / layout cleanup

| | |
|---|---|
| Profiling says | lq_conv1+conv2 = 11.7 ms/chunk (7.8%); path split: pad+cat 8%, im2col copy 29%, addmm 65%. Also per-clip `cache_x = x[..., -CACHE_T:].clone()` copies in `stream_forward` |
| Why low effort | this item is only the cheap part: avoid the separate pad+cat materialization (fold cache frames into the im2col source view), reuse im2col/patch buffers across clips, drop redundant clones. The *fused im2col-GEMM kernel* is Phase 2B-4 |
| Estimated gain | −1–2 ms/chunk (estimate, unverified) |
| Confidence | medium-high |
| Risk | low (streaming-cache semantics must be preserved — bit-diff catches drift across chunk boundaries) |
| Files/functions | `examples/WanVSR/utils/utils.py::CausalConv3d.forward`, `_conv3d_gemm` / `_im2col_gemm_rows`, `Causal_LQ4x_Proj.stream_forward` |
| Flag | `FLASHVSR_LQPROJ_LEAN` |
| Benchmark | primary @768 |
| Correctness | `max\|diff\|==0` vs OFF across a full clip (streaming cache exercised) |
| Default | OFF until log entry |

### 2A-6 (Optional experiment) CUDA Graphs on the steady chunk

| | |
|---|---|
| Profiling says | GPU idle within steady chunk = 3.1% @768 (≤4.8 ms/chunk ceiling), ~0% @1024+. This is the *smallest* item in 2A and may not pay for its complexity |
| Why gated as experiment | graph capture requires a sync-free, allocation-stable, shape-static chunk body. Today the chunk body is *probably not* capture-safe (mask topk sizes, cache rotation, python-side branching). Go/no-go gate: after 2A-1..2A-5 land, re-measure idle; only attempt capture if idle ≥ 2% AND a capture dry-run shows no illegal ops |
| Estimated gain | ≤ −4.8 ms/chunk @768, ~0 at higher resolutions (estimate) |
| Confidence | medium |
| Risk | medium (silent staleness bugs if any buffer is re-bound between replays) |
| Files/functions | `diffsynth/pipelines/flashvsr_tiny.py::__call__` steady-chunk body |
| Flag | `FLASHVSR_CUDA_GRAPHS` |
| Benchmark | primary @768 (where the idle exists); confirm no regression @1536 |
| Correctness | `max\|diff\|==0` vs OFF |
| Default | OFF; likely stays OFF unless the win is clean |

### Phase 2A exit criteria

- Each of 2A-1..2A-5 (2A-6 optional) has: flag, log entry, interpretation,
  correctness result.
- Cumulative log row updated; @1536 spot-check run recorded.
- Expected (to verify): steady chunk ~130–136 ms, E2E ≈ 43–46 FPS @768.

---

## 2. Phase 2B — Medium-effort structural wins

These need more design care than 2A cleanup: streams, new numerics paths, or
custom kernels — but all have strong profiling evidence.

### 2B-1 Decoder overlap on a side CUDA stream  ⟵ highest E2E leverage in 2B

| | |
|---|---|
| Expected impact | decode is 343–430 ms of a ~2.0 s run @768 (17–21% E2E) and 22–23% of (chunks+decode) at 1024/1536. Fully hidden ⇒ **+21–29% FPS E2E** (estimate, unverified) — likely the best gain/effort in the whole roadmap |
| Why not 2A | touches scheduling semantics, not just code: per-chunk streaming decode on a second stream, event-based handoff of `cur_latents`, allocator behaviour across streams, and TCDecoder's stateful mem-blocks (`TAEHV.mem`) must only ever be touched by the decode stream. The tiny pipeline currently decodes once at the end; the long pipeline (`flashvsr_tiny_long.py`) already decodes per chunk — use it as the semantic reference |
| Design sketch | after each chunk's latent update, `record_event`; decode stream waits on the event, decodes the chunk's latents with the matching LQ cond slice, appends to output. Denoise stream never waits on decode except at the very end |
| Correctness validation | output must be `max\|diff\|==0` vs sequential decode (same TCDecoder state transitions in the same order); explicit test with short AND long clips; race detection via 3 repeated runs (identical outputs) |
| Quality/numerical risk | none if bit-identical is enforced; the risk is concurrency bugs, not math |
| Benchmark plan | primary @768 AND @1536 (decode share is larger at high res); also record peak memory (decode buffers now coexist with denoise) |
| Rollback | `FLASHVSR_DECODE_OVERLAP=0` restores the end-of-loop decode verbatim |
| Files/functions | `diffsynth/pipelines/flashvsr_tiny.py::__call__` (chunk loop + decode call), `examples/WanVSR/utils/TCDecoder.py` (`decode_video`, `apply_model_with_memblocks`, `clean_mem`) |

### 2B-2 FP8 GEMM infrastructure (`torch._scaled_mm`)

| | |
|---|---|
| Expected impact | GEMMs ≈ 33 ms/chunk; microbench measured ×1.55–1.72 vs bf16 at exact shapes (qkv/o ×1.59, ffn1 ×1.55, ffn2 ×1.72, lq_linear ×1.55) ⇒ ~−13 ms/chunk ceiling (estimate). bf16 GEMMs are already at 82–85% SOL — FP8 is the only remaining GEMM lever |
| Why not 2A | new numerics path: weight/activation casting strategy, scale management, and a quality gate are required. **Implementation lands in 2B, but the enable decision is governed by the Phase-4 quality protocol** — this item ships default-OFF regardless of speed |
| Scope | qkv/o, ffn1/ffn2, LQ per-layer linears first (per-tensor scales, e4m3 weights pre-cast once); cross-attn q/o optional second step |
| Correctness validation | Phase-4 protocol: PSNR/SSIM vs flag-OFF on examples 0–3, per-layer activation error audit if PSNR < 49 dB |
| Quality/numerical risk | medium — distilled one-step model may be sensitive; unknown until measured |
| Benchmark plan | primary @768 with flag ON vs OFF; per-shape kernel check via one ncu run (confirm FP8 kernels actually selected) |
| Rollback | `FLASHVSR_FP8_GEMM=0` |
| Files/functions | `diffsynth/models/wan_video_dit.py` (Linear call sites: SelfAttention q/k/v/o, DiTBlock ffn, CrossAttention q/o), `examples/WanVSR/utils/utils.py::Causal_LQ4x_Proj.linear_layers` |

### 2B-3 Fused / semi-fused mask generation (top-k path)

| | |
|---|---|
| Expected impact | mask_gen = 7.4 ms/chunk (4.9%); the chain is `mean-pool → einsum → +bias → softmax → topk(gatherTopK @ SM 5.2%, 0.1 waves) → 7–9 radix mini-kernels → compare`. A single fused kernel (or per-head threshold-selection kernel) should land near ~1 ms/chunk ⇒ −5 ms (estimate) |
| Why not 2A | requires a custom Triton kernel (fused softmax+select) or a rewritten selection algorithm; more design/testing than buffer hygiene |
| Correctness validation | the produced boolean block mask must be **identical** to the reference implementation on real inputs (not just statistically similar) — direct tensor equality across a full clip; then E2E bit-diff |
| Quality/numerical risk | low if mask equality is enforced; any tie-breaking difference in top-k must be resolved to match torch semantics or shown to be E2E-neutral (then it moves to Phase 4) |
| Benchmark plan | primary @768; mask_gen shrinks relatively at higher res (2.1% @1536) so no high-res requirement |
| Rollback | `FLASHVSR_FUSED_MASKGEN=0` |
| Files/functions | `diffsynth/models/wan_video_dit.py::generate_draft_block_mask` (+ new kernel module) |

### 2B-4 Fused im2col-GEMM for the LQ projector (only if 2A-5 is not enough)

| | |
|---|---|
| Expected impact | im2col copy = 29% of the conv path ⇒ −4 ms/chunk ceiling (estimate). Gate: skip this item if 2A-5 already gets ≥2 ms and the projector drops below ~6% of GPU |
| Why not 2A | a real fused kernel (CUTLASS 3.x conv3d or Triton implicit-GEMM with the causal window) vs cuDNN is mandatory — cuDNN 9.22 direct conv3d measured **18× slower** (152 ms vs 8.4 ms) at this shape, so there is no library shortcut |
| Correctness validation | bit-diff vs the current gemm path (`test_conv3d_gemm_parity.py` pattern: single-call + streaming-cache) |
| Quality/numerical risk | low (same math, same accumulation dtype required — fp32 accumulate) |
| Benchmark plan | primary @768 + isolated kernel bench at conv1/conv2 shapes; @1536 spot-check (projector share grows slightly with res) |
| Rollback | `FLASHVSR_CONV3D_BACKEND=gemm` (existing path untouched); new path = `FLASHVSR_CONV3D_BACKEND=fused` |
| Files/functions | `examples/WanVSR/utils/utils.py::_conv3d_gemm` (+ new kernel) |

### Phase 2B exit criteria

- 2B-1 decode overlap proven bit-identical and logged @768 + @1536.
- 2B-2 FP8 infra merged default-OFF with a Phase-4 gate ticket.
- 2B-3 mask equality proven; 2B-4 done or explicitly skipped with rationale.
- Cumulative table updated with the 2A+2B stack.

---

## 3. Phase 3 — Major attention kernel work (`_bsfa` v2)

The single largest target, deliberately NOT first: highest effort, highest
risk, and its payoff stacks with (rather than blocks) everything above.

### 3.1 Evidence recap (why the kernel must be rebuilt, not tuned)

From ncu (`--set full`, steady chunk, @768):

```
_bsfa_tma_kernel: 2.03 ms · SM SOL 40.2% · tensor pipe 40.2% (active 46.4%)
DRAM 134 GB/s (3.3%) · L2 hit 92.5% · achieved WGMMA ≈ 393 TFLOP/s
occupancy 12.5% = 1 CTA/SM (178 reg/thread + 229 KB smem/block) · 8 warps/SM
stalls/issue: barrier 2.39 · wait 1.00 · short_sb 0.55  (total 6.37)
scheduler: 68.6% of cycles have NO eligible warp
TMA=0 variant: 2.20 ms · 235 reg · 164 KB smem · long_sb 0.87 (TMA removed it)
```

Reference points at the exact shape (q 8448 × kv 25344, h12, d128, density 0.606):

| Kernel | time | meaning |
|---|---|---|
| cuDNN dense SDPA (full attention) | 1.86 ms | computes 1.65× the FLOPs, still faster |
| ideal sparse = dense × 0.606 | **1.13 ms** | efficiency ceiling |
| `_bsfa_tma` (current) | 2.03 ms | **56% of ideal** |
| FlexAttention + BlockMask | 1.92 ms | torch-native is not the answer |

Interpretation: a single-CTA-per-SM Triton pipeline where all 8 warps
synchronize at every stage barrier; with nothing else resident, the tensor
pipe idles ~60% of the time. This is precisely the failure mode
warp-specialization (producer/consumer) and/or 2-CTA residency solve.

### 3.2 Candidate directions (in evaluation order)

1. **Triton warp-specialized rewrite** — keep the existing mask format and
   glue; restructure into producer (TMA loads) / consumer (WGMMA) warp groups,
   tune `num_stages`/tile sizes so smem ≤ ~114 KB or regs ≤ ~96 to admit a
   second CTA. Lowest integration cost; Triton 3.7 WS maturity is the risk.
2. **CUTLASS FMHA-based implementation** — start from CUTLASS 3.x Hopper FMHA
   (WS pingpong pipeline, proven ≥60–75% tensor util) and add 2D block-mask
   skipping. Highest ceiling, highest integration cost (C++ extension build).
3. **FA3-style hand-rolled pipeline ideas** applied to whichever base wins:
   pingpong warpgroups, softmax/WGMMA overlap, TMA stores.
4. **Exact sparse mask preservation is mandatory** in all directions — the
   trained sparse pattern is quality-bearing; the mask semantics of
   `block_sparse_attn` must be reproduced block-for-block.
5. (Later, optional) **FP8 attention** (QK^T and/or PV in e4m3) — belongs to
   Phase 4; est. additional −15–25 ms/chunk, quality unknown.

### 3.3 Targets, tests, fallback

| | |
|---|---|
| Expected gain | 2.03 → ~1.13–1.3 ms/call ⇒ **−21 to −26 ms/chunk @768** (estimate); relative share grows at higher res (attention ≈ 47% everywhere) |
| Development risk | high (kernel correctness across mask densities/shapes; Hopper pipeline subtleties; possible Triton compiler limitations) |
| Acceptance metrics (ncu, mandatory) | tensor pipe active ≥ 60% elapsed · barrier stall < 1.0/issue · ≥2 CTA/SM or WS with ≥16 warps/SM · duration ≤ 1.3 ms at the reference shape |
| Correctness tests | kernel-level cosine ≥ 0.9999 vs `block_sparse_attn` on randomized real-shape inputs (multiple densities incl. degenerate all-true/all-false rows); E2E PSNR ≥ 49 dB vs sparse backend; multi-chunk streaming bit-stability of the KV/cache interface |
| Fallback path | runtime chain `v2 → _bsfa_tma → block_sparse_attn` behind `FLASHVSR_ATTN_BACKEND=triton2` (sm_90-guarded, silent fallback on any error — same pattern as the existing backends) |
| Benchmark | primary @768 + @1536; ncu acceptance run archived in the log entry |
| Separate PR? | **Yes.** Self-contained, reviewable, revertible; roadmap phases 2A/2B must not wait on it |

---

## 4. Phase 4 — Quality-risk / precision-risk optimizations

Anything that can change output pixels beyond bit-identical or the
established ≥49 dB gate lives here, is **default-OFF permanently** until it
passes the protocol, and is documented with its measured quality delta.

### 4.1 Candidates

1. **FP8 GEMMs enable decision** (infra from 2B-2)
2. **FP8 attention** (QK^T/PV e4m3, from Phase 3 base)
3. **Aggressive elementwise fusion that changes operation order** (e.g.
   folding RMSNorm/modulate chains across dtype boundaries beyond what
   FUSE_NORM does today)
4. **Any approximation of the attention mask / sparse layout** (topk_ratio
   reduction, coarser block masks, approximate selection)
5. **Any cache-behaviour change that could affect temporal consistency**
   (KV window policy, decoder mem-block reuse across chunks)

### 4.2 Quality protocol (applies to every Phase-4 item)

- **Reference set:** examples 0–3 at 768x1408 (and one clip @1536), generated
  fresh with the flag OFF at the current commit (do not reuse stale outputs).
- **Metrics:** per-clip PSNR and SSIM vs reference; `max|diff|`; report the
  worst clip, not the average. Existing harness patterns:
  `test_fuse_norm.py` (PSNR), `test_cache_lossless.py` (bit-diff).
- **Visual check:** side-by-side of the worst-PSNR clip (temporal flicker is
  the failure mode PSNR misses — scrub frame-by-frame around scene motion).
- **Acceptance tiers:** ≥49 dB → eligible for "recommended config" listing;
  45–49 dB → ships flag-gated with documented delta; <45 dB → revert or
  redesign.
- **Comparison rule:** quality is always measured against flag-OFF at the
  same commit (isolates the numeric change from unrelated drift).
- **Default:** OFF. A Phase-4 flag may only become part of the recommended
  config line after two independent bench+quality entries agree.

---

## 5. Benchmark and logging system

All results — successes, failures, reverts — go to
[`PHASE_BENCH_LOG.md`](./PHASE_BENCH_LOG.md), chronologically. Failures are
as valuable as wins; do not delete entries, mark them `revert`.

### 5.1 Required fields per entry

Date/time · phase · optimization name · commit hash (or patch name) · files
changed · env vars used · exact benchmark command · resolution · frames ·
warmup/steady settings · FPS before/after/Δ · steady chunk before/after/Δ ·
peak mem before/after · correctness result · output-difference result (if
applicable) · Nsight report path (if generated) · decision
(`keep-enabled` / `keep-behind-flag` / `revert` / `investigate` / `postpone`)
· **interpretation (2–3 sentences, mandatory)**.

### 5.2 Tables

Per-change table:

| Date | Phase | Optimization | Flag | FPS Before | FPS After | Delta | Steady Chunk Before | Steady Chunk After | Peak Mem Before | Peak Mem After | Correctness | Decision |
|------|-------|--------------|------|------------|-----------|-------|---------------------|--------------------|-----------------|----------------|-------------|----------|

Cumulative stack table (updated whenever a flag joins the recommended set):

| Step | Enabled Optimizations | FPS | Steady Chunk Time | Peak Memory | Delta vs Phase-2 Baseline | Notes |
|------|----------------------|-----|-------------------|-------------|---------------------------|-------|

### 5.3 The gate

> **Do not continue to the next optimization until the current optimization
> has a benchmark entry and a short written interpretation.**

No exceptions — including "obvious" wins and including reverts.

---

## 6. Benchmark commands

Infrastructure lives in `examples/WanVSR/profiling/`:
`run_pipe_target.py` · `nsys_run.sh` · `ncu_run.sh` · `ncu_batch.sh` ·
`analyze_gaps.py` · `ncu_extract.py` · `bench_ceilings.py`.

### 6.1 Primary (untraced — the only FPS source of truth)

```bash
cd examples/WanVSR
FLASHVSR_CONV3D_BACKEND=gemm \
FLASHVSR_TCDECODER_CHANNELS_LAST=1 \
FLASHVSR_FUSE_NORM=1 \
FLASHVSR_ATTN_BACKEND=triton \
FLASHVSR_CACHE_MOD=1 \
FLASHVSR_CACHE_MASK_BIAS=1 \
FLASHVSR_PROF_STEADY=off \
/root/FlashVSR/venv/bin/python profiling/run_pipe_target.py
```

Add the flag under test (`FLASHVSR_<NEW>=1`) for the "after" run; run
before/after back-to-back in the same session; 3 runs, log the median.
High-res spot check: prepend `FLASHVSR_PROF_W=1536 FLASHVSR_PROF_H=2560`.

### 6.2 Optional attribution runs (never for headline FPS)

Traced wall time is distorted (capture stop-flush lands inside the timer;
CPU tracing overhead inflates gaps) — use these only to explain a result:

```bash
# timeline attribution (minimal trace + GPU metrics):
FLASHVSR_NVTX=1 FLASHVSR_PROF_STEADY=0:-1 <knobs> \
  ./profiling/nsys_run.sh <tag> --gpu-metrics-devices=0 --gpu-metrics-frequency=20000
/root/FlashVSR/venv/bin/python profiling/analyze_gaps.py profiling/reports/<tag> \
  --ref-wall-per-chunk <untraced_steady_seconds>

# kernel deep-dive:
FLASHVSR_NVTX=1 FLASHVSR_PROF_WARMUP=0 <knobs> \
  ./profiling/ncu_run.sh <tag> --target-processes application-only \
  --set full -k "regex:<kernel>" --launch-skip 6 --launch-count 4
python3 profiling/ncu_extract.py profiling/reports/ncu/<tag>.ncu-rep --csv <tag>.csv
```

### 6.3 Known workarounds & operational safety (keep these)

- **ncu child-injection deadlock fix** (already baked into `ncu_run.sh`):
  `TRITON_LIBCUDA_PATH=/usr/lib/aarch64-linux-gnu` and
  `--target-processes application-only`. Root cause: triton's
  `libcuda_dirs()` spawns `ldconfig`; ncu's child injection intermittently
  deadlocks that handshake (main python stuck in `subprocess.communicate`).
- Launch long profiling jobs detached: `setsid nohup <cmd> > log 2>&1 &` —
  interactive aborts kill the whole process group otherwise.
- Never write `pkill -f <pattern>` / `pgrep -f <pattern>` where `<pattern>`
  appears verbatim in your own command line (self-kill); use the
  `[b]racket` trick: `pgrep -f "run_pipe_[t]arget"`.
- Keep clocks locked during a measurement campaign (`nvidia-smi -lgc
  1980,1980`); note that the 700 W platform cap still sags clocks under load
  — never compare runs taken minutes apart without checking `dmon` logs if a
  result looks anomalous.

---

## 7. Sequencing summary

```
Step 0   Fresh 3-run baseline  →  PHASE_BENCH_LOG.md
Phase 2A 1 RoPE cache/fusion → 2 KV ring buffer → 3 attn strided IO
         → 4 mask_gen lean → 5 LQ proj lean → (6 CUDA Graphs go/no-go)
Phase 2B 1 decoder overlap → 2 FP8 GEMM infra (OFF) → 3 fused mask topk
         → 4 fused im2col (conditional)
Phase 3  attention kernel v2 (separate PR, parallel-track allowed once 2A done)
Phase 4  quality-gated enables: FP8 GEMM, FP8 attention, fusion reorders,
         sparsity experiments
```

Rationale for the order: 2A banks low-risk efficiency first (which, under the
700 W cap, also buys back clock headroom), 2B adds the structural wins with
contained blast radius, Phase 3 is the big rock developed against an already
faster baseline, and Phase 4 only ever trades quality knowingly, never by
accident.
