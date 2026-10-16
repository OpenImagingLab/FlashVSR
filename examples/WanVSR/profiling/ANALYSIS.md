# FlashVSR v1.1 Tiny — GH200 Deep Profiling Analysis (Phase 1–3)

Campaign date: 2026-07-08 · GH200 480GB (sm_90, 132 SM, 96GB HBM3) · driver 595.58.03
CUDA 13.2 · torch 2.12 (NGC 26.05) · triton 3.7 · cuDNN 9.22 · nsys 2026.2.1 · ncu 2026.1.1
Baseline config = all PR knobs ON (`gemm + NHWC + fuse_norm + triton attn + TMA + caches`).
Clocks locked at 1980 MHz (`nvidia-smi -lgc`), ncu run with `--clock-control none`.

## 0. Headline numbers (untraced references, single runs)

| Config | FPS | steady chunk | px-norm FPS | peak mem |
|---|---|---|---|---|
| 768x1408 full-knobs | **38.55** | 156.2 ms | 38.55 | 12.6 GiB |
| 1024x1920 full-knobs | 21.77 | 274.1 ms | 39.58 | 20.2 GiB |
| 1536x2560 full-knobs | 11.01 | 531.5 ms | 40.05 | 37.4 GiB |
| 768x1408 sparse attn | 35.35 | 176.2 ms | 35.35 | 12.6 GiB |

px-norm FPS *rises* with resolution → the GPU is slightly under-fed at 768 but the
bottleneck structure is scale-invariant (attention ~47% everywhere).

## 1. Where the time goes

### 1.1 E2E wall budget (768x1408, F=81, traced shares match untraced totals)

| Segment | time | share |
|---|---|---|
| chunk0+chunk1 (warm chunks) | ~516 ms | 26% |
| chunks 2..7 (steady, 6×156 ms) | ~937 ms | 47% |
| TCDecoder decode | ~343–430 ms | 17–21% |
| color_fix + misc python | ~10 ms | <1% |

Decode is **fully serialized** after the denoise loop (single stream). Same shape at
1024/1536: decode = 22–23% of (chunks+decode).

### 1.2 Steady chunk GPU ledger (@768, per chunk ≈156 ms wall, ≈151 ms GPU busy, idle 3.1%)

| Phase | ms/chunk | % GPU | evidence |
|---|---|---|---|
| attn kernel `_bsfa_tma_kernel` (30×2.04 ms) | 58.7 | 39% | ncu: SM 40%, tensor 40%, occ 12.5% |
| attn transposes/copies (attn_core − kernel) | 11.6 | 7.7% | (S,n,d)→(n,S,d) `.contiguous()` ×3 + out |
| ffn (2 GEMM + gelu + gate) | 22.4 | 15% | GEMMs healthy (82% SOL) |
| rope apply (q,k) | 10.1 | 6.8% | elementwise, fusable |
| lq_conv1+conv2 (im2col+GEMM) | 11.7 | 7.8% | im2col copy = 29% of path |
| xattn (q/o GEMM + FA2 kernel) | 7.7 | 5.1% | `flash_fwd_kernel` 85 µs |
| mask_gen (pool+einsum+softmax+topk chain) | 7.4 | 4.9% | gatherTopK: SM 5%, 0.1 wave |
| qkv_norm (3 GEMM + RMSNorm) | 5.8 | 3.9% | |
| win_part / reorder / win_rev / cache_trim | 5.7 | 3.7% | copies at 66–81% DRAM BW |
| kv_cat (KV cache concat) | 3.4 | 2.3% | 3235 GB/s — at BW limit |
| mod1 + gate1 | 3.0 | 2.0% | fused kernels healthy (88% mem SOL) |
| head/patchify/unpatchify/lq_linears | ~1.5 | 1% | |
| **GPU idle within chunk** | **4.8** | **3.1%** | not launch-bound |

## 2. Kernel deep-dives (ncu)

### 2.1 `_bsfa_tma_kernel` — the #1 target (39% of GPU)
```
duration 2.03 ms · SM SOL 40.2% · tensor pipe 40.2% (active 46.4%)
DRAM 134 GB/s (3.3%) · L2 hit 92.5% · achieved WGMMA math ≈ 393 TFLOP/s
occupancy 12.5% (1 block/SM: 178 reg/thr + 229 KB smem) · 8 warps/SM
stalls/issue: barrier 2.39 · wait 1.00 · short_sb 0.55 (of 6.37 total)
scheduler: 68.6% cycles with NO eligible warp
```
Diagnosis: single-block-per-SM Triton pipeline; all 8 warps hit the same stage
barriers, nothing else to schedule → tensor pipe idles 60% of the time.
TMA=0 variant: 2.20 ms, long_scoreboard 0.87 (TMA removed it), 235 regs.

Reference points at the exact shape (q 8448 × kv 25344, h12 d128):
| Kernel | time | note |
|---|---|---|
| cuDNN dense SDPA (full attention) | **1.86 ms** | 707 TF/s, computes 1.65× the FLOPs |
| ideal sparse = dense × 0.606 | **1.13 ms** | efficiency ceiling |
| `_bsfa_tma` (ours) | 2.03 ms | **56% of ideal** |
| FlexAttention + BlockMask (torch 2.12) | 1.92 ms | not the answer (59% of ideal) |

→ A warp-specialized / 2-CTA / deeper-pipelined block-sparse kernel (FA3-style,
CUTLASS FMHA or hand-tuned Triton WS) has **~0.9 ms/call** headroom ⇒ ~26 ms/chunk.

### 2.2 GEMMs — already near bf16 ceiling, FP8 is the lever
| kernel (nvjet) | avg | SM SOL | tensor | note |
|---|---|---|---|---|
| 256x128 coopA (qkv/o/ffn) | 88.7 µs | 82.7% | 82.7% | waves=1.0, perfect fit |
| 192x192 coopB (im2col conv) | 302.9 µs | 85.0% | 85.0% | 928 GB/s |

FP8 microbench (torch._scaled_mm, M=8448): qkv/o ×1.59 · ffn1 ×1.55 · ffn2 ×1.72
· lq_linear ×1.55 (1006–1377 TF/s). GEMM total ≈ 33 ms/chunk → FP8 saves ~13 ms/chunk.

### 2.3 Elementwise & copies
Two classes: (a) already BW-bound (kv_cat 3235 GB/s = 81% HBM3; big copies 3268 GB/s)
→ only fix is *not doing the work*; (b) inefficient strided transposes
(65 µs × 8/chunk, SM-bound 71%, only 980 GB/s) — these are the triton-attn-path
`(S,n,d)→(n,S,d)` contiguous() calls → killable via kernel-side strides.

### 2.4 mask_gen topk chain
gatherTopK 94.6 µs at **SM 5.2%, 0.1 waves** + 7–9 radix mini-kernels per call.
Pure latency, single fused kernel could do it in ~1 ms/chunk (now 7.4 ms).

### 2.5 LQ projector conv (im2col+GEMM)
Path split @ conv2 shape: pad+cat 8% · **im2col copy 29%** · addmm 65% (707 TF/s).
cuDNN 9.22 direct conv3d at this shape: **152 ms vs 8.4 ms** (18× slower) — no new
engine on Hopper; GEMM path remains mandatory; fusing im2col into the GEMM
(CUTLASS conv or Triton fused) reclaims ~4 ms/chunk.

### 2.6 Decoder
NHWC convs healthy; decode-window tensor pipe 37.8%, DRAM 16%. Main finding is
architectural: decode is 100% serial tail (see §3 H6).

## 3. Hypothesis results

| # | Hypothesis | Result | Ceiling @768 |
|---|---|---|---|
| H1 | launch-bound → CUDA Graphs | idle only 3.1%/chunk (0.1% @1024) | ≤4.8 ms/chunk |
| H2 | attn kernel inefficiency | 56% of ideal-sparse; barrier-stalled 1-CTA/SM | ~26 ms/chunk (bf16); more with FP8 attn |
| H3 | im2col overhead / cuDNN engine | im2col 29% of path; cuDNN still 18× slower | ~4 ms/chunk |
| H4 | GEMM efficiency / FP8 | bf16 at 82–85% SOL (no tuning left); FP8 ×1.55–1.72 | ~13 ms/chunk |
| H5 | elementwise BW | fused kernels near BW; transposes+rope+win copies removable | ~20 ms/chunk (transposes 11.6 + rope ~8) |
| H6 | decode serialization | decode = 17–23% of E2E, zero overlap today | +21–29% FPS if hidden |
| H7 | kv_cat + mask_gen | 3.4 + 7.4 ms/chunk | ~9 ms/chunk (ring buffer + fused topk) |
| H8 | power/clock residency | **700W platform cap** (900W denied); under load 649–687W, clocks sag to 1635–1815 MHz (84–92%) | efficiency wins compound; brute-force math won't |

Extra: sparse backend (`block_sparse_attn`) steady chunk 176.2 ms vs triton 156.2 ms;
sparse also shows 8.2% idle — its per-call `torch.tensor(..., device=...)` cu_seqlens
creation is a hidden H2D sync the triton path avoids.

## 4. Ranked Phase-2 roadmap (gain × confidence / effort)

| # | Optimization | est. gain @768 (chunk 156 ms) | conf. | effort |
|---|---|---|---|---|
| 1 | **Attention kernel v2** (warp-specialized/2-CTA pipelined block-sparse; CUTLASS FMHA base or Triton WS; keep exact mask) | −26 ms | high (ref kernels prove it) | high |
| 2 | **Decoder overlap** (stream decode per-chunk on side stream, TCDecoder already streaming-capable) | +21–29% FPS E2E | high | med |
| 3 | **Kill attn-path transposes** (strided kernel IO or fold layout into kernel) | −11.6 ms | high | low-med |
| 4 | **FP8 GEMMs** (qkv/o/ffn/lq_linears via _scaled_mm, per-tensor scales, PSNR-gate) | −13 ms | med-high | med |
| 5 | **RoPE fusion** (single kernel for q+k, or fold into attn prologue; also cache per-chunk freqs — 44 ms/chunk CPU-side waste seen in traces) | −8 ms | high | low |
| 6 | **Fused topk mask_gen** (one kernel replaces gatherTopK+radix chain) | −5 ms | med | med |
| 7 | **Fused im2col-GEMM conv** (CUTLASS conv3d or Triton) | −4 ms | med | med |
| 8 | **KV ring buffer** (preallocated, no cat) | −3 ms | high | low |
| 9 | **CUDA Graphs on steady chunk** | −4.8 ms | med (shapes static per chunk) | med |
| 10 | FP8 attention (QK^T/PV in e4m3, needs quality gate) | −15–25 ms extra | low-med | high |

Stacked (1,3,4,5,6,7,8,9 conservative): chunk 156 → ~90–100 ms ⇒ steady denoise
~80–89 FPS; with decoder overlap E2E ≈ **75–90 FPS** (~2–2.3× over 38.5) before
FP8-attention. Power cap (H8) will claw back some of this — efficiency-first
ordering maximizes what survives.

## 5. Methodology notes
- Traced idle% is an upper bound; corrected against untraced per-chunk walls
  (`[chunks]` line from `run_pipe_target.py`). nsys minimal trace (`cuda,nvtx`)
  used for gap analysis; rich trace only for API attribution.
- ncu occupancy/SOL under `--clock-control none` with global 1980 MHz lock;
  absolute TF/s ceilings should assume ~1650–1815 MHz effective under power cap.
- **ncu child-injection deadlock**: triton's `libcuda_dirs()` spawns `ldconfig`;
  ncu tree-injection intermittently deadlocks that handshake (main python stuck in
  `subprocess.communicate`). Fix baked into `ncu_run.sh`: `TRITON_LIBCUDA_PATH`
  env + `--target-processes application-only`.
- nsys python-sampling produced no SAMPLING_CALLCHAINS on this aarch64 build;
  irrelevant given idle ≈3%.

## 6. Artifacts
```
profiling/
  run_pipe_target.py      env-driven target (W/H/F, steady window, per-chunk timing)
  nsys_run.sh / ncu_run.sh / ncu_batch.sh   drivers (dmon logging, deadlock fix)
  analyze_gaps.py         sqlite analyzer -> analysis.md/kernels.csv/phases.csv/gaps.csv/summary.json
  ncu_extract.py          .ncu-rep -> compact metric table (SOL/tensor/occ/stalls)
  bench_ceilings.py       H2/H3/H4 microbenches
  reports/
    r1a_768_fullknobs/    nsys + gpu-metrics, full measured call  (+analysis.md)
    r1b_768_richapi/      nsys rich API trace, chunks 2-6
    r2_768_pysample/      (python sampling unavailable on aarch64)
    r3_768_sparse/        sparse-attn single-diff
    r4_1024_fullknobs/ r5_1536_fullknobs/  resolution shift
    ncu/                  attn_bsfa_tma{0,1}, gemms, elemwise, masktopk, decoder (+csv)
```
NVTX instrumentation: `FLASHVSR_NVTX=1` (default OFF, zero-effect), steady window
via `FLASHVSR_PROFILER_START_CHUNK/STOP_CHUNK` (consumed per-call in
`flashvsr_tiny.py`). One behavioural fix: pipeline now honours `progress_bar_cmd`.
