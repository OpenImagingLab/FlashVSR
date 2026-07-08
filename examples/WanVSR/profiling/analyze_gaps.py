#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deep nsys report analyzer for FlashVSR profiling campaign.

Consumes a report dir produced by nsys_run.sh (profile.nsys-rep) and emits:
  - analysis.md   : human-readable findings
  - kernels.csv   : top kernels within the steady window (time, count, grid, category)
  - phases.csv    : GPU time attributed to each NVTX leaf phase
  - gaps.csv      : every GPU idle gap >= threshold with NVTX phase attribution
  - summary.json  : machine-readable rollup (used later by ANALYSIS.md synthesis)

Method notes
------------
* GPU busy time = union of kernel+memcpy+memset intervals on the device
  (all streams merged), computed inside each steady NVTX chunk window.
* Phase attribution maps each GPU activity to the NVTX range that was open on
  the launching CPU thread at launch-API time (correlationId join), taking the
  innermost containing range whose name is in the known phase set.
* Tracing inflates CPU time, so traced idle% is an UPPER bound. If
  --ref-wall-per-chunk (untraced seconds/chunk) is given we also report a
  corrected idle%  = 1 - busy_per_chunk / ref_wall_per_chunk.
* GPU metrics (if sampled) are averaged per phase using sample timestamps
  falling inside phase-attributed GPU activity intervals.
"""
import argparse
import bisect
import csv
import json
import os
import re
import sqlite3
import subprocess
import sys
from collections import defaultdict

STEADY_CHUNKS = ["chunk2", "chunk3", "chunk4", "chunk5", "chunk6"]

LEAF_PHASES = [
    "lq_proj0", "lq_proj", "lq_conv1", "lq_conv2", "lq_linears",
    "patchify", "rope_freqs", "mod1", "qkv_norm", "rope", "win_part",
    "kv_cat", "reorder", "mask_gen", "attn_core", "cache_trim", "win_rev",
    "o_proj", "gate1", "xattn", "ffn", "head", "unpatchify",
    "decode", "color_fix",
]
# parent ranges we also record for rollups
PARENT_PHASES = ["dit_forward", "self_attn"]
CHUNK_RE = re.compile(r"^chunk(\d+)$")

KEY_METRICS = [
    "SMs Active [Throughput %]",
    "SM Issue [Throughput %]",
    "Tensor Active [Throughput %]",
    "DRAM Read Bandwidth [Throughput %]",
    "DRAM Write Bandwidth [Throughput %]",
]


def categorize(name):
    n = name.lower()
    if any(k in n for k in ["_bsfa", "bsfa_", "block_sparse", "fmha", "flash", "attention"]):
        return "attention"
    if "softmax" in n:
        return "softmax/mask"
    if ("cudnn" in n or "implicit_gemm" in n or "xmma_fprop" in n
            or ("conv" in n and "gemm" not in n)):
        return "conv-cudnn(decoder)"
    if any(k in n for k in ["nvjet", "gemm", "cutlass", "cublas", "addmm", "matmul", "wgmma"]):
        return "gemm"
    if any(k in n for k in ["nchwtonhwc", "nhwctonchw", "tonhwc", "tonchw"]):
        return "layout"
    if any(k in n for k in ["topk", "radix", "sort", "scan", "bitonic"]):
        return "topk/sort(mask)"
    if any(k in n for k in ["upsample", "interpolate", "resize", "grid_sample"]):
        return "resample(decoder)"
    if any(k in n for k in ["cat", "copy", "pad", "transpose", "permute", "contiguous",
                            "gather", "scatter", "index", "slice", "unfold", "im2col", "col2im"]):
        return "copy/cat/im2col"
    if any(k in n for k in ["norm", "rms", "silu", "gelu", "relu", "sigmoid", "elementwise",
                            "vectorized", "reduce", "mul", "add", "div", "triton_", "mean", "pow"]):
        return "elementwise/norm"
    if "memcpy" in n or "memset" in n:
        return "memcpy/memset"
    return "other"


def tid_of(global_tid):
    return global_tid  # keep raw; NVTX and RUNTIME use same encoding


class Analyzer:
    def __init__(self, report_dir, ref_wall_per_chunk=None, gap_threshold_us=2.0):
        self.dir = report_dir
        self.ref_wall_per_chunk = ref_wall_per_chunk
        self.gap_thr_ns = int(gap_threshold_us * 1000)
        self.db = self._open_db()
        self.strings = self._load_strings()

    # ---------- loading ----------
    def _open_db(self):
        rep = os.path.join(self.dir, "profile.nsys-rep")
        db = os.path.join(self.dir, "profile.sqlite")
        if not os.path.exists(db):
            subprocess.run(["nsys", "export", "-t", "sqlite", "--force-overwrite=true",
                            "-o", db, rep], check=True, capture_output=True)
        return sqlite3.connect(db)

    def _load_strings(self):
        return dict(self.db.execute("SELECT id, value FROM StringIds"))

    def _table_exists(self, name):
        return self.db.execute(
            "SELECT count(*) FROM sqlite_master WHERE type='table' AND name=?",
            (name,)).fetchone()[0] > 0

    def load(self):
        # GPU activities
        self.kernels = self.db.execute(
            "SELECT start, end, streamId, correlationId, shortName, demangledName, "
            "gridX, gridY, gridZ, blockX, blockY, blockZ, registersPerThread, "
            "staticSharedMemory, dynamicSharedMemory "
            "FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start").fetchall()
        self.memcpys = self.db.execute(
            "SELECT start, end, streamId, correlationId, bytes, copyKind "
            "FROM CUPTI_ACTIVITY_KIND_MEMCPY ORDER BY start").fetchall() \
            if self._table_exists("CUPTI_ACTIVITY_KIND_MEMCPY") else []
        self.memsets = self.db.execute(
            "SELECT start, end, streamId, correlationId FROM CUPTI_ACTIVITY_KIND_MEMSET "
            "ORDER BY start").fetchall() \
            if self._table_exists("CUPTI_ACTIVITY_KIND_MEMSET") else []

        # Runtime API rows for correlation (launch thread + time) and sync analysis
        self.rt_by_corr = {}
        self.sync_apis = []
        self.launch_apis = []
        for start, end, gtid, corr, name_id in self.db.execute(
                "SELECT start, end, globalTid, correlationId, nameId "
                "FROM CUPTI_ACTIVITY_KIND_RUNTIME"):
            name = self.strings.get(name_id, "")
            self.rt_by_corr[corr] = (start, end, gtid, name)
            if "Synchronize" in name or "cudaEventSynchronize" in name:
                self.sync_apis.append((start, end, gtid, name))
            if name.startswith("cudaLaunchKernel") or name.startswith("cuLaunchKernel"):
                self.launch_apis.append((start, end, gtid))
        self.launch_apis.sort()

        # NVTX ranges (eventType 59 = PushPop)
        self.nvtx = []
        for start, end, text, text_id, gtid in self.db.execute(
                "SELECT start, end, text, textId, globalTid FROM NVTX_EVENTS "
                "WHERE eventType = 59 AND end IS NOT NULL"):
            name = text if text else self.strings.get(text_id, "")
            self.nvtx.append((start, end, name, gtid))
        self.nvtx.sort()

        # per-tid sorted ranges for innermost lookup
        self.nvtx_by_tid = defaultdict(list)
        for start, end, name, gtid in self.nvtx:
            self.nvtx_by_tid[gtid].append((start, end, name))
        self.nvtx_starts = {t: [r[0] for r in v] for t, v in self.nvtx_by_tid.items()}

        # chunk windows
        self.chunks = {}
        for start, end, name, gtid in self.nvtx:
            m = CHUNK_RE.match(name or "")
            if m:
                self.chunks[name] = (start, end)
        self.top_ranges = {name: (s, e) for s, e, name, _ in self.nvtx
                           if name in ("decode", "color_fix")}

        # GPU metrics
        self.metrics = {}
        if self._table_exists("GPU_METRICS") and self._table_exists("TARGET_INFO_GPU_METRICS"):
            id2name = dict(self.db.execute(
                "SELECT metricId, metricName FROM TARGET_INFO_GPU_METRICS"))
            rows = self.db.execute(
                "SELECT timestamp, metricId, value FROM GPU_METRICS").fetchall()
            per = defaultdict(list)
            for ts, mid, val in rows:
                nm = id2name.get(mid)
                if nm in KEY_METRICS:
                    per[nm].append((ts, val))
            self.metrics = {k: sorted(v) for k, v in per.items()}

    # ---------- helpers ----------
    def innermost_phase(self, gtid, t, allowed):
        starts = self.nvtx_starts.get(gtid)
        if not starts:
            return None
        i = bisect.bisect_right(starts, t) - 1
        ranges = self.nvtx_by_tid[gtid]
        best = None
        # walk left; the first containing range is the innermost, but keep
        # walking to find the innermost whose name is in `allowed`.
        steps = 0
        while i >= 0 and steps < 4096:
            s, e, name = ranges[i]
            if s <= t <= e:
                if name in allowed:
                    return name
                if best is None:
                    best = name
            i -= 1
            steps += 1
        return None

    @staticmethod
    def union_busy(intervals, w0, w1):
        """Total covered time of intervals clipped to [w0,w1]."""
        busy = 0
        cur_s = cur_e = None
        for s, e in intervals:
            if e <= w0 or s >= w1:
                continue
            s, e = max(s, w0), min(e, w1)
            if cur_s is None:
                cur_s, cur_e = s, e
            elif s <= cur_e:
                cur_e = max(cur_e, e)
            else:
                busy += cur_e - cur_s
                cur_s, cur_e = s, e
        if cur_s is not None:
            busy += cur_e - cur_s
        return busy

    @staticmethod
    def gaps_in(intervals, w0, w1, thr):
        """Idle gaps >= thr within [w0,w1] given sorted activity intervals."""
        gaps = []
        cursor = w0
        for s, e in intervals:
            if e <= w0 or s >= w1:
                continue
            s, e = max(s, w0), min(e, w1)
            if s > cursor and s - cursor >= thr:
                gaps.append((cursor, s))
            cursor = max(cursor, e)
        if w1 > cursor and w1 - cursor >= thr:
            gaps.append((cursor, w1))
        return gaps

    # ---------- analyses ----------
    def run(self):
        self.load()
        out = {}
        acts = [(k[0], k[1]) for k in self.kernels] + \
               [(m[0], m[1]) for m in self.memcpys] + \
               [(m[0], m[1]) for m in self.memsets]
        acts.sort()
        self.acts = acts

        steady = [(c, self.chunks[c]) for c in STEADY_CHUNKS if c in self.chunks]
        if not steady:  # fall back to whatever chunks exist
            steady = sorted(self.chunks.items())[2:7]
        out["steady_chunks"] = {c: {"span_ms": (w[1] - w[0]) / 1e6} for c, w in steady}

        # --- per-chunk busy/idle & launches ---
        chunk_rows = []
        for cname, (w0, w1) in steady:
            span = w1 - w0
            busy = self.union_busy(acts, w0, w1)
            nk = sum(1 for k in self.kernels if k[0] >= w0 and k[1] <= w1)
            launch_cpu = sum(min(e, w1) - max(s, w0) for s, e, _ in self.launch_apis
                             if e > w0 and s < w1)
            row = dict(chunk=cname, span_ms=span / 1e6, gpu_busy_ms=busy / 1e6,
                       idle_pct=100 * (1 - busy / span), kernels=nk,
                       launch_cpu_ms=launch_cpu / 1e6)
            if self.ref_wall_per_chunk:
                row["idle_pct_corrected"] = 100 * (1 - (busy / 1e9) / self.ref_wall_per_chunk)
            chunk_rows.append(row)
        out["per_chunk"] = chunk_rows

        # --- phase attribution ---
        allowed = set(LEAF_PHASES)
        phase_gpu = defaultdict(float)
        phase_cnt = defaultdict(int)
        phase_intervals = defaultdict(list)
        kern_agg = defaultdict(lambda: [0.0, 0, None])  # name -> [ns, count, meta]
        steady_windows = [w for _, w in steady]

        def in_steady(s, e):
            return any(s >= w0 and e <= w1 for w0, w1 in steady_windows)

        for k in self.kernels:
            s, e, stream, corr, short_id, dem_id = k[0], k[1], k[2], k[3], k[4], k[5]
            if not in_steady(s, e):
                continue
            name = self.strings.get(short_id) or self.strings.get(dem_id) or "?"
            agg = kern_agg[name]
            agg[0] += (e - s)
            agg[1] += 1
            if agg[2] is None:
                agg[2] = (k[6], k[7], k[8], k[9], k[10], k[11], k[12], k[13], k[14])
            rt = self.rt_by_corr.get(corr)
            phase = None
            if rt:
                phase = self.innermost_phase(rt[2], rt[0], allowed)
            phase = phase or "(unattributed)"
            phase_gpu[phase] += (e - s)
            phase_cnt[phase] += 1
            phase_intervals[phase].append((s, e))
        for m in self.memcpys:
            s, e, corr = m[0], m[1], m[3]
            if not in_steady(s, e):
                continue
            rt = self.rt_by_corr.get(corr)
            phase = self.innermost_phase(rt[2], rt[0], allowed) if rt else None
            phase_gpu[(phase or "(unattributed)") + "|memcpy"] += (e - s)

        total_phase = sum(phase_gpu.values())
        out["phases"] = {p: dict(gpu_ms=v / 1e6, pct=100 * v / total_phase,
                                 count=phase_cnt.get(p, 0))
                         for p, v in sorted(phase_gpu.items(), key=lambda x: -x[1])}

        # --- kernel table ---
        ktable = sorted(((v[0], v[1], n, v[2]) for n, v in kern_agg.items()),
                        reverse=True)
        out["kernels_total_ms"] = sum(v[0] for v in kern_agg.values()) / 1e6
        self.ktable = ktable

        # --- gap analysis ---
        gap_rows = []
        gap_total = 0
        for cname, (w0, w1) in steady:
            for gs, ge in self.gaps_in(acts, w0, w1, self.gap_thr_ns):
                dur = ge - gs
                gap_total += dur
                mid = (gs + ge) // 2
                # phase on the main thread at gap time (any tid owning chunk ranges)
                phase = None
                for gtid in self.nvtx_by_tid:
                    p = self.innermost_phase(gtid, mid, set(LEAF_PHASES + PARENT_PHASES))
                    if p:
                        phase = p
                        break
                # overlapping sync API?
                sync = any(s <= ge and e >= gs for s, e, _, _ in self.sync_apis)
                # next kernel after gap
                idx = bisect.bisect_left(self.acts, (ge, ge))
                nxt = None
                for k in self.kernels:
                    if k[0] >= ge:
                        nxt = self.strings.get(k[4]) or "?"
                        break
                gap_rows.append(dict(chunk=cname, start_us=(gs - w0) / 1e3,
                                     dur_us=dur / 1e3, phase=phase or "?",
                                     sync=sync, next_kernel=(nxt or "?")[:80]))
        gap_rows.sort(key=lambda r: -r["dur_us"])
        out["gaps"] = dict(total_ms=gap_total / 1e6, count=len(gap_rows),
                           threshold_us=self.gap_thr_ns / 1e3)
        # gap rollup per phase
        by_phase = defaultdict(float)
        for r in gap_rows:
            by_phase[r["phase"]] += r["dur_us"]
        out["gaps"]["by_phase_us"] = dict(sorted(by_phase.items(), key=lambda x: -x[1]))
        self.gap_rows = gap_rows

        # --- memcpy inventory (steady) ---
        cp = defaultdict(lambda: [0, 0.0, 0])  # kind -> [count, ms, bytes]
        for m in self.memcpys:
            s, e, kind, nbytes = m[0], m[1], m[5], m[4]
            if not in_steady(s, e):
                continue
            c = cp[kind]
            c[0] += 1
            c[1] += (e - s) / 1e6
            c[2] += nbytes or 0
        out["memcpy_steady"] = {str(k): dict(count=v[0], ms=round(v[1], 3),
                                             MiB=round(v[2] / 2**20, 1))
                                for k, v in cp.items()}

        # --- wall budget over whole capture (if chunk0 exists) ---
        if "chunk0" in self.chunks:
            all_chunks = sorted(self.chunks.items(), key=lambda x: x[1][0])
            t0 = all_chunks[0][1][0]
            t1 = max(e for _, (s, e) in all_chunks)
            budget = {c: (e - s) / 1e6 for c, (s, e) in all_chunks}
            for nm, (s, e) in self.top_ranges.items():
                budget[nm] = (e - s) / 1e6
                t1 = max(t1, e)
            covered = sum(budget.values())
            budget["(uncovered python/IO)"] = (t1 - t0) / 1e6 - covered
            out["wall_budget_ms"] = budget

        # --- GPU metrics per phase ---
        if self.metrics:
            out["gpu_metrics_phase_avg"] = {}
            interesting = ["attn_core", "ffn", "qkv_norm", "o_proj", "xattn",
                           "mask_gen", "lq_proj", "decode", "kv_cat", "reorder"]
            for metric, samples in self.metrics.items():
                ts = [t for t, _ in samples]
                vals = [v for _, v in samples]
                mrow = {}
                # steady-window average
                sel = self._avg_in_windows(ts, vals, steady_windows)
                mrow["steady_all"] = sel
                for ph in interesting:
                    ivs = phase_intervals.get(ph)
                    if ivs:
                        mrow[ph] = self._avg_in_windows(ts, vals, ivs)
                if "decode" in self.top_ranges:
                    mrow["decode_window"] = self._avg_in_windows(
                        ts, vals, [self.top_ranges["decode"]])
                out["gpu_metrics_phase_avg"][metric] = mrow

        self.out = out
        return out

    @staticmethod
    def _avg_in_windows(ts, vals, windows):
        tot = 0.0
        n = 0
        for w0, w1 in windows:
            i0 = bisect.bisect_left(ts, w0)
            i1 = bisect.bisect_right(ts, w1)
            for j in range(i0, i1):
                tot += vals[j]
                n += 1
        return round(tot / n, 2) if n else None

    # ---------- outputs ----------
    def write(self):
        out = self.out
        with open(os.path.join(self.dir, "summary.json"), "w") as f:
            json.dump(out, f, indent=2)

        with open(os.path.join(self.dir, "kernels.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["total_ms", "count", "avg_us", "category", "grid", "block",
                        "regs", "smem_static", "smem_dyn", "name"])
            for tot, cnt, name, meta in self.ktable[:60]:
                grid = f"{meta[0]}x{meta[1]}x{meta[2]}" if meta else ""
                blk = f"{meta[3]}x{meta[4]}x{meta[5]}" if meta else ""
                w.writerow([round(tot / 1e6, 3), cnt, round(tot / cnt / 1e3, 1),
                            categorize(name), grid, blk,
                            meta[6] if meta else "", meta[7] if meta else "",
                            meta[8] if meta else "", name[:160]])

        with open(os.path.join(self.dir, "phases.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["phase", "gpu_ms", "pct", "count"])
            for p, d in out["phases"].items():
                w.writerow([p, round(d["gpu_ms"], 3), round(d["pct"], 2), d["count"]])

        with open(os.path.join(self.dir, "gaps.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["chunk", "start_us", "dur_us", "phase", "sync", "next_kernel"])
            for r in self.gap_rows[:400]:
                w.writerow([r["chunk"], round(r["start_us"], 1), round(r["dur_us"], 1),
                            r["phase"], r["sync"], r["next_kernel"]])

        md = [f"# nsys analysis: {os.path.basename(self.dir)}\n"]
        md.append("## Steady chunks (busy/idle)\n")
        md.append("| chunk | span ms | GPU busy ms | idle % (traced)"
                  + (" | idle % (corrected)" if self.ref_wall_per_chunk else "")
                  + " | kernels | launch CPU ms |")
        md.append("|---|---|---|---|---|" + ("---|" if self.ref_wall_per_chunk else ""))
        for r in out["per_chunk"]:
            row = (f"| {r['chunk']} | {r['span_ms']:.1f} | {r['gpu_busy_ms']:.1f} "
                   f"| {r['idle_pct']:.1f} ")
            if self.ref_wall_per_chunk:
                row += f"| {r['idle_pct_corrected']:.1f} "
            row += f"| {r['kernels']} | {r['launch_cpu_ms']:.1f} |"
            md.append(row)
        md.append("\n## GPU time by NVTX phase (steady window)\n")
        md.append("| phase | GPU ms | % | launches |")
        md.append("|---|---|---|---|")
        for p, d in out["phases"].items():
            md.append(f"| {p} | {d['gpu_ms']:.2f} | {d['pct']:.1f} | {d['count']} |")
        md.append("\n## Top kernels (steady window)\n")
        md.append("| total ms | n | avg us | category | grid | name |")
        md.append("|---|---|---|---|---|---|")
        for tot, cnt, name, meta in self.ktable[:40]:
            grid = f"{meta[0]},{meta[1]},{meta[2]}" if meta else ""
            md.append(f"| {tot/1e6:.2f} | {cnt} | {tot/cnt/1e3:.1f} "
                      f"| {categorize(name)} | {grid} | `{name[:90]}` |")
        md.append(f"\n## Gaps (>= {out['gaps']['threshold_us']:.0f} us)\n")
        md.append(f"total gap: **{out['gaps']['total_ms']:.2f} ms** over "
                  f"{out['gaps']['count']} gaps in steady window")
        md.append("\nby phase (us): " + json.dumps(out["gaps"]["by_phase_us"]))
        md.append("\n### Largest 25 gaps\n")
        md.append("| chunk | at us | dur us | phase | sync | next kernel |")
        md.append("|---|---|---|---|---|---|")
        for r in self.gap_rows[:25]:
            md.append(f"| {r['chunk']} | {r['start_us']:.0f} | {r['dur_us']:.1f} "
                      f"| {r['phase']} | {r['sync']} | `{r['next_kernel'][:60]}` |")
        if "memcpy_steady" in out:
            md.append("\n## Memcpy (steady)\n```json\n"
                      + json.dumps(out["memcpy_steady"], indent=2) + "\n```")
        if "wall_budget_ms" in out:
            md.append("\n## Wall budget (full measured call, traced)\n```json\n"
                      + json.dumps({k: round(v, 1) for k, v in out["wall_budget_ms"].items()},
                                   indent=2) + "\n```")
        if "gpu_metrics_phase_avg" in out:
            md.append("\n## GPU metrics averages\n```json\n"
                      + json.dumps(out["gpu_metrics_phase_avg"], indent=2) + "\n```")
        with open(os.path.join(self.dir, "analysis.md"), "w") as f:
            f.write("\n".join(md) + "\n")

        print(f"[analyze] {self.dir}: busy tables written "
              f"(kernels {out['kernels_total_ms']:.1f} ms in steady window; "
              f"gaps {out['gaps']['total_ms']:.2f} ms)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("report_dir")
    ap.add_argument("--ref-wall-per-chunk", type=float, default=None,
                    help="untraced wall seconds per steady chunk (for corrected idle%)")
    ap.add_argument("--gap-us", type=float, default=2.0)
    args = ap.parse_args()
    a = Analyzer(args.report_dir, args.ref_wall_per_chunk, args.gap_us)
    a.run()
    a.write()


if __name__ == "__main__":
    main()
