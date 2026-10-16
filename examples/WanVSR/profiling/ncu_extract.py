#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract a compact per-kernel metric table from a .ncu-rep file.

Usage: ncu_extract.py report.ncu-rep [--csv out.csv]

Emits one row per profiled kernel instance with the metrics that matter for
the FlashVSR bottleneck analysis: duration, SM/mem SOL, DRAM, tensor-pipe
activity, occupancy + limiter, top warp stalls, L2 hit rate, launch config.
"""
import argparse
import csv
import io
import subprocess
import sys
from collections import defaultdict

RAW_METRICS = {
    "gpu__time_duration.sum": ("dur_us", "time"),
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": ("sm_sol_pct", 1),
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": ("mem_sol_pct", 1),
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed": ("dram_pct", 1),
    "dram__bytes.sum.per_second": ("dram_gbs", "rate"),
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed": ("tensor_pct", 1),
    "launch__occupancy_limit_registers": ("occ_lim_reg", 1),
    "launch__occupancy_limit_shared_mem": ("occ_lim_smem", 1),
    "sm__warps_active.avg.pct_of_peak_sustained_active": ("occ_achieved_pct", 1),
    "launch__registers_per_thread": ("regs", 1),
    "launch__shared_mem_per_block_dynamic": ("smem_kb", "size_kb"),
    "launch__grid_size": ("grid", 1),
    "launch__block_size": ("block", 1),
    "launch__waves_per_multiprocessor": ("waves", 1),
    "lts__t_sector_hit_rate.pct": ("l2_hit_pct", 1),
    "l1tex__t_sector_hit_rate.pct": ("l1_hit_pct", 1),
    "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio": ("st_barrier", 1),
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio": ("st_longsb", 1),
    "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio": ("st_shortsb", 1),
    "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio": ("st_wait", 1),
    "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio": ("st_mio", 1),
    "smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio": ("st_lg", 1),
    "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio": ("st_math", 1),
    "smsp__average_warps_issue_stalled_drain_per_issue_active.ratio": ("st_drain", 1),
}

COLS = ["kernel", "n", "dur_us", "sm_sol_pct", "mem_sol_pct", "tensor_pct",
        "dram_gbs", "occ_achieved_pct", "occ_lim", "regs", "smem_kb", "grid",
        "block", "waves", "l2_hit_pct", "l1_hit_pct", "top_stalls"]


TIME_SCALE = {"nsecond": 1e-3, "usecond": 1.0, "msecond": 1e3, "second": 1e6,
              "ns": 1e-3, "us": 1.0, "ms": 1e3, "s": 1e6}
RATE_SCALE = {"byte/s": 1e-9, "Kbyte/s": 1e-6, "Mbyte/s": 1e-3,
              "Gbyte/s": 1.0, "Tbyte/s": 1e3}
SIZE_SCALE = {"byte": 1e-3, "Kbyte": 1.0, "Mbyte": 1e3}


def _scale(kind, unit):
    if kind == "time":
        return TIME_SCALE.get(unit, 1.0)
    if kind == "rate":
        return RATE_SCALE.get(unit, 1.0)
    if kind == "size_kb":
        return SIZE_SCALE.get(unit, 1.0)
    return kind  # numeric passthrough


def load(rep):
    out = subprocess.run(
        ["ncu", "--import", rep, "--page", "raw", "--csv"],
        capture_output=True, text=True, check=True).stdout
    rows = list(csv.reader(io.StringIO(out)))
    hdr = rows[0]
    units = None
    body = rows[1:]
    if body and body[0] and not body[0][0].strip('"').isdigit():
        units = body[0]
        body = body[1:]
    name_i = hdr.index("Kernel Name")
    idx = {}
    for i, h in enumerate(hdr):
        if h in RAW_METRICS:
            key, kind = RAW_METRICS[h]
            unit = units[i] if units else ""
            idx[i] = (key, _scale(kind, unit))
    recs = []
    for r in body:
        if len(r) <= name_i:
            continue
        rec = {"kernel": r[name_i]}
        for i, (key, scale) in idx.items():
            try:
                rec[key] = float(r[i]) * scale
            except (ValueError, IndexError):
                rec[key] = None
        recs.append(rec)
    return recs


def agg(recs):
    groups = defaultdict(list)
    for r in recs:
        groups[r["kernel"]].append(r)
    table = []
    for k, rs in groups.items():
        def m(key):
            vals = [r[key] for r in rs if r.get(key) is not None]
            return sum(vals) / len(vals) if vals else None
        stalls = {s: m(s) or 0 for s in
                  ["st_barrier", "st_longsb", "st_shortsb", "st_wait",
                   "st_mio", "st_lg", "st_math", "st_drain"]}
        top = sorted(stalls.items(), key=lambda x: -x[1])[:3]
        top_s = ",".join(f"{n[3:]}={v:.2f}" for n, v in top if v > 0.05)
        lim = []
        if (m("occ_lim_reg") or 99) <= 2:
            lim.append("reg")
        if (m("occ_lim_smem") or 99) <= 2:
            lim.append("smem")
        row = dict(kernel=k[:70], n=len(rs), dur_us=m("dur_us"),
                   sm_sol_pct=m("sm_sol_pct"), mem_sol_pct=m("mem_sol_pct"),
                   tensor_pct=m("tensor_pct"), dram_gbs=m("dram_gbs"),
                   occ_achieved_pct=m("occ_achieved_pct"),
                   occ_lim="+".join(lim) or "-", regs=m("regs"),
                   smem_kb=m("smem_kb"), grid=m("grid"), block=m("block"),
                   waves=m("waves"), l2_hit_pct=m("l2_hit_pct"),
                   l1_hit_pct=m("l1_hit_pct"), top_stalls=top_s)
        table.append(row)
    table.sort(key=lambda r: -(r["dur_us"] or 0) * r["n"])
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rep")
    ap.add_argument("--csv")
    args = ap.parse_args()
    recs = load(args.rep)
    table = agg(recs)
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLS)
            w.writeheader()
            for r in table:
                w.writerow({c: (round(r[c], 2) if isinstance(r[c], float) else r[c])
                            for c in COLS})
    fmt = ("{kernel:52.52s} n={n:<3d} {dur_us:>8.1f}us sm={sm_sol_pct:>5.1f}% "
           "mem={mem_sol_pct:>5.1f}% tc={tensor_pct} dram={dram_gbs} "
           "occ={occ_achieved_pct:>5.1f}%({occ_lim}) reg={regs:.0f} "
           "smem={smem_kb:.1f}KB grid={grid:.0f} waves={waves:.1f} "
           "l2={l2_hit_pct:.0f}% stalls[{top_stalls}]")
    for r in table:
        rr = dict(r)
        rr["tensor_pct"] = f"{r['tensor_pct']:.1f}%" if r["tensor_pct"] is not None else "n/a"
        rr["dram_gbs"] = f"{r['dram_gbs']:.0f}GB/s" if r["dram_gbs"] is not None else "n/a"
        for key in ("dur_us", "sm_sol_pct", "mem_sol_pct", "occ_achieved_pct",
                    "regs", "smem_kb", "grid", "waves", "l2_hit_pct"):
            if rr[key] is None:
                rr[key] = float("nan")
        try:
            print(fmt.format(**rr))
        except Exception:
            print(rr)


if __name__ == "__main__":
    main()
