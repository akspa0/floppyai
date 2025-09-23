#!/usr/bin/env python3
"""
patterns_to_stream_set.py

Generate a set of KryoFlux C2/OOB stream files (NN.S.raw) for a chosen
track/side range, using synthetic patterns. Files are suitable for
DTC write-from-stream set mode (-i0 -w with -f prefix).

Supported patterns:
- constant: fixed interval across the revolution
- random: Gaussian intervals around mean/std
- alt: alternating long/short intervals
- zeros: long cells (alias of constant with long interval)
- ones: short cells (alias of constant with short interval)
- sweep: interval sweep from min to max across the revolution
- prbs7: pseudo-random bit sequence (LFSR) mapping 0->long, 1->short

Examples:
  python3 FloppyAI/tools/patterns_to_stream_set.py \
    --tracks 0-82 --sides 0,1 --revs 3 \
    --pattern constant --interval-ns 4000 \
    --output-dir ./out/pattern_set_A

  python3 FloppyAI/tools/patterns_to_stream_set.py \
    --tracks 0-40 --sides 0 \
    --pattern random --mean-ns 4000 --std-ns 400 \
    --rev-time-ns 200000000 --revs 3 \
    --output-dir ./out/pattern_set_B

  python3 FloppyAI/tools/patterns_to_stream_set.py \
    --tracks 0-20 --sides 0,1 --revs 3 \
    --pattern prbs7 --long-ns 4200 --short-ns 2200 --seed 123 \
    --output-dir ./out/pattern_set_prbs7

Notes:
- We keep patterns simple and controllable, but ensure non-empty streams.
- The emitter writes OOB info + encoded samples + OOB index per rev + OOB end.
- sck (sample clock) defaults to 24,027,428.5714 Hz unless overridden.
"""
from __future__ import annotations

import argparse
import os
import sys
import math
import random
from pathlib import Path
from typing import List, Tuple

# Ensure repo root on sys.path so we can import FloppyAI.src.* regardless of CWD
HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[2]
SRC_DIR = REPO_ROOT / "FloppyAI" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from stream_export import write_kryoflux_stream  # type: ignore


def parse_range(spec: str) -> List[int]:
    if not spec:
        return []
    if "," in spec:
        vals = []
        for tok in spec.split(","):
            tok = tok.strip()
            if not tok:
                continue
            vals.append(int(tok))
        return vals
    if "-" in spec:
        a, b = spec.split("-", 1)
        a = int(a.strip()); b = int(b.strip())
        if a <= b:
            return list(range(a, b + 1))
        else:
            return list(range(b, a + 1))
    return [int(spec)]


def gen_constant_intervals(rev_time_ns: int, interval_ns: int) -> List[int]:
    interval_ns = max(50, int(interval_ns))
    n = max(1, int(round(rev_time_ns / float(interval_ns))))
    # Adjust last interval so sum is close to rev_time_ns
    intervals = [interval_ns] * n
    total = sum(intervals)
    if total != rev_time_ns:
        intervals[-1] = max(50, intervals[-1] + (rev_time_ns - total))
    return intervals


def gen_random_intervals(rev_time_ns: int, mean_ns: float, std_ns: float) -> List[int]:
    intervals: List[int] = []
    total = 0
    # Keep adding intervals until we reach/reasonably exceed rev_time_ns
    # Clamp intervals to > 0 and reasonable upper bound
    while total < rev_time_ns:
        val = max(50.0, random.gauss(mu=mean_ns, sigma=std_ns))
        ival = int(val)
        intervals.append(ival)
        total += ival
    # Adjust last one to better hit target
    if intervals:
        intervals[-1] = max(50, intervals[-1] - (total - rev_time_ns))
    return intervals


def build_intervals(pattern: str, rev_time_ns: int, **kw) -> List[int]:
    pattern = pattern.lower().strip()
    if pattern == "constant":
        return gen_constant_intervals(rev_time_ns, int(kw.get("interval_ns", 4000)))
    elif pattern == "random":
        return gen_random_intervals(rev_time_ns, float(kw.get("mean_ns", 4000.0)), float(kw.get("std_ns", 400.0)))
    elif pattern == "alt":
        long_ns = int(kw.get("long_ns", 4000))
        short_ns = int(kw.get("short_ns", 2000))
        intervals: List[int] = []
        total = 0
        toggle = True
        while total < rev_time_ns:
            ns = max(50, long_ns if toggle else short_ns)
            intervals.append(ns)
            total += ns
            toggle = not toggle
        if intervals:
            intervals[-1] = max(50, intervals[-1] - (total - rev_time_ns))
        return intervals
    elif pattern == "zeros":
        # Long cells only
        return gen_constant_intervals(rev_time_ns, int(kw.get("long_ns", kw.get("interval_ns", 4000))))
    elif pattern == "ones":
        # Short cells only
        return gen_constant_intervals(rev_time_ns, int(kw.get("short_ns", 2000)))
    elif pattern == "sweep":
        sweep_min = int(kw.get("sweep_min_ns", 2000))
        sweep_max = int(kw.get("sweep_max_ns", 6000))
        # Create a linear sweep across the revolution
        steps = max(2, int(round(rev_time_ns / ((sweep_min + sweep_max) / 2.0))))
        # Ensure at least a handful of steps
        steps = max(16, steps)
        raw = [int(max(50, sweep_min + (sweep_max - sweep_min) * i / (steps - 1))) for i in range(steps)]
        intervals: List[int] = []
        total = 0
        i = 0
        while total < rev_time_ns:
            ns = raw[i % len(raw)]
            intervals.append(ns)
            total += ns
            i += 1
        if intervals:
            intervals[-1] = max(50, intervals[-1] - (total - rev_time_ns))
        return intervals
    elif pattern == "prbs7":
        # Simple PRBS7 LFSR-based bitstream mapping: 0->long, 1->short
        long_ns = int(kw.get("long_ns", 4000))
        short_ns = int(kw.get("short_ns", 2000))
        seed = int(kw.get("seed", 0))
        if seed:
            random.seed(seed)
        # Initialize LFSR with non-zero state
        state = (seed & 0x7F) or 0x5A
        intervals: List[int] = []
        total = 0
        while total < rev_time_ns:
            # PRBS7 taps: x^7 + x^6 + 1 (typical)
            new_bit = ((state >> 6) ^ (state >> 5)) & 1
            state = ((state << 1) & 0x7F) | new_bit
            bit = state & 1
            ns = max(50, short_ns if bit else long_ns)
            intervals.append(ns)
            total += ns
        if intervals:
            intervals[-1] = max(50, intervals[-1] - (total - rev_time_ns))
        return intervals
    else:
        raise ValueError(f"Unknown pattern: {pattern}")


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description="Generate NN.S.raw C2/OOB stream set for DTC write-from-stream (-i0 -w)")
    ap.add_argument("--tracks", required=True, help="Track spec: 'a-b' or 'list' (e.g., 0-82 or 0,2,4)")
    ap.add_argument("--sides", default="0,1", help="Sides list (default: 0,1)")
    ap.add_argument("--revs", type=int, default=3, help="Revolutions per file (default 3)")
    ap.add_argument("--rev-time-ns", type=int, default=200_000_000, help="Target revolution time in ns (default ~300RPM)")
    ap.add_argument("--sck-hz", type=float, default=24_027_428.5714285, help="Sample clock for C2 stream (Hz, default ~24.0274286 MHz)")
    ap.add_argument("--header-mode", choices=["ascii", "oob"], default="oob", help="File header style: ascii preamble or start with OOB info (default oob)")
    ap.add_argument("--version-string", default="3.50", help="Version text for ASCII preamble and KFInfo (default 3.50)")
    ap.add_argument("--no-sck-oob", action="store_true", help="DEPRECATED: Ignored (Type 0x08 Sample Clock OOB is not emitted)")
    # Initial index handling: default is OFF for compatibility
    ap.add_argument("--initial-index", action="store_true", help="Emit an initial OOB index marker at start (default off)")
    ap.add_argument("--no-initial-index", action="store_true", help="DEPRECATED: same as default; prefer --initial-index to enable")
    # KFInfo toggles
    ap.add_argument("--no-kf-version-info", action="store_true", help="Do not emit KFInfo 'KryoFlux stream - version ...' string")
    ap.add_argument("--no-clock-info", action="store_true", help="DEPRECATED: Clocks are disabled by default; use --clock-info to enable")
    ap.add_argument("--clock-info", action="store_true", help="Emit KFInfo 'sck=..., ick=...' string (off by default)")
    ap.add_argument("--hw-info", action="store_true", help="Emit extended HW info KFInfo (host_date, name, hwid, etc.) (off by default)")
    ap.add_argument("--ick-hz", type=float, default=3003428.5714, help="Index clock frequency (Hz) for OOB counters (default ~3.003 MHz)")
    ap.add_argument("--pattern", choices=["constant", "random", "alt", "zeros", "ones", "sweep", "prbs7"], default="constant")
    ap.add_argument("--interval-ns", type=int, default=4000, help="Constant/ones cell interval (ns)")
    ap.add_argument("--mean-ns", type=float, default=4000.0, help="Random pattern mean interval (ns)")
    ap.add_argument("--std-ns", type=float, default=400.0, help="Random pattern stddev interval (ns)")
    ap.add_argument("--long-ns", type=int, default=4000, help="Long cell interval for alt/zeros/prbs7 (ns)")
    ap.add_argument("--short-ns", type=int, default=2000, help="Short cell interval for alt/ones/prbs7 (ns)")
    ap.add_argument("--sweep-min-ns", type=int, default=2000, help="Sweep minimum interval (ns)")
    ap.add_argument("--sweep-max-ns", type=int, default=6000, help="Sweep maximum interval (ns)")
    ap.add_argument("--seed", type=int, default=0, help="Seed for PRBS/random generation (0 = unseeded)")
    ap.add_argument("--output-dir", required=True, help="Directory to write NN.S.raw set")
    args = ap.parse_args(argv)

    tracks = parse_range(args.tracks)
    if not tracks:
        print("No tracks parsed from --tracks", file=sys.stderr)
        return 2
    sides = [int(s.strip()) for s in args.sides.split(',') if s.strip() != '']
    revs = max(1, int(args.revs))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for t in tracks:
        for s in sides:
            # Generate intervals per revolution and concatenate across revs
            intervals: List[int] = []
            rev_lens: List[int] = []
            for _ in range(revs):
                part = build_intervals(
                    pattern=args.pattern,
                    rev_time_ns=int(args.rev_time_ns),
                    interval_ns=int(args.interval_ns),
                    mean_ns=float(args.mean_ns),
                    std_ns=float(args.std_ns),
                    long_ns=int(args.long_ns),
                    short_ns=int(args.short_ns),
                    sweep_min_ns=int(args.sweep_min_ns),
                    sweep_max_ns=int(args.sweep_max_ns),
                    seed=int(args.seed),
                )
                intervals.extend(part)
                rev_lens.append(len(part))
            # Write trackNN.S.raw (provide exact per-rev lengths to align OOB indices)
            fname = f"track{t:02d}.{int(s)}.raw"
            fpath = out_dir / fname
            write_kryoflux_stream(
                intervals,
                track=int(t),
                side=int(s),
                output_path=str(fpath),
                num_revs=revs,
                version=str(args.version_string),
                sck_hz=float(args.sck_hz),
                rev_lengths=rev_lens,
                header_mode=str(args.header_mode),
                include_sck_oob=(not args.no_sck_oob),
                include_initial_index=(True if args.initial_index and not args.no_initial_index else False),
                ick_hz=float(args.ick_hz),
                include_kf_version_info=True,
                include_clock_info=bool(getattr(args, "clock_info", False)),
                include_hw_info=bool(getattr(args, "hw_info", False)),
            )

    print(f"Generated {len(tracks)*len(sides)} files in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
