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
import shutil
from typing import List, Tuple
import importlib
from typing import Optional, Dict, Any

# Optional image support for 'image' pattern (silkscreen)
try:
    from PIL import Image
except Exception:
    Image = None  # type: ignore

# Ensure repo root on sys.path so we can import FloppyAI.src.* regardless of CWD
HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[2]
SRC_DIR = REPO_ROOT / "FloppyAI" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Lazy import writers close to use-site to avoid hard deps (e.g., numpy) when not needed


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


def gen_alt_intervals(rev_time_ns: int, long_ns: int, short_ns: int, segments: int = 64) -> List[int]:
    # Time-domain bars within one revolution: split into segments and alternate long/short
    segments = max(2, int(segments))
    base = []
    use_long = True
    for _ in range(segments):
        base.append(max(50, int(long_ns if use_long else short_ns)))
        use_long = not use_long
    # Normalize to target rev_time by scaling last cell
    total = sum(base)
    if total <= 0:
        return [rev_time_ns]
    scale = rev_time_ns / float(total)
    scaled = [max(50, int(round(v * scale))) for v in base]
    diff = rev_time_ns - sum(scaled)
    scaled[-1] = max(50, scaled[-1] + diff)
    return scaled


def gen_sweep_intervals(rev_time_ns: int, sweep_min: int, sweep_max: int, steps: int = 128) -> List[int]:
    steps = max(2, int(steps))
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


def gen_image_intervals(
    rev_time_ns: int,
    image_path: str,
    long_ns: int,
    short_ns: int,
    row_fraction: float = 0.0,
    threshold: int = 140,
) -> List[int]:
    if Image is None:
        raise RuntimeError("Pillow (PIL) is required for --pattern image. Please pip install pillow.")
    img = Image.open(image_path).convert("L")
    # Decide number of columns (cells per revolution) from target timing
    avg_cell = max(50, int(round((long_ns + short_ns) / 2)))
    columns = max(16, int(round(rev_time_ns / float(avg_cell))))
    # Resize image width to columns, keep height
    w, h = img.size
    if w != columns:
        img = img.resize((columns, h))
        w, h = img.size
    # Choose the row based on track position (0..1), clamp
    rf = 0.0 if h <= 1 else max(0.0, min(1.0, row_fraction))
    row_idx = int(round(rf * (h - 1))) if h > 1 else 0
    px = img.getdata()
    # Extract one row
    row: List[int] = [px[row_idx * w + x] for x in range(w)]
    # Map to intervals
    intervals = [max(50, int(short_ns if v < threshold else long_ns)) for v in row]
    # Adjust last interval to hit target rev_time_ns exactly
    total = sum(intervals)
    if intervals:
        intervals[-1] = max(50, intervals[-1] + (rev_time_ns - total))
    return intervals


def build_intervals(pattern: str, rev_time_ns: int, **kw) -> List[int]:
    pattern = pattern.lower().strip()
    if pattern == "constant":
        return gen_constant_intervals(rev_time_ns, int(kw.get("interval_ns", 4000)))
    elif pattern == "random":
        return gen_random_intervals(rev_time_ns, float(kw.get("mean_ns", 4000.0)), float(kw.get("std_ns", 400.0)))
    elif pattern == "alt":
        # Alternate long/short in fixed segments across the revolution
        return gen_alt_intervals(
            rev_time_ns,
            int(kw.get("long_ns", 4000)),
            int(kw.get("short_ns", 2000)),
            int(kw.get("segments", 64)),
        )
    elif pattern == "zeros":
        # Long cells only
        return gen_constant_intervals(rev_time_ns, int(kw.get("long_ns", kw.get("interval_ns", 4000))))
    elif pattern == "ones":
        # Short cells only
        return gen_constant_intervals(rev_time_ns, int(kw.get("short_ns", 2000)))
    elif pattern == "sweep":
        return gen_sweep_intervals(
            rev_time_ns,
            int(kw.get("sweep_min_ns", 2000)),
            int(kw.get("sweep_max_ns", 6000)),
            int(kw.get("steps", 128)),
        )
    elif pattern == "image":
        # Silkscreen: map one image row to a revolution (radial-style)
        return gen_image_intervals(
            rev_time_ns=rev_time_ns,
            image_path=str(kw.get("image")),
            long_ns=int(kw.get("long_ns", 4200)),
            short_ns=int(kw.get("short_ns", 2200)),
            row_fraction=float(kw.get("row_fraction", 0.0)),
            threshold=int(kw.get("threshold", 140)),
        )
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
        for _ in range(int(kw.get("interval_count", 512))):
            # taps x^7 + x^6 + 1 => feedback = bit6 XOR bit5 of 7-bit register
            fb = ((state >> 6) & 1) ^ ((state >> 5) & 1)
            state = ((state << 1) & 0x7E) | fb
            bit = state & 1
            ns = max(50, long_ns if bit == 0 else short_ns)
            intervals.append(int(ns))
        # Normalize last to hit target time
        total = sum(intervals)
        if intervals:
            intervals[-1] = max(50, intervals[-1] + (rev_time_ns - total))
        return intervals
    else:
        raise ValueError(f"Unknown pattern: {pattern}")


# -----------------------------
# Sanitizer: clamp/split intervals
# -----------------------------
def sanitize_intervals(
    intervals: List[int],
    rev_time_ns: int,
    min_ns: int = 2000,
    max_ns: int = 65_000_000,
    keepalive_ns: int = 8_000_000,
) -> List[int]:
    """
    Pre-sanitize flux intervals to avoid hardware-hostile values.
    - Clamp too-short cells up to min_ns.
    - Split too-long cells into multiple sub-intervals so no single gap exceeds keepalive_ns (preferred) or max_ns.
    - Preserve total revolution time by adjusting the final interval.
    """
    if min_ns < 50:
        min_ns = 50
    if max_ns < min_ns:
        max_ns = min_ns
    if keepalive_ns <= 0:
        keepalive_ns = max_ns
    out: List[int] = []
    for v in intervals:
        # Enforce keepalive: split massive gaps into chunks <= keepalive_ns
        if v > keepalive_ns:
            rem = int(v)
            chunk = int(keepalive_ns)
            while rem > keepalive_ns:
                out.append(chunk)
                rem -= chunk
            if rem > 0:
                out.append(rem)
        else:
            out.append(int(v))
    # Clamp to [min_ns, max_ns]
    out = [max(min_ns, min(int(v), max_ns)) for v in out]
    # Adjust last to hit target rev_time_ns exactly
    total = sum(out)
    if out:
        out[-1] = max(min_ns, min(max_ns, out[-1] + (rev_time_ns - total)))
    return out


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
    ap.add_argument("--writer", choices=["finalized", "dtc", "strict-dtc"], default="finalized", help="Stream writer: 'finalized' (HxC default), 'dtc' (DTC-style header), or 'strict-dtc' (byte-compat DTC, OOB-first)")
    # Initial index handling: default is OFF for compatibility
    ap.add_argument("--initial-index", action="store_true", help="Emit an initial OOB index marker at start (default off)")
    ap.add_argument("--no-initial-index", action="store_true", help="DEPRECATED: same as default; prefer --initial-index to enable")
    # KFInfo toggles
    ap.add_argument("--no-kf-version-info", action="store_true", help="Do not emit KFInfo 'KryoFlux stream - version ...' string")
    ap.add_argument("--no-clock-info", action="store_true", help="DEPRECATED: Clocks are disabled by default; use --clock-info to enable")
    ap.add_argument("--clock-info", action="store_true", help="Emit KFInfo 'sck=..., ick=...' string (off by default)")
    ap.add_argument("--hw-info", action="store_true", help="Emit extended HW info KFInfo (host_date, name, hwid, etc.) (off by default)")
    ap.add_argument("--ick-hz", type=float, default=3003428.5714, help="Index clock frequency (Hz) for OOB counters (default ~3.003 MHz)")
    ap.add_argument("--pattern", choices=["constant", "random", "alt", "zeros", "ones", "sweep", "prbs7", "image"], default="constant")
    ap.add_argument("--interval-ns", type=int, default=4000, help="Constant/ones cell interval (ns)")
    ap.add_argument("--mean-ns", type=float, default=4000.0, help="Random pattern mean interval (ns)")
    ap.add_argument("--std-ns", type=float, default=400.0, help="Random pattern stddev interval (ns)")
    ap.add_argument("--long-ns", type=int, default=4000, help="Long cell interval for alt/zeros/prbs7 (ns)")
    ap.add_argument("--short-ns", type=int, default=2000, help="Short cell interval for alt/ones/prbs7 (ns)")
    ap.add_argument("--sweep-min-ns", type=int, default=2000, help="Sweep minimum interval (ns)")
    ap.add_argument("--sweep-max-ns", type=int, default=6000, help="Sweep maximum interval (ns)")
    ap.add_argument("--steps", type=int, default=128, help="Sweep/alt segments (default 128)")
    ap.add_argument("--seed", type=int, default=0, help="Seed for PRBS/random generation (0 = unseeded)")
    # Image/silkscreen options
    ap.add_argument("--image", help="Path to input image for --pattern image")
    ap.add_argument("--threshold", type=int, default=140, help="Binarization threshold for image pattern (0..255)")
    # Profile preset (shortcuts for common configurations)
    ap.add_argument("--profile", help="Preset profile name to avoid long CLI chains")
    # Sanitizer toggles
    ap.add_argument("--sanitize", action="store_true", help="Enable flux sanitizer: clamp min, split long gaps (keepalive), cap max")
    ap.add_argument("--sanitize-min-ns", type=int, default=2000, help="Minimum interval (ns) after clamping (default 2000 ns ~ 2us)")
    ap.add_argument("--sanitize-keepalive-ns", type=int, default=8_000_000, help="Maximum single gap (ns) before splitting into chunks (default 8 ms)")
    ap.add_argument("--sanitize-max-ns", type=int, default=65_000_000, help="Absolute cap for any single interval (ns) (default 65 ms)")
    ap.add_argument("--output-dir", required=True, help="Directory to write NN.S.raw set")
    args = ap.parse_args(argv)

    # Built-in profiles to minimize CLI typing
    PROFILES: Dict[str, Dict[str, Any]] = {
        # Simple baselines
        "constant-4us": {"pattern": "constant", "interval_ns": 4000, "revs": 3, "writer": "strict-dtc"},
        "alt-4us-2us": {"pattern": "alt", "long_ns": 4000, "short_ns": 2000, "steps": 128, "revs": 3, "writer": "strict-dtc"},
        "sweep-2to6us": {"pattern": "sweep", "sweep_min_ns": 2000, "sweep_max_ns": 6000, "steps": 128, "revs": 3, "writer": "strict-dtc"},
        "prbs7-default": {"pattern": "prbs7", "long_ns": 4200, "short_ns": 2200, "revs": 3, "writer": "strict-dtc"},
        # Image/silkscreen (radial)
        "image-radial": {"pattern": "image", "threshold": 140, "long_ns": 4200, "short_ns": 2200, "revs": 3, "writer": "strict-dtc"},
        # Conservative, hardware-friendly defaults for USB controllers
        "safe-usb": {
            "pattern": "constant",
            "interval_ns": 4200,
            "long_ns": 4200,
            "short_ns": 2200,
            "revs": 3,
            "writer": "strict-dtc",
            "sanitize": True,
            "sanitize_min_ns": 2000,
            "sanitize_keepalive_ns": 8_000_000,
            "sanitize_max_ns": 65_000_000,
        },
    }

    def apply_profile(a) -> None:
        if not a.profile:
            return
        p = PROFILES.get(a.profile.strip().lower())
        if not p:
            print(f"Unknown profile: {a.profile}", file=sys.stderr)
            return
        # Apply profile values if the user did not explicitly override them
        for k, v in p.items():
            if not hasattr(a, k):
                continue
            cur = getattr(a, k)
            # Treat falsy defaults as unset when profile supplies a value
            if cur in (None, 0, False, ""):
                setattr(a, k, v)
        # Always apply pattern & writer from profile
        setattr(a, "pattern", p.get("pattern", a.pattern))
        setattr(a, "writer", p.get("writer", a.writer))

    apply_profile(args)

    tracks = parse_range(args.tracks)
    if not tracks:
        print("No tracks parsed from --tracks", file=sys.stderr)
        return 2
    sides = [int(s.strip()) for s in args.sides.split(',') if s.strip() != '']
    revs = max(1, int(args.revs))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_tracks = len(tracks)
    for idx_t, t in enumerate(tracks):
        for s in sides:
            # Generate intervals per revolution and concatenate across revs
            intervals: List[int] = []
            rev_lens: List[int] = []
            for _ in range(revs):
                # Pass track-relative progress for image mapping (0..1)
                row_fraction = (idx_t / (total_tracks - 1)) if total_tracks > 1 else 0.0
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
                    steps=int(args.steps),
                    image=args.image,
                    threshold=int(args.threshold),
                    row_fraction=float(row_fraction),
                )
                # Optional sanitizer to make flux hardware-friendly
                if bool(getattr(args, "sanitize", False)):
                    part = sanitize_intervals(
                        part,
                        rev_time_ns=int(args.rev_time_ns),
                        min_ns=int(args.sanitize_min_ns),
                        max_ns=int(args.sanitize_max_ns),
                        keepalive_ns=int(args.sanitize_keepalive_ns),
                    )
                intervals.extend(part)
                rev_lens.append(len(part))
            # Write trackNN.S.raw (provide exact per-rev lengths to align OOB indices)
            fname = f"track{t:02d}.{int(s)}.raw"
            fpath = out_dir / fname
            if args.writer == "dtc":
                # Lazy-import DTC-style writer to avoid hard dependency when not used
                sed_mod = importlib.import_module("stream_export_dtc")
                write_kryoflux_stream_dtc = getattr(sed_mod, "write_kryoflux_stream_dtc")
                write_kryoflux_stream_dtc(
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
                    include_clock_info=True if not getattr(args, "no_clock_info", False) else False,
                    include_hw_info=True if getattr(args, "hw_info", False) or True else True,
                )
            elif args.writer == "strict-dtc":
                # Strict DTC writer: OOB-first enforced, KFInfo/clock/hw included by default
                sds_mod = importlib.import_module("stream_export_dtc_strict")
                write_kryoflux_stream_dtc_strict = getattr(sds_mod, "write_kryoflux_stream_dtc_strict")
                write_kryoflux_stream_dtc_strict(
                    intervals,
                    track=int(t),
                    side=int(s),
                    output_path=str(fpath),
                    num_revs=revs,
                    version=str(args.version_string),
                    sck_hz=float(args.sck_hz),
                    rev_lengths=rev_lens,
                    header_mode="oob",  # forced inside strict writer as well
                    include_sck_oob=(not args.no_sck_oob),
                    include_initial_index=(True if args.initial_index and not args.no_initial_index else False),
                    ick_hz=float(args.ick_hz),
                )
                # Also write a duplicate NN.S.raw for maximum DTC compatibility across builds
                dup_name = f"{int(t):02d}.{int(s)}.raw"
                dup_path = out_dir / dup_name
                try:
                    shutil.copyfile(fpath, dup_path)
                except Exception:
                    # Fallback: read/write if copyfile not possible across FS
                    data = Path(fpath).read_bytes()
                    dup_path.write_bytes(data)
            else:
                se_mod = importlib.import_module("stream_export")
                write_kryoflux_stream = getattr(se_mod, "write_kryoflux_stream")
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
