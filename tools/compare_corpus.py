#!/usr/bin/env python3
"""
compare_corpus.py

Compare multiple passes of captured streams to quantify stability across passes.
Designed to work with RUN_DIR layouts that contain subdirectories like:
  pass_01_side_0/, pass_01_side_1/, pass_02_side_0/, ... each with trackNN.S.raw

Usage:
  python3 FloppyAI/tools/compare_corpus.py --root <RUN_DIR> [--json out.json]

Output:
  - Console summary per track/side across passes
  - Optional JSON with per-key metrics and deltas vs baseline pass
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Ensure repo root on sys.path
import sys
HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[2]
SRC_DIR = REPO_ROOT / "FloppyAI" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from flux_analyzer import FluxAnalyzer  # type: ignore


def find_pass_dirs(root: Path) -> List[Path]:
    dirs = []
    for p in sorted(root.iterdir()):
        if p.is_dir() and p.name.startswith("pass_"):
            dirs.append(p)
    return dirs


def scan_tracks(pass_dir: Path) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for p in pass_dir.rglob("track*.raw"):
        name = p.name  # trackNN.S.raw
        try:
            core = name.replace("track", "").replace(".raw", "")  # NN.S
            t_str, s_str = core.split(".")
            t = int(t_str)
            s = int(s_str)
            key = f"t{t:02d}.s{s}"
            mapping[key] = p
        except Exception:
            continue
    return mapping


def analyze_file(path: Path) -> Dict[str, Any]:
    fa = FluxAnalyzer()
    parsed = fa.parse(str(path))
    stats = parsed.get("stats", {}) or {}
    return {
        "path": str(path),
        "total_fluxes": int(stats.get("total_fluxes") or 0),
        "num_revolutions": int(stats.get("num_revolutions") or 0),
        "inferred_rpm": float(stats.get("inferred_rpm") or 0.0),
        "mean_interval_ns": float(stats.get("mean_interval_ns") or 0.0),
        "std_interval_ns": float(stats.get("std_interval_ns") or 0.0),
    }


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description="Compare multiple capture passes across track/side")
    ap.add_argument("--root", required=True, help="Run directory containing pass_* subdirs")
    ap.add_argument("--json", default="", help="Optional path to write JSON summary")
    args = ap.parse_args(argv)

    root = Path(args.root)
    if not root.is_dir():
        print(f"Root not found: {root}")
        return 2

    pass_dirs = find_pass_dirs(root)
    if len(pass_dirs) < 2:
        print("Need at least two pass_* directories to compare.")
        return 1

    # Build key -> list of (pass_name, path)
    corpus: Dict[str, List[Tuple[str, Path]]] = {}
    for d in pass_dirs:
        tracks = scan_tracks(d)
        for key, p in tracks.items():
            corpus.setdefault(key, []).append((d.name, p))

    if not corpus:
        print("No track*.raw files found in pass_* directories.")
        return 1

    # Analyze and compare vs baseline (first pass by sorted order)
    report: Dict[str, Any] = {}
    print("Corpus comparison (baseline = first pass)\n")
    for key in sorted(corpus.keys()):
        entries = sorted(corpus[key], key=lambda x: x[0])
        if len(entries) < 2:
            continue
        baseline_name, baseline_path = entries[0]
        base_metrics = analyze_file(baseline_path)
        lines = [f"{key}:"]
        lines.append(f"  baseline {baseline_name}: rpm={base_metrics['inferred_rpm']:.2f}, revs={base_metrics['num_revolutions']}, mean={base_metrics['mean_interval_ns']:.2f}ns, fluxes={base_metrics['total_fluxes']}")
        comp_rows = []
        for name, path in entries[1:]:
            m = analyze_file(path)
            drpm = m['inferred_rpm'] - base_metrics['inferred_rpm']
            dmean = m['mean_interval_ns'] - base_metrics['mean_interval_ns']
            dflux = m['total_fluxes'] - base_metrics['total_fluxes']
            lines.append(f"  {name}: rpm={m['inferred_rpm']:.2f} (Δ{drpm:+.2f}), mean={m['mean_interval_ns']:.2f}ns (Δ{dmean:+.2f}), fluxes={m['total_fluxes']} (Δ{dflux:+d})")
            comp_rows.append({
                "pass": name,
                "metrics": m,
                "delta_vs_baseline": {
                    "rpm": drpm,
                    "mean_interval_ns": dmean,
                    "total_fluxes": dflux,
                }
            })
        print("\n".join(lines))
        report[key] = {
            "baseline": {"pass": baseline_name, "metrics": base_metrics},
            "comparisons": comp_rows,
        }

    if args.json:
        try:
            outp = Path(args.json)
            outp.parent.mkdir(parents=True, exist_ok=True)
            with open(outp, 'w') as fp:
                json.dump(report, fp, indent=2)
            print(f"\nJSON summary written to {outp}")
        except Exception as e:
            print(f"WARN: failed to write JSON: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
