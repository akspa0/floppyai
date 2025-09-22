#!/usr/bin/env python3
"""
analyze_captures.py

Batch-analyze a folder of KryoFlux stream files using FluxAnalyzer.
Outputs a concise console report and writes a JSON summary next to the data.

Usage:
  python3 FloppyAI/tools/analyze_captures.py --dir <path> [--glob 'track*.raw']
  python3 FloppyAI/tools/analyze_captures.py --files file1.raw file2.raw

The JSON summary is saved as <dir>/analysis_summary.json when --dir is used.
"""
from __future__ import annotations

import argparse
import json
import glob
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

# Ensure repo root on sys.path
HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[2]
SRC_DIR = REPO_ROOT / "FloppyAI" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from flux_analyzer import FluxAnalyzer  # type: ignore


def analyze_one(path: str) -> Dict[str, Any]:
    fa = FluxAnalyzer()
    parsed = fa.parse(path)
    stats = parsed.get('stats', {}) or {}
    return {
        'file': path,
        'total_fluxes': int(stats.get('total_fluxes') or 0),
        'num_revolutions': int(stats.get('num_revolutions') or 0),
        'inferred_rpm': float(stats.get('inferred_rpm') or 0.0),
        'decoder_sck_hz': float(stats.get('decoder_sck_hz')) if 'decoder_sck_hz' in stats else None,
        'decoder_oob_index_count': int(stats.get('decoder_oob_index_count')) if 'decoder_oob_index_count' in stats else None,
        'decoder_total_samples': int(stats.get('decoder_total_samples')) if 'decoder_total_samples' in stats else None,
        'mean_interval_ns': float(stats.get('mean_interval_ns') or 0.0),
        'std_interval_ns': float(stats.get('std_interval_ns') or 0.0),
    }


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description="Analyze KryoFlux stream files in bulk")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--dir', dest='dir', help='Directory to scan')
    g.add_argument('--files', nargs='*', help='Explicit files')
    ap.add_argument('--glob', default='*.raw', help="Glob pattern within --dir (default: *.raw)")
    args = ap.parse_args(argv)

    files: List[str]
    out_dir: Path | None = None

    if args.dir:
        out_dir = Path(args.dir)
        files = sorted(glob.glob(str(out_dir / args.glob)))
    else:
        files = args.files or []

    if not files:
        print("No files to analyze.")
        return 2

    rows: List[Dict[str, Any]] = []
    for f in files:
        try:
            row = analyze_one(f)
            rows.append(row)
            print(f"- {os.path.basename(f)}: revs={row['num_revolutions']}, rpm={row['inferred_rpm']:.2f}, fluxes={row['total_fluxes']}")
        except Exception as e:
            print(f"! {f}: ERROR {e}")

    # Emit JSON summary if analyzing a directory
    if out_dir:
        summary_path = out_dir / 'analysis_summary.json'
        try:
            with open(summary_path, 'w') as fp:
                json.dump({'files': rows}, fp, indent=2)
            print(f"Summary written to {summary_path}")
        except Exception as e:
            print(f"WARN: failed to write summary: {e}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
