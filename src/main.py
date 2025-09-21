#!/usr/bin/env python3
import datetime
import os
import sys
from pathlib import Path
# Ensure local sibling imports work whether run as a script or with `-m`
sys.path.insert(0, str(Path(__file__).parent))
"""
FloppyAI CLI Tool
Main entrypoint for analyzing, reading, writing, and generating flux streams.
Usage: python -m FloppyAI.src.main <command> [args]
"""

import argparse
import sys
from pathlib import Path
import os

from flux_analyzer import FluxAnalyzer
from overlay_detection import (
    detect_side_overlay_mfm,
    detect_side_overlay_gcr,
)
from rendering import (
    render_disk_surface,
    render_instability_map,
)
from dtc_wrapper import DTCWrapper
from custom_encoder import CustomEncoder
from custom_decoder import CustomDecoder
from cmd_diff import compare_reads as compare_reads_cmd
from cmd_corpus import analyze_corpus as analyze_corpus_cmd
from cmd_stream_ops import (
    analyze_stream as analyze_stream_cmd,
    read_track as read_track_cmd,
    write_track as write_track_cmd,
    generate_dummy as generate_dummy_cmd,
    encode_data as encode_data_cmd,
)
from patterns import generate_pattern
from stream_export import write_internal_raw, write_kryoflux_stream
from encoding.image2flux import generate_from_image
from encoding.silkscreen import generate_silkscreen
from analysis.structure_finder import recover_image as recover_image_func
from encoding.pattern_images import generate_polar_pattern, save_polar_png
try:
    from cmd_experiments import run_experiment_matrix as run_experiment_matrix_cmd
except ImportError:
    run_experiment_matrix_cmd = None
try:
    from analysis.analyze_disk import run as analyze_disk_cmd
except Exception:
    analyze_disk_cmd = None
import glob
import json
import re
import numpy as np
import matplotlib.pyplot as plt
import json
import openai
from utils.json_io import dump_json

def _json_default(o):
    try:
        import numpy as _np
        if isinstance(o, (_np.integer,)):
            return int(o)
        if isinstance(o, (_np.floating,)):
            return float(o)
        if isinstance(o, (_np.ndarray,)):
            return o.tolist()
    except Exception:
        pass
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, set):
        return list(o)
    # Fallback stringification to avoid crashes
    return str(o)

def get_output_dir(output_dir=None):
    """Resolve output directory.

    - If output_dir is None: create test_outputs/<timestamp>/
    - If output_dir is provided: use it directly (no timestamp subfolder)
    """
    if output_dir is None:
        base_dir = Path("test_outputs")
        base_dir.mkdir(exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = base_dir / timestamp
        run_dir.mkdir(exist_ok=True)
        return run_dir
    else:
        run_dir = Path(output_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

def log_output(func):
    """Decorator to log console output to file."""
    def wrapper(*args, **kwargs):
        run_dir = get_output_dir()
        log_file = run_dir / "run.log"
        original_stdout = sys.stdout
        with open(log_file, 'w') as f:
            sys.stdout = f
            try:
                return func(*args, **kwargs)
            finally:
                sys.stdout = original_stdout
        print(f"Log saved to {log_file}")
    return wrapper

@log_output
def analyze_stream(args):
    """Analyze a .raw stream file."""
    run_dir = get_output_dir(args.output_dir)
    analyzer = FluxAnalyzer()
    try:
        parsed = analyzer.parse(args.input)
        print("Parsed Data:")
        stats = parsed.get('stats', {})
        print(f"  Total Fluxes: {stats.get('total_fluxes', 0)}")
        print(f"  Mean Interval: {stats.get('mean_interval_ns', 0):.2f} ns")
        print(f"  Std Dev: {stats.get('std_interval_ns', 0):.2f} ns")
        print(f"  Num Revolutions: {stats.get('num_revolutions', 0)}")
        if not stats:
            print("  No flux data - possibly unformatted blank or parsing issue.")
        
        try:
            analysis = analyzer.analyze()
            print("\nAnalysis:")
            anomalies = analysis.get('anomalies', {})
            noise_profile = analysis.get('noise_profile', {})
            print(f"  Short Cells: {len(anomalies.get('short_cells', []))}")
            print(f"  Long Intervals: {len(anomalies.get('long_intervals', []))}")
            print(f"  Avg Variance (Noise): {noise_profile.get('avg_variance', 0):.2f}")
            print(f"  Density Estimate: {analysis.get('density_estimate_bits_per_rev', 0)} bits/rev")
        except Exception as e:
            print(f"  Analysis error: {e}")
            print("  No analysis - check flux data.")
        
        # Generate visualizations in run_dir
        base_name = str(run_dir / Path(args.input).stem)
        analyzer.visualize(base_name, "intervals")
        analyzer.visualize(base_name, "histogram")
        if len(analyzer.revolutions) > 1:
            analyzer.visualize(base_name, "heatmap")
        
        print(f"\nVisualizations saved in {run_dir} with prefix '{Path(args.input).stem}_'.")
    except Exception as e:
        print(f"Error analyzing {args.input}: {e}")
        return 1
    return 0

def _profile_safe_max(profile: str | None) -> int:
    if not profile:
        return 80
    if profile.startswith('35'):
        return 80
    if profile.startswith('525'):
        return 81
    return 80

def _parse_tracks(tracks_arg: str | None, default_max: int) -> list[int]:
    if not tracks_arg:
        return list(range(0, default_max + 1))
    s = str(tracks_arg).strip()
    if '-' in s and ',' not in s:
        a, b = s.split('-', 1)
        start = int(a)
        end = int(b)
        if start > end:
            start, end = end, start
        return list(range(start, end + 1))
    # comma list
    parts = [p.strip() for p in s.split(',') if p.strip()]
    return sorted({int(p) for p in parts})

def _parse_sides(sides_arg: str | None) -> list[int]:
    if not sides_arg:
        return [0, 1]
    parts = [p.strip() for p in str(sides_arg).split(',') if p.strip()]
    sides = [int(p) for p in parts]
    return sorted({s for s in sides if s in (0, 1)})

def generate_disk(args):
    """Generate a full-disk set of .raw streams (NN.S.raw) for DTC write flows."""
    run_dir = get_output_dir(args.output_dir)
    disk_dir = run_dir / 'disk_image'
    disk_dir.mkdir(parents=True, exist_ok=True)

    profile = getattr(args, 'profile', None)
    safe_max = _profile_safe_max(profile)
    tracks = _parse_tracks(getattr(args, 'tracks', None), safe_max)
    sides = _parse_sides(getattr(args, 'sides', None))

    # Enforce safe limits unless --allow-extended
    max_req = max(tracks) if tracks else 0
    if not getattr(args, 'allow_extended', False) and max_req > safe_max:
        print(f"Requested max track {max_req} exceeds safe limit {safe_max} for profile {profile or 'default'}.")
        print("Pass --allow-extended to override (not recommended).")
        return 2

    density = getattr(args, 'density', 1.0) or 1.0
    base_cell_ns = float(args.cell_length) / float(density)
    rpm = float(getattr(args, 'rpm', 360.0) or 360.0)
    pattern = getattr(args, 'pattern', 'random')
    seed = getattr(args, 'seed', None)
    out_fmt = getattr(args, 'output_format', 'kryoflux')
    revs = int(getattr(args, 'revolutions', 1) or 1)

    files = []
    for t in tracks:
        for s in sides:
            # vary seed per file for uniqueness if seed provided
            kwargs = {}
            use_seed = None if seed is None else int(seed) + (t * 2 + s)
            if use_seed is not None:
                kwargs['seed'] = use_seed
            intervals = generate_pattern(
                name=pattern,
                revolutions=revs,
                base_cell_ns=base_cell_ns,
                rpm=rpm,
                **kwargs,
            )
            fname = f"{t:02d}.{s}.raw"
            out_path = str(disk_dir / fname)
            if out_fmt == 'internal':
                write_internal_raw(intervals, t, s, out_path, num_revs=revs)
            else:
                write_kryoflux_stream(intervals, t, s, out_path, num_revs=revs, rpm=rpm)
            files.append({'track': t, 'side': s, 'file': out_path})

    manifest = {
        'profile': profile,
        'safe_max': safe_max,
        'requested_max': max_req,
        'pattern': pattern,
        'density': density,
        'rpm': rpm,
        'cell_length': args.cell_length,
        'revolutions': revs,
        'output_format': out_fmt,
        'tracks': tracks,
        'sides': sides,
        'files': files,
        'disk_dir': str(disk_dir),
    }
    dump_json(disk_dir / 'disk_image_manifest.json', manifest)
    print(f"Generated full-disk set to {disk_dir} with {len(files)} files.")
    print("On Linux DTC host, write with:")
    print("  FloppyAI/scripts/linux/dtc_write_read_set.sh --image-dir /path/to/disk_image --drive 0 --revs 3")
    return 0

def analyze_corpus(args):
    """Aggregate multiple surface_map.json files to produce a corpus-level summary."""
    run_dir = get_output_dir(args.output_dir)
    inputs = []
    base = Path(args.inputs)
    # Ensure dedicated corpus folder for all corpus-level outputs
    corpus_dir = run_dir / 'corpus'
    corpus_dir.mkdir(parents=True, exist_ok=True)

    # Resolve effective RPM from profile or explicit value
    rpm_profile_map = {
        '35HD': 300.0,      # 3.5" 1.44MB (MFM)
        '35DD': 300.0,      # 3.5" 720KB (MFM)
        '35HDGCR': 300.0,   # 3.5" 800K (Apple GCR variable data rate)
        '35DDGCR': 300.0,   # 3.5" 400K (Apple GCR variable data rate)
        '525HD': 360.0,     # 5.25" 1.2MB (MFM)
        '525DD': 300.0,     # 5.25" 360KB (MFM)
        '525DDGCR': 300.0,  # 5.25" Apple II 140/280KB (GCR)
    }
    try:
        profile = getattr(args, 'profile', None)
    except Exception:
        profile = None
    effective_rpm = (
        float(args.rpm) if getattr(args, 'rpm', None) is not None else rpm_profile_map.get(profile, 360.0)
    )

    # Optionally generate surface_map.json for directories containing .raw files
    if getattr(args, 'generate_missing', False) and base.is_dir():
        raw_dirs = {p.parent for p in base.rglob('*.raw')}
        missing_maps = []
        for d in sorted(raw_dirs):
            try:
                print(f"Generating surface map for {d} ...")
                label = Path(d).name
                safe = re.sub(r'[^A-Za-z0-9_.-]', '_', label)
                disk_dir = (run_dir / 'disks' / safe)
                disk_dir.mkdir(parents=True, exist_ok=True)
                # Invoke analyze_disk programmatically, directing outputs to disk_dir
                disk_args = argparse.Namespace(
                    input=str(d), track=None, side=None, rpm=effective_rpm,
                    lm_host=getattr(args, 'lm_host', 'localhost:1234'),
                    lm_model=getattr(args, 'lm_model', 'local-model'),
                    lm_temperature=getattr(args, 'lm_temperature', 0.2),
                    summarize=getattr(args, 'summarize', False),
                    output_dir=str(disk_dir), summary_format='json',
                    media_type=getattr(args, 'media_type', None),
                    # Overlay flags propagated from corpus args
                    format_overlay=getattr(args, 'format_overlay', False),
                    angular_bins=getattr(args, 'angular_bins', 0),
                    overlay_alpha=getattr(args, 'overlay_alpha', 0.8),
                    overlay_color=getattr(args, 'overlay_color', '#ff3333'),
                    overlay_mode=getattr(args, 'overlay_mode', 'mfm'),
                    gcr_candidates=getattr(args, 'gcr_candidates', '10,12,8,9,11,13'),
                    overlay_sectors_hint=getattr(args, 'overlay_sectors_hint', None),
                    export_format='png',
                    align_to_sectors=getattr(args, 'align_to_sectors', 'off'),
                    label_sectors=getattr(args, 'label_sectors', False),
                )
                analyze_disk(disk_args)
                # Verify that surface_map.json was produced; if missing, track it
                smap = disk_dir / 'surface_map.json'
                if not smap.exists():
                    print(f"Warning: surface_map.json missing for {d}; skipping in corpus aggregation")
                    missing_maps.append(str(smap))
                # Copy/rename composite to a concise name per disk
                comp_src = disk_dir / f"{safe}_composite_report.png"
                comp_dst = disk_dir / f"{safe}_composite.png"
                if comp_src.exists():
                    try:
                        import shutil
                        shutil.copyfile(str(comp_src), str(comp_dst))
                    except Exception as e:
                        print(f"Warning: failed to copy composite for {label}: {e}")
                # Copy/rename polar disk-surface image
                polar_src = disk_dir / f"{safe}_surface_disk_surface.png"
                polar_dst = disk_dir / f"{safe}_disk_surface.png"
                if polar_src.exists():
                    try:
                        import shutil
                        shutil.copyfile(str(polar_src), str(polar_dst))
                    except Exception as e:
                        print(f"Warning: failed to copy polar surface for {label}: {e}")
            except Exception as e:
                print(f"Failed to analyze {d}: {e}")
        # Persist missing inputs list for transparency
        try:
            if missing_maps:
                with open(corpus_dir / 'corpus_missing_inputs.txt', 'w') as mf:
                    mf.write("The following per-disk runs did not produce surface_map.json (skipped):\n")
                    for p in missing_maps:
                        mf.write(str(p) + "\n")
        except Exception:
            pass

    if base.is_dir():
        found = {p for p in base.rglob('surface_map.json')}
        # Include newly generated runs under test_outputs as well
        test_out = Path('test_outputs')
        if test_out.exists():
            found.update(test_out.rglob('surface_map.json'))
        # Also include any surface_map.json created in this run directory
        try:
            found.update(run_dir.rglob('surface_map.json'))
        except Exception:
            pass
        inputs = sorted(found)
        # Persist a manifest of inputs for transparency
        try:
            corpus_dir.mkdir(parents=True, exist_ok=True)
            with open(corpus_dir / 'corpus_inputs.txt', 'w') as mf:
                if inputs:
                    mf.write("Found the following surface_map.json files:\n")
                    for p in inputs:
                        mf.write(str(p) + "\n")
                else:
                    mf.write("No surface_map.json files found under inputs/test_outputs/run_dir searches.\n")
        except Exception:
            pass
    elif base.name.endswith('.json'):
        inputs = [base]
    if not inputs:
        print(f"No surface_map.json found under {args.inputs}")
        try:
            with open(corpus_dir / 'corpus_no_inputs.txt', 'w') as nf:
                nf.write(f"No surface_map.json found under: {args.inputs}\n")
                nf.write(f"Run directory: {run_dir}\n")
        except Exception:
            pass
        return 1

    corpus = []
    for mp in inputs:
        try:
            with open(mp, 'r') as f:
                sm = json.load(f)
            corpus.append((mp, sm))
            print(f"Loaded {mp}")
        except Exception as e:
            print(f"Failed to load {mp}: {e}")

    per_disk = []
    all_side0 = []
    all_side1 = []
    all_side0_var = []
    all_side1_var = []
    plot_side0_by_disk = []  # list of lists (densities)
    plot_side1_by_disk = []
    labels_side0 = []  # legend labels derived from stream filenames
    labels_side1 = []
    per_disk_tracks_side0 = []  # list of (label, [(track, dens)])
    per_disk_tracks_side1 = []
    disk_labels = []
    def side_entries(track_obj, side_int):
        """Return list for side key handling both int and string keys from JSON."""
        if not isinstance(track_obj, dict):
            return []
        if side_int in track_obj:
            return track_obj.get(side_int, [])
        return track_obj.get(str(side_int), [])

    for mp, sm in corpus:
        s0 = []
        s1 = []
        v0 = []
        v1 = []
        per_disk_t0 = []
        per_disk_t1 = []
        rep_file_s0 = None
        rep_file_s1 = None
        for track in sm:
            if track == 'global':
                continue
            for entry in side_entries(sm[track], 0):
                if isinstance(entry, dict):
                    d = entry.get('analysis', {}).get('density_estimate_bits_per_rev')
                    var = entry.get('analysis', {}).get('noise_profile', {}).get('avg_variance') if isinstance(entry.get('analysis', {}), dict) else None
                    if rep_file_s0 is None:
                        rep_file_s0 = entry.get('file')
                    if isinstance(d, (int, float)):
                        s0.append(float(d))
                        all_side0.append(float(d))
                        try:
                            per_disk_t0.append((int(track), float(d)))
                        except Exception:
                            pass
                    if isinstance(var, (int, float)):
                        v0.append(float(var))
                        all_side0_var.append(float(var))
            for entry in side_entries(sm[track], 1):
                if isinstance(entry, dict):
                    d = entry.get('analysis', {}).get('density_estimate_bits_per_rev')
                    var = entry.get('analysis', {}).get('noise_profile', {}).get('avg_variance') if isinstance(entry.get('analysis', {}), dict) else None
                    if rep_file_s1 is None:
                        rep_file_s1 = entry.get('file')
                    if isinstance(d, (int, float)):
                        s1.append(float(d))
                        all_side1.append(float(d))
                        try:
                            per_disk_t1.append((int(track), float(d)))
                        except Exception:
                            pass
                    if isinstance(var, (int, float)):
                        v1.append(float(var))
                        all_side1_var.append(float(var))
        plot_side0_by_disk.append(s0)
        plot_side1_by_disk.append(s1)
        # Prefer representative stream filename as the disk label for user-facing plots
        rep_name_s0 = Path(rep_file_s0).name if isinstance(rep_file_s0, str) else None
        rep_name_s1 = Path(rep_file_s1).name if isinstance(rep_file_s1, str) else None
        preferred_label = rep_name_s0 or rep_name_s1 or Path(mp).parent.name
        disk_labels.append(preferred_label)
        # Legend labels prefer representative stream filename; fallback to preferred_label
        labels_side0.append(rep_name_s0 or preferred_label)
        labels_side1.append(rep_name_s1 or preferred_label)
        per_disk_tracks_side0.append((disk_labels[-1], per_disk_t0))
        per_disk_tracks_side1.append((disk_labels[-1], per_disk_t1))
        disk_stats = {
            'path': str(mp),
            'label': preferred_label,
            'side0': {
                'count': len(s0),
                'min': float(np.min(s0)) if s0 else None,
                'max': float(np.max(s0)) if s0 else None,
                'avg': float(np.mean(s0)) if s0 else None,
                'median': float(np.median(s0)) if s0 else None,
                'avg_variance': float(np.mean(v0)) if v0 else None,
            },
            'side1': {
                'count': len(s1),
                'min': float(np.min(s1)) if s1 else None,
                'max': float(np.max(s1)) if s1 else None,
                'avg': float(np.mean(s1)) if s1 else None,
                'median': float(np.median(s1)) if s1 else None,
                'avg_variance': float(np.mean(v1)) if v1 else None,
            }
        }
        per_disk.append(disk_stats)

    corpus_summary = {
        'num_disks': len(per_disk),
        'side0': {
            'count': len(all_side0),
            'min': float(np.min(all_side0)) if all_side0 else None,
            'max': float(np.max(all_side0)) if all_side0 else None,
            'avg': float(np.mean(all_side0)) if all_side0 else None,
            'median': float(np.median(all_side0)) if all_side0 else None,
        },
        'side1': {
            'count': len(all_side1),
            'min': float(np.min(all_side1)) if all_side1 else None,
            'max': float(np.max(all_side1)) if all_side1 else None,
            'avg': float(np.mean(all_side1)) if all_side1 else None,
            'median': float(np.median(all_side1)) if all_side1 else None,
        },
        'per_disk': per_disk,
    }

    out_path = corpus_dir / 'corpus_summary.json'
    dump_json(out_path, corpus_summary)
    print(f"Corpus summary saved to {out_path}")

    # Visualizations for corpus
    try:
        # Histograms of density per side
        if all_side0:
            plt.figure(figsize=(8,4))
            plt.hist(all_side0, bins=40, color='steelblue', alpha=0.8)
            plt.title('Density Distribution - Side 0 (all disks)')
            plt.xlabel('Bits per Revolution')
            plt.ylabel('Count')
            plt.tight_layout()
            plt.savefig(str(corpus_dir / 'corpus_side0_density_hist.png'), dpi=150)
            plt.close()
        if all_side1:
            plt.figure(figsize=(8,4))
            plt.hist(all_side1, bins=40, color='indianred', alpha=0.8)
            plt.title('Density Distribution - Side 1 (all disks)')
            plt.xlabel('Bits per Revolution')
            plt.ylabel('Count')
            plt.tight_layout()
            plt.savefig(str(corpus_dir / 'corpus_side1_density_hist.png'), dpi=150)
            plt.close()

        # Boxplots of density by disk for each side
        if any(len(x) for x in plot_side0_by_disk):
            plt.figure(figsize=(max(8, len(plot_side0_by_disk)*0.6), 5))
            plt.boxplot([x if x else [np.nan] for x in plot_side0_by_disk], showfliers=False)
            plt.title('Density by Disk - Side 0')
            plt.xlabel('Disk')
            plt.ylabel('Bits per Revolution')
            plt.xticks(range(1, len(disk_labels)+1), disk_labels, rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(str(corpus_dir / 'corpus_side0_density_boxplot.png'), dpi=150)
            plt.close()
        if any(len(x) for x in plot_side1_by_disk):
            plt.figure(figsize=(max(8, len(plot_side1_by_disk)*0.6), 5))
            plt.boxplot([x if x else [np.nan] for x in plot_side1_by_disk], showfliers=False)
            plt.title('Density by Disk - Side 1')
            plt.xlabel('Disk')
            plt.ylabel('Bits per Revolution')
            plt.xticks(range(1, len(disk_labels)+1), disk_labels, rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(str(corpus_dir / 'corpus_side1_density_boxplot.png'), dpi=150)
            plt.close()

        # Scatter of density vs variance across all entries per side (if variance available)
        if all_side0 and all_side0_var:
            plt.figure(figsize=(6,5))
            plt.scatter(all_side0, all_side0_var, s=8, alpha=0.5, color='steelblue')
            plt.title('Side 0: Density vs Avg Variance (all entries)')
            plt.xlabel('Bits per Revolution')
            plt.ylabel('Avg Variance')
            plt.tight_layout()
            plt.savefig(str(corpus_dir / 'corpus_side0_density_vs_variance.png'), dpi=150)
            plt.close()
        if all_side1 and all_side1_var:
            plt.figure(figsize=(6,5))
            plt.scatter(all_side1, all_side1_var, s=8, alpha=0.5, color='indianred')
            plt.title('Side 1: Density vs Avg Variance (all entries)')
            plt.xlabel('Bits per Revolution')
            plt.ylabel('Avg Variance')
            plt.tight_layout()
            plt.savefig(str(corpus_dir / 'corpus_side1_density_vs_variance.png'), dpi=150)
            plt.close()
        print("Corpus visualizations saved (histograms, boxplots, scatter plots)")
        # Overlay density curves per disk (approximate via normalized histograms)
        if any(len(x) for x in plot_side0_by_disk):
            plt.figure(figsize=(9,5))
            for dens, label in zip(plot_side0_by_disk, labels_side0):
                if not dens:
                    continue
                counts, bins = np.histogram(dens, bins=min(40, max(10, int(len(dens)/5))), density=True)
                centers = (bins[1:] + bins[:-1]) / 2.0
                # simple smoothing
                if len(counts) > 3:
                    kernel = np.ones(3)/3
                    counts = np.convolve(counts, kernel, mode='same')
                plt.plot(centers, counts, alpha=0.6, label=label)
            plt.title('Side 0 Density Overlays by Disk (normalized)')
            plt.xlabel('Bits per Revolution')
            plt.ylabel('Normalized Density')
            plt.legend(fontsize=8, ncol=2)
            plt.tight_layout()
            plt.savefig(str(corpus_dir / 'corpus_side0_density_overlays.png'), dpi=150)
            plt.close()
        if any(len(x) for x in plot_side1_by_disk):
            plt.figure(figsize=(9,5))
            for dens, label in zip(plot_side1_by_disk, labels_side1):
                if not dens:
                    continue
                counts, bins = np.histogram(dens, bins=min(40, max(10, int(len(dens)/5))), density=True)
                centers = (bins[1:] + bins[:-1]) / 2.0
                if len(counts) > 3:
                    kernel = np.ones(3)/3
                    counts = np.convolve(counts, kernel, mode='same')
                plt.plot(centers, counts, alpha=0.6, label=label)
            plt.title('Side 1 Density Overlays by Disk (normalized)')
            plt.xlabel('Bits per Revolution')
            plt.ylabel('Normalized Density')
            plt.legend(fontsize=8, ncol=2)
            plt.tight_layout()
            plt.savefig(str(corpus_dir / 'corpus_side1_density_overlays.png'), dpi=150)
            plt.close()

        # Per-disk track vs density scatter saved in corpus folder
        for label, tracks in per_disk_tracks_side0:
            if tracks:
                xs, ys = zip(*sorted(tracks, key=lambda t: t[0]))
                fig, axs = plt.subplots(1, 1, figsize=(7,4))
                pcm = axs.scatter(xs, ys, s=10, alpha=0.7, color='steelblue')
                axs.set_title(f'{label} - Side 0: Track vs Density')
                axs.set_xlabel('Track Index')
                axs.set_ylabel('Bits per Revolution')
                plt.tight_layout()
                safe_label = re.sub(r'[^A-Za-z0-9_.-]', '_', label)
                plt.savefig(str(run_dir / 'corpus' / f'corpus_tracks_side0_{safe_label}.png'), dpi=150)
                plt.close()
        for label, tracks in per_disk_tracks_side1:
            if tracks:
                xs, ys = zip(*sorted(tracks, key=lambda t: t[0]))
                plt.figure(figsize=(7,4))
                plt.scatter(xs, ys, s=10, alpha=0.7, color='indianred')
                plt.title(f'{label} - Side 1: Track vs Density')
                plt.xlabel('Track Index')
                plt.ylabel('Bits per Revolution')
                plt.tight_layout()
                plt.savefig(str(run_dir / 'corpus' / f'corpus_tracks_side1_{safe_label}.png'), dpi=150)
                plt.close()

        # Surfaces montage across all disks (one image per disk)
        try:
            # Prefer images copied under run_dir/disks/<label>/<label>_disk_surface.png
            surface_images = []
            disks_dir = run_dir / 'disks'
            if disks_dir.exists():
                for d in sorted([p for p in disks_dir.iterdir() if p.is_dir()]):
                    # Try canonical names first
                    cand = None
                    for pat in [f"{d.name}_disk_surface.png", f"{d.name}_surface_disk_surface.png"]:
                        fp = d / pat
                        if fp.exists():
                            cand = fp
                            break
                    if cand is None:
                        hits = list(d.glob("*_disk_surface.png")) or list(d.glob("*surface_disk_surface.png"))
                        if hits:
                            cand = hits[0]
                    if cand is not None:
                        surface_images.append((d.name, cand))
            # Fallback: look near each input map
            if not surface_images:
                for mp, _ in corpus:
                    pdir = Path(mp).parent
                    hits = list(pdir.glob("*_disk_surface.png")) or list(pdir.glob("*surface_disk_surface.png"))
                    if hits:
                        surface_images.append((pdir.name, hits[0]))

            if surface_images:
                n = len(surface_images)
                cols = min(5, max(2, int(np.ceil(np.sqrt(n)))))
                rows = int(np.ceil(n / cols))
                fig = plt.figure(figsize=(cols*4.0, rows*4.0))
                for i, (lbl, ipath) in enumerate(surface_images, start=1):
                    ax = fig.add_subplot(rows, cols, i)
                    try:
                        img = plt.imread(str(ipath))
                        ax.imshow(img)
                        ax.set_title(lbl, fontsize=9)
                    except Exception as e:
                        ax.text(0.5, 0.5, f"Failed: {Path(ipath).name}", ha='center', va='center')
                    ax.axis('off')
                plt.tight_layout()
                grid_path = corpus_dir / 'corpus_surfaces_grid.png'
                plt.savefig(str(grid_path), dpi=220)
                plt.close()
                print(f"Corpus surfaces montage saved to {grid_path}")
                # Convenience copy into disks/ to aid browsing
                try:
                    disks_dir2 = run_dir / 'disks'
                    disks_dir2.mkdir(parents=True, exist_ok=True)
                    import shutil
                    shutil.copyfile(str(grid_path), str(disks_dir2 / '_corpus_surfaces_grid.png'))
                except Exception:
                    pass
            else:
                print("No disk surface images found for montage; run analyze_corpus with --generate-missing or ensure per-disk outputs exist.")
        except Exception as em:
            print(f"Failed to build corpus surfaces montage: {em}")

        # Side-specific surfaces montages (side 0 and side 1)
        try:
            def build_side_grid(side: int, out_name: str):
                images = []
                disks_dir = run_dir / 'disks'
                if disks_dir.exists():
                    for d in sorted([p for p in disks_dir.iterdir() if p.is_dir()]):
                        cand = None
                        # Prefer canonical side images
                        fp = d / f"{d.name}_surface_side{side}.png"
                        if fp.exists():
                            cand = fp
                        else:
                            hits = list(d.glob(f"*_surface_side{side}.png"))
                            if hits:
                                cand = hits[0]
                        if cand is not None:
                            images.append((d.name, cand))
                if not images:
                    # Fallback: near each surface_map.json's directory
                    for mp, _ in corpus:
                        pdir = Path(mp).parent
                        hits = list(pdir.glob(f"*_surface_side{side}.png"))
                        if hits:
                            images.append((pdir.name, hits[0]))
                if images:
                    n = len(images)
                    cols = min(5, max(2, int(np.ceil(np.sqrt(n)))))
                    rows = int(np.ceil(n / cols))
                    fig = plt.figure(figsize=(cols*4.0, rows*4.0))
                    for i, (lbl, ipath) in enumerate(images, start=1):
                        ax = fig.add_subplot(rows, cols, i)
                        try:
                            img = plt.imread(str(ipath))
                            ax.imshow(img)
                            ax.set_title(lbl, fontsize=9)
                        except Exception:
                            ax.text(0.5, 0.5, f"Failed: {Path(ipath).name}", ha='center', va='center')
                        ax.axis('off')
                    plt.tight_layout()
                    outp = corpus_dir / out_name
                    plt.savefig(str(outp), dpi=220)
                    plt.close()
                    print(f"Corpus side{side} surfaces montage saved to {outp}")
                else:
                    print(f"No side {side} surface images found for montage.")

            build_side_grid(0, 'corpus_side0_surfaces_grid.png')
            build_side_grid(1, 'corpus_side1_surfaces_grid.png')
        except Exception as em2:
            print(f"Failed to build side-specific montages: {em2}")
    except Exception as e:
        print(f"Corpus visualization generation failed: {e}")

    # Optional LLM corpus summary
    if getattr(args, 'summarize', False):
        try:
            host_port = args.lm_host if ':' in args.lm_host else f"{args.lm_host}:1234"
            client = openai.OpenAI(
                base_url=f"http://{host_port}/v1",
                api_key="lm-studio"
            )
            summary_data = {
                'num_disks': corpus_summary.get('num_disks', 0),
                'side0': corpus_summary.get('side0', {}),
                'side1': corpus_summary.get('side1', {}),
                'per_disk': [
                    {
                        'label': d.get('label'),
                        'side0': {
                            'count': d['side0'].get('count'),
                            'avg': d['side0'].get('avg'),
                            'median': d['side0'].get('median')
                        },
                        'side1': {
                            'count': d['side1'].get('count'),
                            'avg': d['side1'].get('avg'),
                            'median': d['side1'].get('median')
                        }
                    } for d in corpus_summary.get('per_disk', [])
                ]
            }

            # If no measurements at all, write deterministic minimal narrative and skip model
            if (summary_data['side0'].get('count') in [0, None]) and (summary_data['side1'].get('count') in [0, None]):
                parsed = {
                    'counts': {'num_disks': summary_data['num_disks']},
                    'side0': summary_data['side0'],
                    'side1': summary_data['side1'],
                    'narrative': 'No valid density measurements found across the provided disks. Verify inputs contain surface_map.json with analysis.density_estimate_bits_per_rev populated.'
                }
                llm_json = corpus_dir / 'llm_corpus_summary.json'
                dump_json(llm_json, parsed)
                llm_txt = corpus_dir / 'llm_corpus_summary.txt'
                with open(llm_txt, 'w') as tf:
                    tf.write(f"FloppyAI LLM Corpus Summary - Generated on {datetime.datetime.now().isoformat()}\n\n")
                    tf.write(parsed.get('narrative', ''))
                print(f"LLM corpus summary saved to {llm_txt} and {llm_json}")
                return 0

            schema_description = (
                "Respond ONLY with a JSON object matching this schema: {\n"
                "  counts: { num_disks: number },\n"
                "  side0: { count:number, min:number|null, max:number|null, avg:number|null, median:number|null },\n"
                "  side1: { count:number, min:number|null, max:number|null, avg:number|null, median:number|null },\n"
                "  per_disk: [{ label:string, side0:{count:number, avg:number|null, median:number|null}, side1:{count:number, avg:number|null, median:number|null} }],\n"
                "  narrative: string\n"
                "}. The 'narrative' must reference ONLY the fields above and avoid domain terms not present in the data. No extra keys, no text outside JSON."
            )

            system_msg = (
                "You are an expert in floppy disk magnetic flux analysis."
                " Output strictly valid JSON per the given schema."
                " Do NOT include any extra text, code fences, explanations, or hidden reasoning."
            )

            payload = {
                'counts': {'num_disks': summary_data['num_disks']},
                'side0': summary_data['side0'],
                'side1': summary_data['side1'],
                'per_disk': summary_data['per_disk'],
            }
            user_prompt = (
                "JSON schema requirements:\n" + schema_description + "\n\n"
                "Data:\n" + json.dumps(payload, indent=2)
            )

            def try_parse_json(text: str):
                s = text.strip()
                if s.startswith('```'):
                    s = s.strip('`')
                    i = s.find('{')
                    if i != -1:
                        s = s[i:]
                i = s.find('{')
                if i > 0:
                    s = s[i:]
                return json.loads(s)

            response = client.chat.completions.create(
                model=args.lm_model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=400,
                temperature=getattr(args, 'lm_temperature', 0.2),
            )
            content = response.choices[0].message.content or ""
            parsed = None
            try:
                parsed = try_parse_json(content)
            except Exception:
                retry_prompt = (
                    "Return ONLY valid JSON per the schema with no extra text. Do not use code fences.\n\n"
                    "Schema:\n" + schema_description + "\n\nData:\n" + json.dumps(payload, indent=2)
                )
                response2 = client.chat.completions.create(
                    model=args.lm_model,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": retry_prompt},
                    ],
                    max_tokens=400,
                    temperature=getattr(args, 'lm_temperature', 0.2),
                )
                content2 = response2.choices[0].message.content or ""
                try:
                    parsed = try_parse_json(content2)
                except Exception:
                    parsed = None

            if parsed is None:
                parsed = {
                    'counts': {'num_disks': summary_data['num_disks']},
                    'side0': summary_data['side0'],
                    'side1': summary_data['side1'],
                    'per_disk': summary_data['per_disk'],
                    'narrative': (
                        "Deterministic corpus summary: {n} disk(s) analyzed. "
                        "Side0 avg {s0a}, median {s0m}; Side1 avg {s1a}, median {s1m}. "
                        "Per-disk medians: " + ", ".join([
                            f"{d['label']}: S0 {d['side0'].get('median')} | S1 {d['side1'].get('median')}" for d in summary_data['per_disk']
                        ])
                    ).format(
                        n=summary_data['num_disks'],
                        s0a=summary_data['side0'].get('avg'),
                        s0m=summary_data['side0'].get('median'),
                        s1a=summary_data['side1'].get('avg'),
                        s1m=summary_data['side1'].get('median'),
                    )
                }

            llm_json = corpus_dir / 'llm_corpus_summary.json'
            dump_json(llm_json, parsed)
            llm_txt = corpus_dir / 'llm_corpus_summary.txt'
            with open(llm_txt, 'w') as tf:
                tf.write(f"FloppyAI LLM Corpus Summary - Generated on {datetime.datetime.now().isoformat()}\n\n")
                tf.write(parsed.get('narrative', ''))
            print(f"LLM corpus summary saved to {llm_txt} and {llm_json}")
        except Exception as e:
            print(f"LLM corpus summary failed: {e}")
            print("Skipping corpus summary generation.")
    print(f"Corpus outputs saved to {corpus_dir}")
    return 0

def classify_surface(args):
    """Classify blank-like vs written-like per track/side using a simple density threshold."""
    run_dir = get_output_dir(args.output_dir)
    with open(args.input, 'r') as f:
        sm = json.load(f)
    threshold = args.blank_density_thresh
    results = {}
    def side_entries(track_obj, side_int):
        if not isinstance(track_obj, dict):
            return []
        if side_int in track_obj:
            return track_obj.get(side_int, [])
        return track_obj.get(str(side_int), [])

    for track in sm:
        if track == 'global':
            continue
        results[track] = {}
        for side in [0, 1]:
            labels = []
            for entry in side_entries(sm.get(track, {}), side):
                if not isinstance(entry, dict):
                    continue
                dens = entry.get('analysis', {}).get('density_estimate_bits_per_rev')
                var = entry.get('analysis', {}).get('noise_profile', {}).get('avg_variance') if 'analysis' in entry else None
                label = 'blank_like' if isinstance(dens, (int, float)) and dens < threshold else 'written_like'
                labels.append({
                    'file': entry.get('file'),
                    'density_estimate_bits_per_rev': dens,
                    'avg_variance': var,
                    'label': label
                })
            results[track][side] = labels
    out = {
        'input': args.input,
        'blank_density_thresh': threshold,
        'classification': results,
    }
    out_path = run_dir / 'classification.json'
    dump_json(out_path, out)
    print(f"Classification saved to {out_path}")
    return 0

def read_track(args):
    """Read a track from hardware."""
    run_dir = get_output_dir(args.output_dir)
    output_raw = str(run_dir / f"read_track_{args.track}_{args.side}.raw")
    wrapper = DTCWrapper(simulation_mode=args.simulate)
    success = wrapper.read_track(
        track=args.track,
        side=args.side,
        output_raw_path=output_raw,
        revolutions=args.revolutions
    )
    if success:
        print(f"Read saved to {output_raw}")
        # Optionally analyze the new stream
        if args.analyze:
            analyze_stream(argparse.Namespace(input=output_raw, output_dir=run_dir))
    else:
        return 1
    return 0

def write_track(args):
    """Write a stream to a track on hardware."""
    wrapper = DTCWrapper(simulation_mode=args.simulate)
    success = wrapper.write_track(
        input_raw_path=args.input,
        track=args.track,
        side=args.side
    )
    if success:
        run_dir = get_output_dir(args.output_dir)
        print(f"Write log in {run_dir}")
    return 0 if success else 1

def generate_dummy(args):
    """Generate a pattern-based flux stream and write to .raw.

    Uses patterns.generate_pattern() to build intervals, then writes either a
    KryoFlux-like stream (default) or a simple internal raw for analysis/testing.
    """
    run_dir = get_output_dir(args.output_dir)

    # Resolve base cell with density scaling and RPM for normalization
    density = getattr(args, 'density', 1.0) or 1.0
    base_cell_ns = float(args.cell_length) / float(density)
    rpm = float(getattr(args, 'rpm', 360.0) or 360.0)

    pattern = getattr(args, 'pattern', 'random')
    seed = getattr(args, 'seed', None)
    out_fmt = getattr(args, 'output_format', 'kryoflux')

    # Build flux intervals for the requested pattern
    flux_intervals = generate_pattern(
        name=pattern,
        revolutions=args.revolutions,
        base_cell_ns=base_cell_ns,
        rpm=rpm,
        seed=seed,
    )

    # Choose output path and writer
    safe_pat = str(pattern).replace('/', '_')
    output_raw = str(run_dir / f"generated_{safe_pat}_t{args.track}_s{args.side}.raw")

    if out_fmt == 'internal':
        write_internal_raw(flux_intervals, args.track, args.side, output_raw, num_revs=args.revolutions)
    else:
        write_kryoflux_stream(flux_intervals, args.track, args.side, output_raw, num_revs=args.revolutions, rpm=rpm)

    print(f"Generated pattern '{pattern}' saved to {output_raw}")
    print(f"Intervals: {len(flux_intervals)} | Revs: {args.revolutions} | Base cell: {base_cell_ns:.1f} ns | RPM: {rpm}")
    # Optionally analyze (best with kryoflux-format output)
    if getattr(args, 'analyze', False):
        try:
            analyze_stream(argparse.Namespace(input=output_raw, output_dir=run_dir))
        except Exception as e:
            print(f"Analysis failed (likely due to non-KryoFlux format): {e}")
    return 0

def encode_data(args):
    """Encode binary data to .raw using custom encoder."""
    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}")
        return 1
    
    run_dir = get_output_dir(args.output_dir)
    if args.output is None:
        output_raw = str(run_dir / f"encoded_track_{args.track}_{args.side}.raw")
    else:
        output_raw = args.output
    
    # Read input data
    with open(args.input, 'rb') as f:
        data_bytes = f.read()
    
    if len(data_bytes) == 0:
        print("No data in input file")
        return 1
    
    # Create encoder
    encoder = CustomEncoder(
        density=args.density,
        variable_mode=args.variable
    )
    
    # Encode
    flux_intervals = encoder.encode_data(data_bytes, num_revs=args.revolutions)
    
    # Generate .raw
    encoder.generate_raw(
        flux_intervals, args.track, args.side, output_raw,
        num_revs=args.revolutions
    )
    
    # Calculate and print density
    achieved_density = encoder.calculate_density(data_bytes)
    print(f"Encoded {len(data_bytes)*8} bits to {output_raw}")
    print(f"Achieved density: {achieved_density:.1f} bits per revolution")
    print(f"Total flux intervals: {len(flux_intervals)}")
    
    # Optional auto-write
    if args.write:
        wrapper = DTCWrapper(simulation_mode=args.simulate)
        success = wrapper.write_track(output_raw, args.track, args.side)
        if success:
            print(f"Successfully wrote to track {args.track} side {args.side}")
        else:
            print("Write failed")
            return 1
    
    # Optional analyze
    if args.analyze:
        analyze_stream(argparse.Namespace(input=output_raw, output_dir=run_dir))
    
    return 0

def decode_data(args):
    """Decode .raw to binary using custom decoder."""
    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}")
        return 1
    
    run_dir = get_output_dir(args.output_dir)
    if args.output is None:
        base = Path(args.input).stem
        args.output = str(run_dir / f"{base}_decoded.bin")
    
    # Create decoder
    decoder = CustomDecoder(
        density=args.density,
        variable_mode=args.variable,
        tolerance=0.15,  # Slightly higher for real flux
        rpm=args.rpm
    )
    
    try:
        decoded = decoder.decode_file(
            args.input,
            output_path=args.output,
            num_revs=args.revolutions
        )
    except Exception as e:
        print(f"Decoding failed: {e}")
        return 1
    
    print(f"Decoded {len(decoded)} bytes to {args.output}")
    
    # Optional verification
    if args.expected and os.path.exists(args.expected):
        with open(args.expected, 'rb') as f:
            expected = f.read()
        if len(decoded) != len(expected):
            print(f"Length mismatch: decoded {len(decoded)} vs expected {len(expected)}")
        else:
            mismatches = sum(a != b for a, b in zip(decoded, expected))
            match_pct = (1 - mismatches / len(decoded)) * 100 if len(decoded) > 0 else 0
            print(f"Verification: {match_pct:.2f}% match ({mismatches} byte errors)")
            if mismatches == 0:
                print("Perfect recovery!")
            else:
                print("Partial recovery - consider AI EC for improvement")
    
    return 0


def summarize_disk_analysis(surface_map, output_dir, args):
    """Generate LLM-powered summary of disk analysis results."""
    try:
        host_port = args.lm_host if ':' in args.lm_host else f"{args.lm_host}:1234"
        client = openai.OpenAI(
            base_url=f"http://{host_port}/v1",
            api_key="lm-studio"
        )

        # Extract key metrics for summary
        summary_data = {
            'num_tracks': len([k for k in surface_map.keys() if k != 'global']),
            'effective_rpm': surface_map['global'].get('effective_rpm'),
            'media_type': surface_map['global'].get('media_type'),
        }

        # Calculate overall statistics
        all_densities = []
        all_variances = []
        for track_key, track_data in surface_map.items():
            if track_key == 'global':
                continue
            for side_key, side_data in track_data.items():
                analysis = side_data.get('analysis', {})
                density = analysis.get('density_estimate_bits_per_rev')
                variance = analysis.get('noise_profile', {}).get('avg_variance')
                if density is not None:
                    all_densities.append(density)
                if variance is not None:
                    all_variances.append(variance)

        if all_densities:
            summary_data['density_stats'] = {
                'count': len(all_densities),
                'min': float(np.min(all_densities)),
                'max': float(np.max(all_densities)),
                'avg': float(np.mean(all_densities)),
                'median': float(np.median(all_densities))
            }

        if all_variances:
            summary_data['variance_stats'] = {
                'count': len(all_variances),
                'min': float(np.min(all_variances)),
                'max': float(np.max(all_variances)),
                'avg': float(np.mean(all_variances)),
                'median': float(np.median(all_variances))
            }

        # Create LLM prompt
        schema_description = (
            "Respond ONLY with a JSON object matching this schema: {\n"
            "  summary_stats: { num_tracks: number, effective_rpm: number, media_type: string|null },\n"
            "  density_stats: { count:number, min:number, max:number, avg:number, median:number }|null,\n"
            "  variance_stats: { count:number, min:number, max:number, avg:number, median:number }|null,\n"
            "  analysis_summary: string\n"
            "}. Provide a concise analysis summary in natural language. No extra keys, no text outside JSON."
        )

        system_msg = (
            "You are an expert in floppy disk magnetic flux analysis."
            " Output strictly valid JSON per the given schema."
            " Do NOT include any extra text, code fences, explanations, or hidden reasoning."
        )

        payload = summary_data
        user_prompt = (
            "JSON schema requirements:\n" + schema_description + "\n\n"
            "Data:\n" + json.dumps(payload, indent=2)
        )
        response = client.chat.completions.create(
            model=args.lm_model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=400,
            temperature=getattr(args, 'lm_temperature', 0.2),
        )
        content = response.choices[0].message.content or ""
        parsed = json.loads(content)

        # Save summary
        summary_json_path = output_dir / 'llm_disk_summary.json'
        dump_json(summary_json_path, parsed)

        summary_txt_path = output_dir / 'llm_disk_summary.txt'
        with open(summary_txt_path, 'w') as f:
            f.write(f"FloppyAI Disk Analysis Summary - Generated on {datetime.datetime.now().isoformat()}\n\n")
            f.write(parsed.get('analysis_summary', ''))

        print(f"LLM summary saved to {summary_txt_path}")

    except Exception as e:
        print(f"LLM summary failed: {e}")
        # Fallback to basic summary
        basic_summary = {
            'summary_stats': summary_data,
            'analysis_summary': f"Analysis completed for disk with {summary_data.get('num_tracks', 0)} tracks."
        }
        summary_json_path = output_dir / 'llm_disk_summary.json'
        dump_json(summary_json_path, basic_summary)

def main():
    parser = argparse.ArgumentParser(
        description="FloppyAI: Flux Stream Analysis and Custom Encoding Tool"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Analyze a .raw stream file
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a .raw stream file")
    analyze_parser.add_argument("input", help="Path to .raw file")
    analyze_parser.add_argument("--output-dir", help="Custom output directory (default: test_outputs/timestamp/)")
    analyze_parser.set_defaults(func=analyze_stream)

    # Read track to .raw
    read_parser = subparsers.add_parser("read", help="Read track from hardware to .raw")
    read_parser.add_argument("track", type=int, help="Track number (0-79)")
    read_parser.add_argument("side", type=int, choices=[0, 1], help="Side (0 or 1)")
    read_parser.add_argument("--revs", type=int, default=3, dest="revolutions", help="Revolutions to read (default: 3)")
    read_parser.add_argument("--simulate", action="store_true", help="Simulate (no hardware)")
    read_parser.add_argument("--analyze", action="store_true", help="Analyze output after reading")
    read_parser.add_argument("--output-dir", help="Custom output directory (default: test_outputs/timestamp/)")
    read_parser.set_defaults(func=read_track)

    # Write .raw to hardware
    write_parser = subparsers.add_parser("write", help="Write .raw to hardware track")
    write_parser.add_argument("input", help="Input .raw file path")
    write_parser.add_argument("track", type=int, help="Track number")
    write_parser.add_argument("side", type=int, choices=[0, 1], help="Side")
    write_parser.add_argument("--simulate", action="store_true", help="Simulate (no hardware)")
    write_parser.add_argument("--output-dir", help="Custom output directory (default: test_outputs/timestamp/)")
    write_parser.set_defaults(func=write_track)

    # Generate dummy/test stream
    gen_parser = subparsers.add_parser("generate", help="Generate test .raw stream")
    gen_parser.add_argument("track", type=int, help="Track number (for naming)")
    gen_parser.add_argument("side", type=int, choices=[0, 1], help="Side (for naming)")
    gen_parser.add_argument("--revs", type=int, default=1, dest="revolutions", help="Revolutions (default: 1)")
    gen_parser.add_argument("--cell", type=int, default=4000, dest="cell_length", help="Nominal cell length ns (default: 4000)")
    gen_parser.add_argument("--density", type=float, default=1.0, help="Density scaling (>1.0 shortens cells)")
    gen_parser.add_argument("--rpm", type=float, default=360.0, help="Assumed RPM for normalization (default: 360)")
    gen_parser.add_argument("--pattern", default="random", help="Pattern name: random|prbs7|prbs15|alt|runlen|chirp|dc_bias|burst")
    gen_parser.add_argument("--seed", type=int, help="Optional seed for pattern generation")
    gen_parser.add_argument("--output-format", choices=["kryoflux", "internal"], default="kryoflux", dest="output_format", help="Output format (default: kryoflux)")
    gen_parser.add_argument("--analyze", action="store_true", help="Analyze after generating (best with kryoflux format)")
    gen_parser.add_argument("--output-dir", help="Custom output directory (default: test_outputs/timestamp/)")
    gen_parser.set_defaults(func=generate_dummy)

    # Generate full disk image set (NN.S.raw)
    gen_disk = subparsers.add_parser("generate_disk", help="Generate full-disk NN.S.raw set for DTC")
    gen_disk.add_argument("--cell", type=int, default=4000, dest="cell_length", help="Nominal cell length ns (default: 4000)")
    gen_disk.add_argument("--density", type=float, default=1.0, help="Density scaling (>1.0 shortens cells)")
    gen_disk.add_argument("--rpm", type=float, default=360.0, help="Assumed RPM (default: 360)")
    gen_disk.add_argument("--pattern", default="random", help="Pattern: random|prbs7|prbs15|alt|runlen|chirp|dc_bias|burst")
    gen_disk.add_argument("--seed", type=int, help="Optional seed")
    gen_disk.add_argument("--revs", type=int, default=1, dest="revolutions", help="Revolutions per file (default: 1)")
    gen_disk.add_argument("--profile", choices=["35HD","35DD","35HDGCR","35DDGCR","525HD","525DD","525DDGCR"], help="Drive/media profile for safe track limits (GCR variants auto-select GCR heuristics)")
    gen_disk.add_argument("--tracks", help="Track range 'a-b' or comma list (default: safe by profile)")
    gen_disk.add_argument("--sides", help="Comma list of sides, e.g., '0,1' (default: 0,1)")
    gen_disk.add_argument("--output-format", choices=["kryoflux","internal"], default="kryoflux", dest="output_format", help="Output format (default: kryoflux)")
    gen_disk.add_argument("--allow-extended", action="store_true", dest="allow_extended", help="Allow tracks beyond safe limits (not recommended)")
    gen_disk.add_argument("--output-dir", help="Custom output directory (default: test_outputs/timestamp/)")
    gen_disk.set_defaults(func=generate_disk)

    # Encode
    encode_parser = subparsers.add_parser("encode", help="Encode binary data to custom .raw stream")
    encode_parser.add_argument("input", help="Input binary data file (e.g., data.bin)")
    encode_parser.add_argument("track", type=int, help="Track number (for naming/metadata)")
    encode_parser.add_argument("side", type=int, choices=[0, 1], help="Side (for naming/metadata)")
    encode_parser.add_argument("--density", type=float, default=1.0, help="Encoding density (default: 1.0; >1.0 for higher)")
    encode_parser.add_argument("--variable", action="store_true", help="Use variable RLL-like cell lengths")
    encode_parser.add_argument("--revs", type=int, default=1, dest="revolutions", help="Number of revolutions (default: 1)")
    encode_parser.add_argument("--output", help="Output .raw path (default: test_outputs/timestamp/encoded_*.raw)")
    encode_parser.add_argument("--write", action="store_true", help="Auto-write generated .raw to hardware after encoding")
    encode_parser.add_argument("--simulate", action="store_true", help="Simulate DTC operations (for --write)")
    encode_parser.add_argument("--analyze", action="store_true", help="Analyze generated .raw after encoding")
    encode_parser.add_argument("--output-dir", help="Custom output directory (default: test_outputs/timestamp/)")
    encode_parser.set_defaults(func=encode_data)

    # Decode
    decode_parser = subparsers.add_parser("decode", help="Decode .raw to binary")
    decode_parser.add_argument("input", help="Input .raw file path")
    decode_parser.add_argument("--output", help="Output binary path (default: run_dir/<stem>_decoded.bin)")
    decode_parser.add_argument("--density", type=float, default=1.0, help="Assumed density for decoding")
    decode_parser.add_argument("--variable", action="store_true", help="Variable cell lengths")
    decode_parser.add_argument("--rpm", type=float, default=300.0, help="Drive RPM assumption (default 300)")
    decode_parser.add_argument("--revs", type=int, default=1, dest="revolutions", help="Revolutions to decode (default: 1)")
    decode_parser.add_argument("--expected", help="Optional expected binary to compare against")
    decode_parser.add_argument("--output-dir", help="Custom output directory (default: test_outputs/timestamp/)")
    decode_parser.set_defaults(func=decode_data)

    # Image → Flux (round-trip experiments)
    img2flux = subparsers.add_parser("image2flux", help="Generate a .raw flux stream from an image for round-trip experiments")
    img2flux.add_argument("image", help="Path to input image (grayscale or color)")
    img2flux.add_argument("track", type=int, help="Track number (for naming/metadata)")
    img2flux.add_argument("side", type=int, choices=[0, 1], help="Side (for naming/metadata)")
    img2flux.add_argument("--revs", type=int, default=1, dest="revolutions", help="Revolutions to emit (default: 1)")
    img2flux.add_argument("--angular-bins", type=int, default=720, dest="angular_bins", help="Angular bins for mapping the image row (default: 720)")
    img2flux.add_argument("--on-count", type=int, default=4, dest="on_count", help="Transitions per ON bin (default: 4)")
    img2flux.add_argument("--off-count", type=int, default=1, dest="off_count", help="Transitions per OFF bin (default: 1)")
    img2flux.add_argument("--interval-ns", type=int, default=2000, dest="interval_ns", help="Interval length per transition in ns (default: 2000)")
    img2flux.add_argument("--rpm", type=float, default=300.0, help="Nominal RPM for metadata when writing KryoFlux-like streams (default: 300)")
    img2flux.add_argument("--dither", choices=["none","ordered","floyd"], default="none", help="Dithering for binarization (default: none)")
    img2flux.add_argument("--threshold", type=float, default=0.5, help="Threshold for binarization (0..1; default: 0.5)")
    img2flux.add_argument("--output-format", choices=["kryoflux","internal"], default="kryoflux", dest="output_format", help="Output format (default: kryoflux)")
    img2flux.add_argument("--output-dir", help="Custom output directory (default: test_outputs/timestamp/)")

    def _run_image2flux(args):
        run_dir = get_output_dir(args.output_dir)
        base = Path(args.image).stem
        out = str(run_dir / f"image2flux_{base}_t{args.track}_s{args.side}.raw")
        try:
            path, n = generate_from_image(
                image_path=args.image,
                track=args.track,
                side=args.side,
                output_path=out,
                revolutions=args.revolutions,
                angular_bins=args.angular_bins,
                on_count=args.on_count,
                off_count=args.off_count,
                interval_ns=args.interval_ns,
                output_format=args.output_format,
                rpm=args.rpm,
                dither=args.dither,
                threshold=args.threshold,
            )
            print(f"Image→flux generated: {path} (intervals={n})")
            print("Next: transfer to Linux DTC host to write/read; then analyze here.")
        except Exception as e:
            print(f"image2flux failed: {e}")
            return 1
        return 0

    img2flux.set_defaults(func=_run_image2flux)

    # Silkscreen: map an image across tracks and angle (per-track .raws)
    silk = subparsers.add_parser("silkscreen", help="Silkscreen an image onto a disk side as per-track .raw streams")
    silk.add_argument("image", help="Path to input image (any resolution)")
    silk.add_argument("--side", type=int, choices=[0,1], default=0, help="Side (default: 0)")
    silk.add_argument("--tracks", help="Track range 'a-b' or comma list (default: safe by profile)")
    silk.add_argument("--profile", choices=["35HD","35DD","35HDGCR","35DDGCR","525HD","525DD","525DDGCR"], help="Profile to set safe track limits and RPM default")
    silk.add_argument("--rpm", type=float, default=300.0, help="Drive RPM (default 300 for 3.5-inch)")
    silk.add_argument("--angular-bins", type=int, default=720, dest="angular_bins", help="Angular bins for θ (default: 720)")
    silk.add_argument("--avg-interval-ns", type=int, default=2200, dest="avg_interval_ns", help="Target average interval per transition in ns (default: 2200)")
    silk.add_argument("--min-interval-ns", type=int, default=2000, dest="min_interval_ns", help="Minimum interval ns (default: 2000)")
    silk.add_argument("--max-interval-ns", type=int, default=8000, dest="max_interval_ns", help="Maximum interval ns (default: 8000)")
    silk.add_argument("--on-count-max", type=int, default=6, dest="on_count_max", help="Max transitions for darkest bins (default: 6)")
    silk.add_argument("--off-count-min", type=int, default=1, dest="off_count_min", help="Min transitions for lightest bins (default: 1)")
    silk.add_argument("--dither", choices=["none","ordered","floyd"], default="floyd", help="Dithering for θ (default: floyd)")
    silk.add_argument("--threshold", type=float, default=0.5, help="Threshold for binarization (0..1; default 0.5)")
    silk.add_argument("--revs", type=int, default=1, dest="revolutions", help="Revolutions to write per track (default: 1)")
    silk.add_argument("--output-format", choices=["kryoflux","internal"], default="kryoflux", dest="output_format", help="Output format (default: kryoflux)")
    silk.add_argument("--allow-extended", action="store_true", dest="allow_extended", help="Allow tracks beyond safe profile limit (not recommended)")
    silk.add_argument("--output-dir", help="Custom output directory (default: test_outputs/timestamp/)")
    silk.add_argument("--disk-name", dest="disk_name", help="Disk name label for output subfolder (default: image filename)")

    def _run_silkscreen(args):
        run_dir = get_output_dir(args.output_dir)
        profile = getattr(args, 'profile', None)
        safe_max = _profile_safe_max(profile)
        tracks = _parse_tracks(getattr(args, 'tracks', None), safe_max)
        # Enforce safe limits unless explicitly overridden
        max_req = max(tracks) if tracks else 0
        if not getattr(args, 'allow_extended', False) and max_req > safe_max:
            print(f"Requested max track {max_req} exceeds safe limit {safe_max} for profile {profile or 'default'}.")
            print("Pass --allow-extended to override (not recommended).")
            return 2
        out_dir = run_dir / 'silkscreen'
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            # Default disk name from image stem if not provided
            try:
                image_stem = Path(args.image).stem
            except Exception:
                image_stem = "disk"
            disk_name = args.disk_name or image_stem
            manifest = generate_silkscreen(
                image_path=args.image,
                tracks=tracks,
                side=args.side,
                output_dir=str(out_dir),
                angular_bins=args.angular_bins,
                rpm=float(args.rpm),
                avg_interval_ns=args.avg_interval_ns,
                min_interval_ns=args.min_interval_ns,
                max_interval_ns=args.max_interval_ns,
                on_count_max=args.on_count_max,
                off_count_min=args.off_count_min,
                dither=args.dither,
                threshold=args.threshold,
                revolutions=args.revolutions,
                output_format=args.output_format,
                disk_name=disk_name,
            )
            mf_path = run_dir / 'silkscreen_manifest.json'
            dump_json(mf_path, manifest)
            print(f"Silkscreen set written under {out_dir} ({len(manifest.get('files', []))} files)")
            gt_png = manifest.get('ground_truth_png')
            if gt_png:
                print(f"Ground truth (polar) saved to: {gt_png}")
            print("Next: transfer to Linux DTC host to write/read per-track .raw, then analyze here.")
        except Exception as e:
            print(f"silkscreen failed: {e}")
            return 1
        return 0

    silk.set_defaults(func=_run_silkscreen)

    # Silkscreen from built-in patterns (no external image required)
    silkp = subparsers.add_parser("silkscreen_pattern", help="Generate per-track .raws by silkscreening a built-in pattern")
    silkp.add_argument("pattern", choices=["checker","wedges","bars_theta","bars_radial"], help="Pattern name")
    silkp.add_argument("--side", type=int, choices=[0,1], default=0, help="Side (default: 0)")
    silkp.add_argument("--tracks", help="Track range 'a-b' or comma list (default: safe by profile)")
    silkp.add_argument("--profile", choices=["35HD","35DD","35HDGCR","35DDGCR","525HD","525DD","525DDGCR"], help="Profile to set safe track limits and RPM default")
    silkp.add_argument("--rpm", type=float, default=300.0, help="Drive RPM (default 300 for 3.5-inch)")
    silkp.add_argument("--angular-bins", type=int, default=720, dest="angular_bins", help="Angular bins for θ (default: 720)")
    silkp.add_argument("--avg-interval-ns", type=int, default=2200, dest="avg_interval_ns", help="Target average interval per transition in ns (default: 2200)")
    silkp.add_argument("--min-interval-ns", type=int, default=2000, dest="min_interval_ns", help="Minimum interval ns (default: 2000)")
    silkp.add_argument("--max-interval-ns", type=int, default=8000, dest="max_interval_ns", help="Maximum interval ns (default: 8000)")
    silkp.add_argument("--on-count-max", type=int, default=6, dest="on_count_max", help="Max transitions for darkest bins (default: 6)")
    silkp.add_argument("--off-count-min", type=int, default=1, dest="off_count_min", help="Min transitions for lightest bins (default: 1)")
    silkp.add_argument("--dither", choices=["none","ordered","floyd"], default="floyd", help="Dithering for θ (default: floyd)")
    silkp.add_argument("--threshold", type=float, default=0.5, help="Threshold for binarization (0..1; default 0.5)")
    silkp.add_argument("--revs", type=int, default=1, dest="revolutions", help="Revolutions to write per track (default: 1)")
    silkp.add_argument("--output-format", choices=["kryoflux","internal"], default="kryoflux", dest="output_format", help="Output format (default: kryoflux)")
    silkp.add_argument("--allow-extended", action="store_true", dest="allow_extended", help="Allow tracks beyond safe profile limit (not recommended)")
    silkp.add_argument("--output-dir", help="Custom output directory (default: test_outputs/timestamp/)")
    silkp.add_argument("--disk-name", dest="disk_name", help="Disk name label for output subfolder (default: pattern name)")
    # Pattern-specific controls
    silkp.add_argument("--k", type=int, default=12, help="Wedges: number of sectors (default: 12)")
    silkp.add_argument("--duty", type=float, default=0.5, help="Duty cycle for bars/wedges (0-1, default: 0.5)")
    silkp.add_argument("--theta-period", type=int, default=36, dest="theta_period", help="Theta bars/checker period in bins (default: 36)")
    silkp.add_argument("--radial-period", type=int, default=8, dest="radial_period", help="Radial bars/checker period in bins (default: 8)")

    def _run_silkscreen_pattern(args):
        run_dir = get_output_dir(args.output_dir)
        profile = getattr(args, 'profile', None)
        safe_max = _profile_safe_max(profile)
        tracks = _parse_tracks(getattr(args, 'tracks', None), safe_max)
        max_req = max(tracks) if tracks else 0
        if not getattr(args, 'allow_extended', False) and max_req > safe_max:
            print(f"Requested max track {max_req} exceeds safe limit {safe_max} for profile {profile or 'default'}.")
            print("Pass --allow-extended to override (not recommended).")
            return 2
        out_dir = run_dir / 'silkscreen_pattern'
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            radial_bins = len(tracks)
            polar = generate_polar_pattern(
                name=args.pattern,
                angular_bins=int(args.angular_bins),
                radial_bins=int(radial_bins),
                k=int(getattr(args, 'k', 12)),
                duty=float(getattr(args, 'duty', 0.5)),
                theta_period=int(getattr(args, 'theta_period', 36)),
                radial_period=int(getattr(args, 'radial_period', 8)),
            )
            # Save the intended pattern for convenience
            try:
                save_polar_png(polar, str(out_dir / 'pattern.png'))
            except Exception:
                pass
            # Default disk name from pattern name if not provided
            disk_name = args.disk_name or f"{args.pattern}"
            manifest = generate_silkscreen(
                image_path=None,
                tracks=tracks,
                side=args.side,
                output_dir=str(out_dir),
                angular_bins=args.angular_bins,
                rpm=float(args.rpm),
                avg_interval_ns=args.avg_interval_ns,
                min_interval_ns=args.min_interval_ns,
                max_interval_ns=args.max_interval_ns,
                on_count_max=args.on_count_max,
                off_count_min=args.off_count_min,
                dither=args.dither,
                threshold=args.threshold,
                revolutions=args.revolutions,
                output_format=args.output_format,
                polar_override=polar,
                disk_name=disk_name,
            )
            from utils.json_io import dump_json as _dump
            _dump(out_dir / 'silkscreen_pattern_manifest.json', manifest)
            print(f"Silkscreen pattern '{args.pattern}' written under {out_dir} ({len(manifest.get('files', []))} files)")
            print("Next: transfer to Linux DTC host to write/read per-track .raw, then analyze here.")
        except Exception as e:
            print(f"silkscreen_pattern failed: {e}")
            return 1
        return 0

    silkp.set_defaults(func=_run_silkscreen_pattern)

    # Recover image: reconstruct polar image from surface_map.json and compare with expected
    rec = subparsers.add_parser("recover_image", help="Reconstruct a polar (track×angle) image from surface_map.json")
    rec.add_argument("input", help="Path to surface_map.json")
    rec.add_argument("--side", type=int, choices=[0,1], default=0, help="Side to recover (default: 0)")
    rec.add_argument("--tracks", help="Track range 'a-b' or comma list (default: all present)")
    rec.add_argument("--angular-bins", type=int, default=720, dest="angular_bins", help="Angular bins for reconstruction (default: 720)")
    rec.add_argument("--expected", help="Optional expected polar image (PNG) to compare against")
    rec.add_argument("--output-dir", help="Custom output directory (default: test_outputs/timestamp/)")

    def _run_recover(args):
        run_dir = get_output_dir(args.output_dir)
        out_dir = run_dir / 'recover'
        out_dir.mkdir(parents=True, exist_ok=True)
        # Tracks: if provided, parse; else None to auto-select from surface_map
        profile = None
        safe_max = _profile_safe_max(profile)
        tracks = None
        if getattr(args, 'tracks', None):
            tracks = _parse_tracks(args.tracks, safe_max)
        try:
            man = recover_image_func(
                surface_map_path=args.input,
                output_dir=str(out_dir),
                side=int(args.side),
                tracks=tracks,
                angular_bins=int(args.angular_bins),
                expected=args.expected,
            )
            from utils.json_io import dump_json as _dump
            _dump(out_dir / 'recover_manifest.json', man)
            print(f"Recovered polar image saved to {man.get('recovered_png')}\nMetrics: {man.get('metrics')}")
        except Exception as e:
            print(f"recover_image failed: {e}")
            return 1
        return 0

    rec.set_defaults(func=_run_recover)

    # Analyze Disk (delegates to analysis.analyze_disk.run if available)
    analyze_disk_parser = subparsers.add_parser("analyze_disk", help="Batch analyze .raw streams for disk surface map")
    analyze_disk_parser.add_argument("input", nargs='?', default="../example_stream_data/", help="Directory or single .raw file to analyze")
    analyze_disk_parser.add_argument("--track", type=int, help="Manual track number if not parsable from filename")
    analyze_disk_parser.add_argument("--side", type=int, choices=[0, 1], help="Manual side number if not parsable from filename")
    analyze_disk_parser.add_argument("--rpm", type=float, help="Drive RPM for normalization/validation")
    analyze_disk_parser.add_argument("--profile", choices=["35HD","35DD","35HDGCR","35DDGCR","525HD","525DD","525DDGCR"], help="Drive/media profile (GCR variants auto-select GCR overlays)")
    analyze_disk_parser.add_argument("--lm-host", default="localhost:1234", help="LM Studio host (IP or host:port)")
    analyze_disk_parser.add_argument("--lm-model", default="local-model", help="LM model name")
    analyze_disk_parser.add_argument("--lm-temperature", type=float, default=0.2, dest="lm_temperature", help="Temperature for LLM summary")
    analyze_disk_parser.add_argument("--summarize", action="store_true", help="Generate LLM-powered summary report")
    analyze_disk_parser.add_argument("--summary-format", choices=["json", "text"], default="json", dest="summary_format", help="Summary output format")
    analyze_disk_parser.add_argument("--output-dir", help="Custom output directory (default: test_outputs/timestamp/)")
    analyze_disk_parser.add_argument("--media-type", choices=["35HD","35DD","525HD","525DD"], dest="media_type", help="Override media type")
    analyze_disk_parser.add_argument("--format-overlay", action="store_true", dest="format_overlay", help="Render format-aware sector overlays (heuristic)")
    analyze_disk_parser.add_argument("--angular-bins", type=int, default=0, dest="angular_bins", help="Angular bins or sector count hint (0 = auto)")
    analyze_disk_parser.add_argument("--overlay-alpha", type=float, default=0.8, dest="overlay_alpha", help="Overlay line alpha")
    analyze_disk_parser.add_argument("--overlay-color", default="#ff3333", dest="overlay_color", help="Overlay line color")
    analyze_disk_parser.add_argument("--overlay-mode", choices=["mfm","gcr","auto"], default="auto", dest="overlay_mode", help="Overlay heuristic mode (auto = pick from profile)")
    analyze_disk_parser.add_argument("--gcr-candidates", default="10,12,8,9,11,13", dest="gcr_candidates", help="Comma-separated GCR sector count candidates")
    analyze_disk_parser.add_argument("--overlay-sectors-hint", type=int, dest="overlay_sectors_hint", help="Explicit sector count hint")
    analyze_disk_parser.add_argument("--export-format", choices=["png","svg","both"], default="png", dest="export_format", help="Output format for figures (default: png)")
    # Fidelity presets and overrides
    analyze_disk_parser.add_argument("--quality", choices=["normal","high","ultra"], default="ultra", dest="quality", help="Rendering fidelity preset (default: ultra)")
    analyze_disk_parser.add_argument("--theta-samples", type=int, dest="theta_samples", help="Override angular resolution (samples around circle)")
    analyze_disk_parser.add_argument("--radial-upsample", type=int, dest="radial_upsample", help="Override radial upsample factor")
    analyze_disk_parser.add_argument("--render-dpi", type=int, dest="render_dpi", help="Override figure save DPI")
    analyze_disk_parser.add_argument("--align-to-sectors", choices=["off","side","track","auto"], default="off", dest="align_to_sectors", help="Rotate polar plots so sector 0° aligns to detected boundary (default: off; auto = side)")
    analyze_disk_parser.add_argument("--label-sectors", action="store_true", dest="label_sectors", help="Annotate sector numbers on polar plots when wedge count is known")
    analyze_disk_parser.set_defaults(func=(analyze_disk_cmd if analyze_disk_cmd else lambda _args: (print("analyze_disk not available"), 1)[1]))

    # Analyze Corpus
    corpus_parser = subparsers.add_parser("analyze_corpus", help="Aggregate multiple surface_map.json files for a corpus summary")
    corpus_parser.add_argument("inputs", help="Directory containing runs or a single surface_map.json")
    corpus_parser.add_argument("--output-dir", help="Custom output directory (default: test_outputs/timestamp/)")
    corpus_parser.add_argument("--generate-missing", action="store_true", dest="generate_missing", help="Generate missing surface maps before aggregating")
    corpus_parser.add_argument("--rpm", type=float, help="Drive RPM for normalization when generating missing maps")
    corpus_parser.add_argument("--profile", choices=["35HD","35DD","35HDGCR","35DDGCR","525HD","525DD","525DDGCR"], help="Drive profile (sets RPM if --rpm not specified; GCR variants auto-select GCR overlays)")
    corpus_parser.add_argument("--media-type", choices=["35HD","35DD","525HD","525DD"], dest="media_type", help="Override media type for generated runs")
    corpus_parser.add_argument("--summarize", action="store_true", help="Generate LLM-powered corpus summary report")
    corpus_parser.add_argument("--lm-host", default="localhost:1234", help="LM Studio host (IP or host:port)")
    corpus_parser.add_argument("--lm-model", default="local-model", help="LM model name")
    corpus_parser.add_argument("--lm-temperature", type=float, default=0.2, dest="lm_temperature", help="Temperature for LLM corpus summary")
    corpus_parser.add_argument("--format-overlay", action="store_true", dest="format_overlay", help="Render format-aware overlays in per-disk runs")
    corpus_parser.add_argument("--angular-bins", type=int, default=0, dest="angular_bins", help="Angular bins or sector count hint for overlays")
    corpus_parser.add_argument("--overlay-alpha", type=float, default=0.8, dest="overlay_alpha", help="Overlay line alpha")
    corpus_parser.add_argument("--overlay-color", default="#ff3333", dest="overlay_color", help="Overlay line color")
    corpus_parser.add_argument("--overlay-mode", choices=["mfm","gcr","auto"], default="auto", dest="overlay_mode", help="Overlay heuristic mode (auto = pick from profile)")
    corpus_parser.add_argument("--gcr-candidates", default="10,12,8,9,11,13", dest="gcr_candidates", help="Comma-separated GCR sector count candidates")
    corpus_parser.add_argument("--overlay-sectors-hint", type=int, dest="overlay_sectors_hint", help="Explicit sector count hint to use when detection is inconclusive")
    corpus_parser.set_defaults(func=analyze_corpus)

    # Classify surface
    classify_parser = subparsers.add_parser("classify_surface", help="Classify blank-like vs written-like for a surface_map.json")
    classify_parser.add_argument("input", help="Path to surface_map.json")
    classify_parser.add_argument("--blank-density-thresh", type=float, default=1000.0, dest="blank_density_thresh", help="Density threshold below which an entry is considered blank-like")
    classify_parser.add_argument("--output-dir", help="Custom output directory (default: test_outputs/timestamp/)")
    classify_parser.set_defaults(func=classify_surface)

    # Plan pool
    pool_parser = subparsers.add_parser("plan_pool", help="Select top-quality tracks to form a dense bit pool")
    pool_parser.add_argument("input", help="Path to surface_map.json")
    pool_parser.add_argument("--min-density", type=float, default=2000.0, dest="min_density", help="Minimum density (bits/rev) for candidate selection")
    pool_parser.add_argument("--top-percent", type=float, default=0.2, dest="top_percent", help="Top percentile of candidates to keep (0-1)")
    pool_parser.add_argument("--output-dir", help="Custom output directory (default: test_outputs/timestamp/)")
    pool_parser.set_defaults(func=(plan_pool if 'plan_pool' in globals() else (lambda _args: (print("plan_pool not available"), 1)[1])))

    # Compare reads
    cmp_parser = subparsers.add_parser("compare_reads", help="Compare multiple reads of the same disk (2+ surface_map.json paths or directories)")
    cmp_parser.add_argument("inputs", nargs='+', help="Paths to surface_map.json or directories that contain them")
    cmp_parser.add_argument("--output-dir", help="Custom output directory (default: test_outputs/timestamp/)")
    cmp_parser.set_defaults(func=compare_reads_cmd)

    # Experiments command (matrix)
    exp_parser = subparsers.add_parser("experiment", help="Run systematic flux analysis experiments")
    exp_subparsers = exp_parser.add_subparsers(dest="experiment_command", help="Experiment subcommands")
    matrix_parser = exp_subparsers.add_parser("matrix", help="Run experiment matrix with various parameters")
    matrix_parser.add_argument("--experiment", default="flux_analysis", help="Experiment name")
    matrix_parser.add_argument("--patterns", nargs='+', default=['random', 'prbs7', 'alt'], help="Test patterns to use")
    matrix_parser.add_argument("--densities", nargs='+', type=float, default=[0.5, 1.0, 1.5, 2.0], help="Density multipliers to test")
    matrix_parser.add_argument("--tracks", nargs='+', type=int, default=list(range(0, 10)), help="Track numbers to test (default: 0-9)")
    matrix_parser.add_argument("--sides", nargs='+', type=int, choices=[0, 1], default=[0, 1], help="Sides to test")
    matrix_parser.add_argument("--revolutions", type=int, default=3, help="Revolutions to read/write per test")
    matrix_parser.add_argument("--repetitions", type=int, default=3, help="Number of repetitions per parameter combination")
    matrix_parser.add_argument("--no-simulate", action="store_false", dest="simulate", help="Disable simulation mode (use with caution!)")
    matrix_parser.add_argument("--output-dir", help="Custom output directory")
    matrix_parser.set_defaults(func=(run_experiment_matrix_cmd if run_experiment_matrix_cmd else lambda _args: (print("experiment matrix not available"), 1)[1]))

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 0
    return args.func(args) if getattr(args, 'func', None) else 0

if __name__ == "__main__":
    # For direct execution from src/ directory
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    sys.exit(main())