#!/usr/bin/env python3
import datetime
import os
import sys
from pathlib import Path
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
from dtc_wrapper import DTCWrapper
from custom_encoder import CustomEncoder
from custom_decoder import CustomDecoder
import glob
import json
import re
import numpy as np
import matplotlib.pyplot as plt
import json
import openai

def get_output_dir(output_dir=None):
    """Get or create timestamped output directory."""
    base_dir = Path("test_outputs") if output_dir is None else Path(output_dir)
    base_dir.mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = base_dir / timestamp
    run_dir.mkdir(exist_ok=True)
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

def analyze_corpus(args):
    """Aggregate multiple surface_map.json files to produce a corpus-level summary."""
    run_dir = get_output_dir(args.output_dir)
    inputs = []
    base = Path(args.inputs)

    # Optionally generate surface_map.json for directories containing .raw files
    if getattr(args, 'generate_missing', False) and base.is_dir():
        raw_dirs = {p.parent for p in base.rglob('*.raw')}
        for d in sorted(raw_dirs):
            try:
                print(f"Generating surface map for {d} ...")
                # Invoke analyze_disk programmatically
                disk_args = argparse.Namespace(
                    input=str(d), track=None, side=None, rpm=getattr(args, 'rpm', 360),
                    lm_host=getattr(args, 'lm_host', 'localhost:1234'),
                    lm_model=getattr(args, 'lm_model', 'local-model'),
                    lm_temperature=getattr(args, 'lm_temperature', 0.2),
                    summarize=getattr(args, 'summarize', False),
                    output_dir=None, summary_format='json'
                )
                analyze_disk(disk_args)
            except Exception as e:
                print(f"Failed to analyze {d}: {e}")

    if base.is_dir():
        found = {p for p in base.rglob('surface_map.json')}
        # Include newly generated runs under test_outputs as well
        test_out = Path('test_outputs')
        if test_out.exists():
            found.update(test_out.rglob('surface_map.json'))
        inputs = sorted(found)
    elif base.name.endswith('.json'):
        inputs = [base]
    if not inputs:
        print(f"No surface_map.json found under {args.inputs}")
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
        for track in sm:
            if track == 'global':
                continue
            for entry in side_entries(sm[track], 0):
                if isinstance(entry, dict):
                    d = entry.get('analysis', {}).get('density_estimate_bits_per_rev')
                    var = entry.get('analysis', {}).get('noise_profile', {}).get('avg_variance') if isinstance(entry.get('analysis', {}), dict) else None
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
        disk_labels.append(Path(mp).parent.name)
        per_disk_tracks_side0.append((disk_labels[-1], per_disk_t0))
        per_disk_tracks_side1.append((disk_labels[-1], per_disk_t1))
        disk_stats = {
            'path': str(mp),
            'label': Path(mp).parent.name,
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

    out_path = run_dir / 'corpus_summary.json'
    with open(out_path, 'w') as f:
        json.dump(corpus_summary, f, indent=2)
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
            plt.savefig(str(run_dir / 'corpus_side0_density_hist.png'), dpi=150)
            plt.close()
        if all_side1:
            plt.figure(figsize=(8,4))
            plt.hist(all_side1, bins=40, color='indianred', alpha=0.8)
            plt.title('Density Distribution - Side 1 (all disks)')
            plt.xlabel('Bits per Revolution')
            plt.ylabel('Count')
            plt.tight_layout()
            plt.savefig(str(run_dir / 'corpus_side1_density_hist.png'), dpi=150)
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
            plt.savefig(str(run_dir / 'corpus_side0_density_boxplot.png'), dpi=150)
            plt.close()
        if any(len(x) for x in plot_side1_by_disk):
            plt.figure(figsize=(max(8, len(plot_side1_by_disk)*0.6), 5))
            plt.boxplot([x if x else [np.nan] for x in plot_side1_by_disk], showfliers=False)
            plt.title('Density by Disk - Side 1')
            plt.xlabel('Disk')
            plt.ylabel('Bits per Revolution')
            plt.xticks(range(1, len(disk_labels)+1), disk_labels, rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(str(run_dir / 'corpus_side1_density_boxplot.png'), dpi=150)
            plt.close()

        # Scatter of density vs variance across all entries per side (if variance available)
        if all_side0 and all_side0_var:
            plt.figure(figsize=(6,5))
            plt.scatter(all_side0, all_side0_var, s=8, alpha=0.5, color='steelblue')
            plt.title('Side 0: Density vs Avg Variance (all entries)')
            plt.xlabel('Bits per Revolution')
            plt.ylabel('Avg Variance')
            plt.tight_layout()
            plt.savefig(str(run_dir / 'corpus_side0_density_vs_variance.png'), dpi=150)
            plt.close()
        if all_side1 and all_side1_var:
            plt.figure(figsize=(6,5))
            plt.scatter(all_side1, all_side1_var, s=8, alpha=0.5, color='indianred')
            plt.title('Side 1: Density vs Avg Variance (all entries)')
            plt.xlabel('Bits per Revolution')
            plt.ylabel('Avg Variance')
            plt.tight_layout()
            plt.savefig(str(run_dir / 'corpus_side1_density_vs_variance.png'), dpi=150)
            plt.close()
        print("Corpus visualizations saved (histograms, boxplots, scatter plots)")
        # Overlay density curves per disk (approximate via normalized histograms)
        if any(len(x) for x in plot_side0_by_disk):
            plt.figure(figsize=(9,5))
            for dens, label in zip(plot_side0_by_disk, disk_labels):
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
            plt.savefig(str(run_dir / 'corpus_side0_density_overlays.png'), dpi=150)
            plt.close()
        if any(len(x) for x in plot_side1_by_disk):
            plt.figure(figsize=(9,5))
            for dens, label in zip(plot_side1_by_disk, disk_labels):
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
            plt.savefig(str(run_dir / 'corpus_side1_density_overlays.png'), dpi=150)
            plt.close()

        # Per-disk track vs density scatter saved in corpus folder
        for label, tracks in per_disk_tracks_side0:
            if tracks:
                xs, ys = zip(*sorted(tracks, key=lambda t: t[0]))
                plt.figure(figsize=(7,4))
                plt.scatter(xs, ys, s=10, alpha=0.7, color='steelblue')
                plt.title(f'{label} - Side 0: Track vs Density')
                plt.xlabel('Track Index')
                plt.ylabel('Bits per Revolution')
                plt.tight_layout()
                safe_label = re.sub(r'[^A-Za-z0-9_.-]', '_', label)
                plt.savefig(str(run_dir / f'corpus_tracks_side0_{safe_label}.png'), dpi=150)
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
                safe_label = re.sub(r'[^A-Za-z0-9_.-]', '_', label)
                plt.savefig(str(run_dir / f'corpus_tracks_side1_{safe_label}.png'), dpi=150)
                plt.close()
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
                llm_json = run_dir / 'llm_corpus_summary.json'
                with open(llm_json, 'w') as jf:
                    json.dump(parsed, jf, indent=2)
                llm_txt = run_dir / 'llm_corpus_summary.txt'
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

            llm_json = run_dir / 'llm_corpus_summary.json'
            with open(llm_json, 'w') as jf:
                json.dump(parsed, jf, indent=2)
            llm_txt = run_dir / 'llm_corpus_summary.txt'
            with open(llm_txt, 'w') as tf:
                tf.write(f"FloppyAI LLM Corpus Summary - Generated on {datetime.datetime.now().isoformat()}\n\n")
                tf.write(parsed.get('narrative', ''))
            print(f"LLM corpus summary saved to {llm_txt} and {llm_json}")
        except Exception as e:
            print(f"LLM corpus summary failed: {e}")
            print("Skipping corpus summary generation.")
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
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
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
    """Generate a dummy stream for testing custom flux."""
    run_dir = get_output_dir(args.output_dir)
    output_raw = str(run_dir / f"generated_track_{args.track}_{args.side}.raw")
    wrapper = DTCWrapper(simulation_mode=True)  # Always simulate for generate
    wrapper.generate_dummy_stream(
        track=args.track,
        side=args.side,
        output_raw_path=output_raw,
        revolutions=args.revolutions,
        cell_length_ns=args.cell_length
    )
    print(f"Generated saved to {output_raw}")
    # Analyze the dummy
    if args.analyze:
        analyze_stream(argparse.Namespace(input=output_raw, output_dir=run_dir))
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

def analyze_disk(args):
    """Batch analyze .raw files from directory or single file to surface map."""
    run_dir = get_output_dir(args.output_dir)
    surface_map = {}  # {track: {side: list of {stats, analysis}}}
    
    input_path = args.input
    raw_files = []
    if Path(input_path).is_file():
        filename = Path(input_path).name
        if re.search(r'(\d+)\.(\d)\.raw$', filename):
            parent_dir = Path(input_path).parent
            other_raws = list(parent_dir.glob("*.raw"))
            if len(other_raws) > 1:
                print(f"Detected numbered file {filename}; batching parent directory {parent_dir} for full disk analysis.")
                input_path = parent_dir
                raw_files = [str(p) for p in input_path.glob("*.raw")]
            else:
                raw_files = [str(input_path)]
        else:
            raw_files = [str(input_path)]
    else:
        raw_files = [str(p) for p in Path(input_path).glob("*.raw")]

    if not raw_files:
        print(f"No .raw files found in {input_path}")
        return 1

    print(f"Analyzing {len(raw_files)} stream file(s) from {input_path}...")

    # Collect valid files with metadata for sorting
    file_metadata = []
    for raw_path in raw_files:
        filename = Path(raw_path).name
        match = re.search(r'(\d{1,3})\.(\d)\.raw$', filename)
        track = None
        side = None
        if match:
            track_str, side_str = match.groups()
            try:
                track = int(track_str)
                side = int(side_str)
                if track > 83:
                    # Fallback for concatenated prefix+track: take last two digits
                    track_str = track_str[-2:]  # e.g., '180' -> '80'
                    track = int(track_str)
                    print(f"Adjusted high track {track_str} to {track} for {filename} (prefix concatenation fallback)")
                if 0 <= track <= 83 and side in [0, 1]:
                    file_metadata.append((track, side, raw_path, filename))
                else:
                    print(f"Skipping out-of-range track {track} or side {side} in {filename}")
            except ValueError:
                print(f"Skipping unparseable filename: {filename}")
        else:
            if args.track is not None and args.side is not None:
                track = args.track
                side = args.side
                if 0 <= track <= 83 and side in [0, 1]:
                    file_metadata.append((track, side, raw_path, filename))
                else:
                    print(f"Skipping out-of-range manual track {track} or side {side}")
            else:
                print(f"Skipping {filename}: no track/side pattern and no manual specified")
    
    if not file_metadata:
        print("No valid files to process")
        return 1
    
    # Sort by track, then side for ordered processing (00.0, 00.1, 01.0, etc.)
    file_metadata.sort(key=lambda x: (x[0], x[1]))
    
    expected_total = 84 * 2  # 0-83 tracks, 2 sides
    found_total = len(file_metadata)
    print(f"Processing {found_total} valid files from {input_path} (expected up to {expected_total} for full disk)...")
    print("Order: track 00 side 0, 00 side 1, ..., 83 side 1")
    
    # First pass: parse all files and collect data per track/side
    track_side_data = {}  # { (track, side): list of (analyzer, filename, parsed, analysis) }
    for track, side, raw_path, filename in file_metadata:
        try:
            analyzer = FluxAnalyzer()
            parsed = analyzer.parse(raw_path)
            analysis = analyzer.analyze()
            key = (track, side)
            if key not in track_side_data:
                track_side_data[key] = []
            track_side_data[key].append((analyzer, filename, parsed, analysis))
            print(f"Parsed track {track:02d} side {side}: {filename}")
        except Exception as e:
            print(f"Error parsing {filename}: {e}")
    
    # Collect all individual entries for JSON
    for (track, side), data_list in track_side_data.items():
        if track not in surface_map:
            surface_map[track] = {}
        if side not in surface_map[track]:
            surface_map[track][side] = []
        for analyzer, filename, parsed, analysis in data_list:
            surface_map[track][side].append({
                'file': filename,
                'path': raw_path,
                'stats': parsed['stats'],
                'analysis': analysis
            })
        print(f"Processed track {track:02d} side {side} ({len(data_list)} files)")
    
    # Save map with aggregated stats if multiple files
    for track in surface_map:
        for side in surface_map[track]:
            entries = surface_map[track][side]
            data_entries = [e for e in entries if 'stats' in e and not e.get('aggregate', False)]  # Filter data only
            if len(data_entries) > 1:
                # Simple aggregate: average stats
                avg_stats = {}
                for key in ['mean_interval_ns', 'std_interval_ns', 'total_fluxes']:
                    values = [e['stats'].get(key, 0) for e in data_entries]
                    avg_stats[key] = np.mean(values) if values else 0
                # Add as special entry
                surface_map[track][side].append({
                    'aggregate': True,
                    'stats': avg_stats,
                    'analysis': 'Aggregated across files'
                })
            
            # Add side-specific summary (after aggregation)
            if len(data_entries) >= 1:
                avg_protection = np.mean([e['analysis'].get('protection_score', 0) for e in data_entries])
                side_max_density = np.mean([e['analysis'].get('max_theoretical_density_bits_per_rev', 0) for e in data_entries])
                surface_map[track][side].append({
                    'side_summary': True,
                    'avg_protection_score': float(avg_protection),
                    'avg_max_density': int(side_max_density),
                    'likely_protected': avg_protection > 0.3
                })
    
    # Global aggregation for entire disk visualization and analysis
    global_flux = []
    global_revs = []
    total_revs = 0
    total_flux_sum = 0
    for key in track_side_data:
        for analyzer, _, parsed, _ in track_side_data[key]:
            global_flux.extend(analyzer.flux_data)
            global_revs.extend(analyzer.revolutions)
            total_revs += len(analyzer.revolutions)
            total_flux_sum += np.sum(analyzer.flux_data)
    
    if global_flux:
        global_analyzer = FluxAnalyzer()
        global_analyzer.flux_data = np.array(global_flux)
        global_analyzer.revolutions = global_revs
        
        # Compute global stats
        if len(global_flux) > 0:
            total_time = np.sum(global_flux)
            rev_time = total_time / max(1, total_revs)
            global_stats = {
                'total_fluxes': len(global_flux),
                'mean_interval_ns': float(np.mean(global_flux)),
                'std_interval_ns': float(np.std(global_flux)),
                'min_interval_ns': int(np.min(global_flux)),
                'max_interval_ns': int(np.max(global_flux)),
                'total_revolution_time_ns': float(total_time),
                'num_revolutions': total_revs,
                'measured_rev_time_ns': float(rev_time),
                'measured_rpm': 60000000000 / rev_time if rev_time > 0 else 0,
            }
            global_analyzer.stats = global_stats
            
            # RPM validation if provided
            rpm_drift = None
            if args.rpm:
                expected_rev_time = 60000000000 / args.rpm
                actual_rev_time = rev_time
                rpm_drift = abs((actual_rev_time - expected_rev_time) / expected_rev_time * 100)
                print(f"RPM validation: Expected {args.rpm} RPM, measured ~{global_stats['measured_rpm']:.1f} RPM, drift {rpm_drift:.2f}% (stable if <1%)")
                global_stats['rpm_drift_pct'] = rpm_drift
                # Normalize global for known RPM
                global_scale = expected_rev_time / actual_rev_time if actual_rev_time > 0 else 1.0
                global_stats['normalized_scale'] = global_scale
                global_stats['estimated_full_fluxes'] = len(global_flux) * global_scale
                global_stats['density_bits_per_full_rev'] = int(8 * global_stats['estimated_full_fluxes'])
            else:
                print(f"Global measured RPM: {global_stats['measured_rpm']:.1f} (use --rpm 360 for normalization and validation)")
        
        # Single global visualizations for entire disk
        global_base = str(run_dir / "entire_disk")
        global_analyzer.visualize(global_base, "intervals")
        global_analyzer.visualize(global_base, "histogram")
        if len(global_revs) > 1:
            global_analyzer.visualize(global_base, "heatmap")
        print("Global disk visualizations saved with prefix 'entire_disk_'")
        
        global_analysis = global_analyzer.analyze()
        
        # Global protection insights
        # Safe means, default 0 if empty
        protection_values = [entry['analysis'].get('protection_score', 0) for track in surface_map if track != 'global' for side in surface_map[track] for entry in surface_map[track][side] if isinstance(entry, dict) and 'analysis' in entry and 'protection_score' in entry['analysis']]
        max_density_values = [entry['analysis'].get('max_theoretical_density_bits_per_rev', 0) for track in surface_map if track != 'global' for side in surface_map[track] for entry in surface_map[track][side] if isinstance(entry, dict) and 'analysis' in entry and 'max_theoretical_density_bits_per_rev' in entry['analysis']]
        global_protection = np.mean(protection_values) if protection_values else 0.0
        global_max_density = np.mean(max_density_values) if max_density_values else 0
        
        side0_protection_values = [entry['analysis'].get('protection_score', 0) for track in surface_map if track != 'global' for entry in surface_map[track].get(0, []) if isinstance(entry, dict) and 'analysis' in entry and 'protection_score' in entry['analysis']]
        side1_protection_values = [entry['analysis'].get('protection_score', 0) for track in surface_map if track != 'global' for entry in surface_map[track].get(1, []) if isinstance(entry, dict) and 'analysis' in entry and 'protection_score' in entry['analysis']]
        side0_protection = np.mean(side0_protection_values) if side0_protection_values else 0.0
        side1_protection = np.mean(side1_protection_values) if side1_protection_values else 0.0
        side_diff = abs(side0_protection - side1_protection)
        
        # Side-specific globals (fix: read protection_score from entry['analysis'])
        side0_protection_values_fix = [entry.get('analysis', {}).get('protection_score', 0) for track in surface_map if track != 'global' for entry in surface_map[track].get(0, []) if isinstance(entry, dict) and 'analysis' in entry]
        side1_protection_values_fix = [entry.get('analysis', {}).get('protection_score', 0) for track in surface_map if track != 'global' for entry in surface_map[track].get(1, []) if isinstance(entry, dict) and 'analysis' in entry]
        side0_protection = np.mean(side0_protection_values_fix) if side0_protection_values_fix else 0.0
        side1_protection = np.mean(side1_protection_values_fix) if side1_protection_values_fix else 0.0
        side_diff = abs(side0_protection - side1_protection)
        
        # Save per-side heatmaps if >1 track
        if len(surface_map) > 2:  # >1 track + global
            side0_densities = [entry.get('analysis', {}).get('density_estimate_bits_per_rev', 0) for track in surface_map if track != 'global' for entry in surface_map[track].get(0, []) if 'analysis' in entry]
            side1_densities = [entry.get('analysis', {}).get('density_estimate_bits_per_rev', 0) for track in surface_map if track != 'global' for entry in surface_map[track].get(1, []) if 'analysis' in entry]
            if side0_densities and side1_densities:
                fig, axs = plt.subplots(1, 2, figsize=(12, 5))
                axs[0].bar(range(len(side0_densities)), side0_densities)
                axs[0].set_title('Side 0 Density per Track')
                axs[0].set_xlabel('Track')
                axs[0].set_ylabel('Bits per Rev')
                axs[1].bar(range(len(side1_densities)), side1_densities)
                axs[1].set_title('Side 1 Density per Track')
                axs[1].set_xlabel('Track')
                axs[1].set_ylabel('Bits per Rev')
                plt.tight_layout()
                plt.savefig(str(run_dir / 'side_density_heatmap.png'), dpi=150)
                plt.close()
                print("Side density comparison saved to side_density_heatmap.png")
            else:
                print("Skipping side heatmap: insufficient data on one or both sides")
        
        # Add global summary to surface_map
        surface_map['global'] = {
            'stats': global_stats,
            'analysis': global_analysis,
            'num_tracks': len(surface_map) - 1,
            'num_sides': sum(len(sides) for sides in surface_map.values()) - 1,
            'global_protection_score': float(global_protection),
            'global_max_density': int(global_max_density),
            'side0_protection': float(side0_protection),
            'side1_protection': float(side1_protection),
            'protection_side_diff': float(side_diff),
            'insights': {
                'likely_copy_protection_on_side': '1' if side1_protection > side0_protection + 0.1 else '0' if side0_protection > side1_protection + 0.1 else 'balanced',
                'packing_potential': f"Up to {global_max_density} bits/rev feasible with variable cells; current avg {global_analysis.get('density_estimate_bits_per_rev', 0)}; try density>1.5 in encode for protection-like schemes"
            }
        }
    else:
        print("No flux data overall")
    
    # Helper: render disk surface polar plots per side
    def _render_disk_surface(sm: dict, out_prefix: Path):
        def side_entries(track_obj, side_int):
            if not isinstance(track_obj, dict):
                return []
            if side_int in track_obj:
                return track_obj.get(side_int, [])
            return track_obj.get(str(side_int), [])
        max_track = max([int(k) for k in sm.keys() if k != 'global'], default=83)
        T = max(max_track+1, 1)
        # Build per-side radial arrays of average density
        radials = {}
        masks = {}
        for side in [0, 1]:
            radial = np.zeros(T)
            has = np.zeros(T, dtype=bool)
            for tk in sm:
                if tk == 'global':
                    continue
                try:
                    ti = int(tk)
                except Exception:
                    continue
                dens_vals = []
                for entry in side_entries(sm[tk], side):
                    if isinstance(entry, dict):
                        d = entry.get('analysis', {}).get('density_estimate_bits_per_rev')
                        if isinstance(d, (int, float)):
                            dens_vals.append(float(d))
                if dens_vals:
                    radial[ti] = float(np.mean(dens_vals))
                    has[ti] = True
            radials[side] = radial
            masks[side] = has
        if not (masks[0].any() or masks[1].any()):
            return
        theta = np.linspace(0, 2*np.pi, 360)
        r = np.arange(T)
        R, TH = np.meshgrid(r, theta, indexing='ij')
        fig, axs = plt.subplots(1, 2, subplot_kw={'projection': 'polar'}, figsize=(11,5))
        pcm = None
        for idx, side in enumerate([0, 1]):
            radial = radials[side]
            has = masks[side]
            if has.any():
                vmin = np.percentile(radial[has], 5)
                vmax = np.percentile(radial[has], 95)
                vmax = vmax if vmax > vmin else vmin + 1.0
                Z = np.repeat(radial[:, None], theta.shape[0], axis=1)
                pcm = axs[idx].pcolormesh(TH.T, R.T, Z.T, cmap='viridis', vmin=vmin, vmax=vmax, shading='auto')
                axs[idx].set_ylim(0, T)
                axs[idx].set_yticks([0, T//4, T//2, 3*T//4, T-1])
                axs[idx].set_yticklabels(["0", str(T//4), str(T//2), str(3*T//4), str(T-1)])
                axs[idx].set_title(f"Side {side}")
            else:
                axs[idx].set_title(f"Side {side} (no data)")
                axs[idx].set_ylim(0, T)
        if pcm is not None:
            cbar = fig.colorbar(pcm, ax=axs.ravel().tolist(), pad=0.08)
            cbar.set_label('Bits per Revolution')
        plt.tight_layout()
        plt.savefig(str(out_prefix) + "_disk_surface.png", dpi=150)
        plt.close()

    # Save map
    map_path = run_dir / "surface_map.json"
    with open(map_path, 'w') as f:
        json.dump(surface_map, f, indent=2, default=str)
    print(f"Surface map saved to {map_path}")
    # Render disk-surface plots
    try:
        # Derive a human-friendly label from input path (file stem or directory name)
        try:
            in_path = Path(args.input)
            label = in_path.stem if in_path.is_file() else in_path.name
        except Exception:
            label = 'disk'
        safe_label = re.sub(r'[^A-Za-z0-9_.-]', '_', label)
        _render_disk_surface(surface_map, run_dir / Path(f'{safe_label}_surface'))
        print("Disk surface plot saved (<label>_surface_disk_surface.png)")
                return track_obj.get(side_int, [])
            return track_obj.get(str(side_int), [])
        T = max([int(k) for k in surface_map.keys() if k != 'global'], default=83) + 1
        dens0 = np.full(T, np.nan)
        dens1 = np.full(T, np.nan)
        var0 = np.full(T, np.nan)
        var1 = np.full(T, np.nan)
        for tk in surface_map:
            if tk == 'global':
                continue
            try:
                ti = int(tk)
            except Exception:
                continue
            dvals0, dvals1, vvals0, vvals1 = [], [], [], []
            for e in side_entries(surface_map[tk], 0):
                if isinstance(e, dict):
                    d = e.get('analysis', {}).get('density_estimate_bits_per_rev')
                    v = e.get('analysis', {}).get('noise_profile', {}).get('avg_variance') if isinstance(e.get('analysis', {}), dict) else None
                    if isinstance(d, (int, float)): dvals0.append(float(d))
                    if isinstance(v, (int, float)): vvals0.append(float(v))
            for e in side_entries(surface_map[tk], 1):
                if isinstance(e, dict):
                    d = e.get('analysis', {}).get('density_estimate_bits_per_rev')
                    v = e.get('analysis', {}).get('noise_profile', {}).get('avg_variance') if isinstance(e.get('analysis', {}), dict) else None
                    if isinstance(d, (int, float)): dvals1.append(float(d))
                    if isinstance(v, (int, float)): vvals1.append(float(v))
            if dvals0: dens0[ti] = np.mean(dvals0)
            if dvals1: dens1[ti] = np.mean(dvals1)
            if vvals0: var0[ti] = np.mean(vvals0)
            if vvals1: var1[ti] = np.mean(vvals1)

        tracks = np.arange(T)
        # Density by track (both sides)
        plt.figure(figsize=(10,4))
        if np.isfinite(dens0).any():
            plt.plot(tracks, dens0, label='Side 0', color='steelblue')
        if np.isfinite(dens1).any():
            plt.plot(tracks, dens1, label='Side 1', color='indianred')
        plt.title(f'{label}: Density by Track')
        plt.xlabel('Track')
        plt.ylabel('Bits per Revolution')
        plt.legend()
        plt.tight_layout()
        plt.savefig(str(run_dir / f'{safe_label}_density_by_track.png'), dpi=150)
        plt.close()

        # Variance by track (both sides)
        if np.isfinite(var0).any() or np.isfinite(var1).any():
            plt.figure(figsize=(10,4))
            if np.isfinite(var0).any():
                plt.plot(tracks, var0, label='Side 0', color='steelblue')
            if np.isfinite(var1).any():
                plt.plot(tracks, var1, label='Side 1', color='indianred')
            plt.title(f'{label}: Noise Variance by Track')
            plt.xlabel('Track')
            plt.ylabel('Avg Variance')
            plt.legend()
            plt.tight_layout()
            plt.savefig(str(run_dir / f'{safe_label}_variance_by_track.png'), dpi=150)
            plt.close()

        # Keep outputs simple: removed top/bottom per-track bar charts
    except Exception as e:
        print(f"Disk surface plot failed: {e}")
    total_tracks = len(surface_map) - 1  # exclude global
    total_entries = sum(len(sides) for sides in surface_map.values()) - 1
    global_insights = surface_map['global'].get('insights', {})
    print(f"Analyzed {total_tracks} tracks across {total_entries} sides (found {found_total}/{expected_total})")
    print(f"Global insights: {global_insights}")

    # LLM Summary if requested
    if args.summarize:
        try:
            host_port = args.lm_host if ':' in args.lm_host else f"{args.lm_host}:1234"
            client = openai.OpenAI(
                base_url=f"http://{host_port}/v1",
                api_key="lm-studio"
            )
            # Craft prompt with key insights
            # Build richer aggregation for summary
            # Collect actual density estimates per side across entries
            side_densities = {0: [], 1: []}
            density_by_track = {0: [], 1: []}  # list of (track, density)
            protection_by_track = {0: [], 1: []}  # list of (track, score)
            rpm_by_side = {0: [], 1: []}
            side_counts = {0: 0, 1: 0}
            for track in surface_map:
                if track == 'global':
                    continue
                for side in [0, 1]:
                    for entry in surface_map.get(track, {}).get(side, []):
                        if isinstance(entry, dict):
                            if 'analysis' in entry and isinstance(entry['analysis'], dict):
                                dens = entry['analysis'].get('density_estimate_bits_per_rev')
                                if isinstance(dens, (int, float)):
                                    side_densities[side].append(float(dens))
                                    density_by_track[side].append((track, float(dens)))
                                pscore = entry['analysis'].get('protection_score')
                                if isinstance(pscore, (int, float)):
                                    protection_by_track[side].append((track, float(pscore)))
                            if 'stats' in entry and isinstance(entry['stats'], dict):
                                irpm = entry['stats'].get('inferred_rpm')
                                if isinstance(irpm, (int, float)):
                                    rpm_by_side[side].append(float(irpm))
                            # Count data entries (exclude marker dicts like side_summary)
                            if 'analysis' in entry and 'stats' in entry:
                                side_counts[side] += 1

            def density_stats(values):
                if not values:
                    return {"min": None, "max": None, "avg": None, "median": None, "std": None}
                return {
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "avg": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "std": float(np.std(values)),
                }

            topN = 5
            top_tracks = {
                0: sorted(protection_by_track[0], key=lambda t: t[1], reverse=True)[:topN],
                1: sorted(protection_by_track[1], key=lambda t: t[1], reverse=True)[:topN],
            }

            # Density top/bottom tracks per side
            density_top_tracks = {
                0: sorted(density_by_track[0], key=lambda t: t[1], reverse=True)[:topN],
                1: sorted(density_by_track[1], key=lambda t: t[1], reverse=True)[:topN],
            }
            density_bottom_tracks = {
                0: sorted(density_by_track[0], key=lambda t: t[1])[:topN],
                1: sorted(density_by_track[1], key=lambda t: t[1])[:topN],
            }

            # RPM stats and validity ratios (consider 200-500 RPM as valid window around 360 RPM)
            def rpm_stats(values):
                if not values:
                    return {"min": None, "max": None, "avg": None, "median": None, "std": None}
                return {
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "avg": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "std": float(np.std(values)),
                }

            def valid_ratio(values, lo=200.0, hi=500.0):
                if not values:
                    return 0.0
                arr = np.array(values)
                return float(np.mean((arr >= lo) & (arr <= hi)))

            global_stats = surface_map['global'].get('stats', {}) if isinstance(surface_map.get('global'), dict) else {}
            summary_data = {
                "counts": {
                    "found_total": found_total,
                    "expected_total": expected_total,
                    "num_tracks": total_tracks,
                    "per_side": {
                        "0": {"entries": side_counts[0], "expected": 84, "coverage_pct": float(100.0 * side_counts[0] / 84.0) if 84 else None},
                        "1": {"entries": side_counts[1], "expected": 84, "coverage_pct": float(100.0 * side_counts[1] / 84.0) if 84 else None},
                    }
                },
                "rpm": {
                    # Enforce constant RPM per user guidance
                    "measured": float(args.rpm) if getattr(args, 'rpm', None) else 360.0,
                    "drift_pct": 0.0,
                },
                "rpm_stats": {
                    "side0": {"min": float(args.rpm), "max": float(args.rpm), "avg": float(args.rpm), "median": float(args.rpm), "std": 0.0} if getattr(args, 'rpm', None) else {"min": 360.0, "max": 360.0, "avg": 360.0, "median": 360.0, "std": 0.0},
                    "side1": {"min": float(args.rpm), "max": float(args.rpm), "avg": float(args.rpm), "median": float(args.rpm), "std": 0.0} if getattr(args, 'rpm', None) else {"min": 360.0, "max": 360.0, "avg": 360.0, "median": 360.0, "std": 0.0},
                    "global": {"min": float(args.rpm), "max": float(args.rpm), "avg": float(args.rpm), "median": float(args.rpm), "std": 0.0} if getattr(args, 'rpm', None) else {"min": 360.0, "max": 360.0, "avg": 360.0, "median": 360.0, "std": 0.0},
                },
                "rpm_validity": {
                    "side0_ratio": 1.0,
                    "side1_ratio": 1.0,
                    "global_ratio": 1.0,
                    "window": [200.0, 500.0],
                },
                "protection": {
                    "global_score": surface_map['global'].get('global_protection_score', 0),
                    "side0_avg": surface_map['global'].get('side0_protection', 0),
                    "side1_avg": surface_map['global'].get('side1_protection', 0),
                    "side_diff": surface_map['global'].get('protection_side_diff', 0),
                    "top_tracks_by_side": {
                        "0": [{"track": t, "score": s} for t, s in top_tracks[0]],
                        "1": [{"track": t, "score": s} for t, s in top_tracks[1]],
                    },
                },
                "density": {
                    "side0": density_stats(side_densities[0]),
                    "side1": density_stats(side_densities[1]),
                    "max_theoretical_global": surface_map['global'].get('global_max_density', 0),
                    "top_tracks_by_side": {
                        "0": [{"track": t, "bits_per_rev": d} for t, d in density_top_tracks[0]],
                        "1": [{"track": t, "bits_per_rev": d} for t, d in density_top_tracks[1]],
                    },
                    "bottom_tracks_by_side": {
                        "0": [{"track": t, "bits_per_rev": d} for t, d in density_bottom_tracks[0]],
                        "1": [{"track": t, "bits_per_rev": d} for t, d in density_bottom_tracks[1]],
                    }
                },
                "insights": surface_map['global'].get('insights', {}),
            }

            # Strict JSON-only output prompt
            schema_description = (
                "Respond ONLY with a JSON object matching this schema: {\n"
                "  counts: { found_total: number, expected_total: number, num_tracks: number, per_side: { '0': {entries:number, expected:number, coverage_pct:number|null}, '1': {entries:number, expected:number, coverage_pct:number|null} } },\n"
                "  rpm: { measured: number|null, drift_pct: number|null },\n"
                "  rpm_stats: { side0: {min:number|null, max:number|null, avg:number|null, median:number|null, std:number|null}, side1: {min:number|null, max:number|null, avg:number|null, median:number|null, std:number|null}, global: {min:number|null, max:number|null, avg:number|null, median:number|null, std:number|null} },\n"
                "  rpm_validity: { side0_ratio:number, side1_ratio:number, global_ratio:number, window:[number, number] },\n"
                "  protection: { global_score: number, side0_avg: number, side1_avg: number, side_diff: number, top_tracks_by_side: { '0': [{track:number, score:number}], '1': [{track:number, score:number}] } },\n"
                "  density: { side0: {min:number|null, max:number|null, avg:number|null, median:number|null, std:number|null}, side1: {min:number|null, max:number|null, avg:number|null, median:number|null, std:number|null}, max_theoretical_global: number, top_tracks_by_side: { '0': [{track:number, bits_per_rev:number}], '1': [{track:number, bits_per_rev:number}] }, bottom_tracks_by_side: { '0': [{track:number, bits_per_rev:number}], '1': [{track:number, bits_per_rev:number}] } },\n"
                "  narrative: string  \n"
                "}. The 'narrative' must reference ONLY fields present in this JSON and must not introduce domain terms like sectors/cylinders/heads. No extra keys, no text outside JSON."
            )

            system_msg = (
                "You are an expert in floppy disk magnetic flux analysis."
                " Output strictly valid JSON per the given schema."
                " Do NOT include any extra text, code fences, explanations, or hidden reasoning."
                " If a value is unavailable, use null. Do not invent values."
                " The narrative must only reference keys present in the provided JSON data."
                " Avoid domain terms not present in the data (e.g., 'sectors', 'cylinders', 'heads')."
            )

            user_prompt = (
                "JSON schema requirements:\n" + schema_description + "\n\n"
                "Data:\n" + json.dumps(summary_data, indent=2)
            )

            def try_parse_json(text: str):
                # Strip possible code fences
                stripped = text.strip()
                if stripped.startswith("```"):
                    stripped = stripped.strip("`")
                    # After stripping backticks, try to find first '{'
                    brace_idx = stripped.find('{')
                    if brace_idx != -1:
                        stripped = stripped[brace_idx:]
                # Remove leading BOM or stray chars before first '{'
                first_brace = stripped.find('{')
                if first_brace > 0:
                    stripped = stripped[first_brace:]
                return json.loads(stripped)

            # First attempt
            response = client.chat.completions.create(
                model=args.lm_model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=500,
                temperature=getattr(args, 'lm_temperature', 0.2)
            )
            content = response.choices[0].message.content or ""

            parsed_json = None
            try:
                parsed_json = try_parse_json(content)
            except Exception:
                # Retry once with an explicit reminder
                retry_prompt = (
                    "Return ONLY valid JSON per the schema with no extra text. Do not use code fences.\n\n"
                    "Schema:\n" + schema_description + "\n\nData:\n" + json.dumps(summary_data, indent=2)
                )
                response2 = client.chat.completions.create(
                    model=args.lm_model,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": retry_prompt},
                    ],
                    max_tokens=500,
                    temperature=getattr(args, 'lm_temperature', 0.2)
                )
                content2 = response2.choices[0].message.content or ""
                try:
                    parsed_json = try_parse_json(content2)
                except Exception:
                    parsed_json = None

            # Fallback deterministic JSON if parsing failed
            if parsed_json is None:
                parsed_json = {
                    "counts": summary_data["counts"],
                    "rpm": summary_data["rpm"],
                    "rpm_stats": summary_data["rpm_stats"],
                    "rpm_validity": summary_data["rpm_validity"],
                    "protection": summary_data["protection"],
                    "density": summary_data["density"],
                    "narrative": (
                        "Deterministic summary: Processed {found}/{expected} files across {tracks} tracks. "
                        "RPM data validity (ratio within {wlo}-{whi} RPM): side0 {rv0:.2f}, side1 {rv1:.2f}, global {rvg:.2f}. "
                        "Protection averages — side0 {s0:.2f}, side1 {s1:.2f} (diff {sd:.2f}). "
                        "Density (bits/rev) — side0 avg {d0avg}, median {d0med}; side1 avg {d1avg}, median {d1med}."
                    ).format(
                        found=summary_data["counts"]["found_total"],
                        expected=summary_data["counts"]["expected_total"],
                        tracks=summary_data["counts"]["num_tracks"],
                        rv0=summary_data["rpm_validity"]["side0_ratio"],
                        rv1=summary_data["rpm_validity"]["side1_ratio"],
                        rvg=summary_data["rpm_validity"]["global_ratio"],
                        wlo=summary_data["rpm_validity"]["window"][0],
                        whi=summary_data["rpm_validity"]["window"][1],
                        s0=summary_data["protection"]["side0_avg"],
                        s1=summary_data["protection"]["side1_avg"],
                        sd=summary_data["protection"]["side_diff"],
                        d0avg=(None if summary_data["density"]["side0"]["avg"] is None else round(summary_data["density"]["side0"]["avg"], 1)),
                        d0med=(None if summary_data["density"]["side0"]["median"] is None else round(summary_data["density"]["side0"]["median"], 1)),
                        d1avg=(None if summary_data["density"]["side1"]["avg"] is None else round(summary_data["density"]["side1"]["avg"], 1)),
                        d1med=(None if summary_data["density"]["side1"]["median"] is None else round(summary_data["density"]["side1"]["median"], 1)),
                    )
                }

            # Save JSON (sanitize to avoid NaN/Inf and enforce strict JSON)
            def _sanitize(obj):
                if isinstance(obj, dict):
                    return {k: _sanitize(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [_sanitize(x) for x in obj]
                if isinstance(obj, float):
                    if np.isnan(obj) or np.isinf(obj):
                        return None
                    return float(obj)
                return obj

            parsed_json = _sanitize(parsed_json)
            json_path = run_dir / "llm_summary.json"
            with open(json_path, 'w') as jf:
                json.dump(parsed_json, jf, indent=2, allow_nan=False)

            # Render text if requested or default
            summary_path = run_dir / "llm_summary.txt"
            if getattr(args, 'summary_format', 'json') == 'text':
                narrative = parsed_json.get('narrative', '')
            else:
                # Default to generating text from JSON as well
                narrative = parsed_json.get('narrative', '')
            with open(summary_path, 'w') as f:
                f.write(f"FloppyAI LLM Summary - Generated on {datetime.datetime.now().isoformat()}\n\n")
                f.write(narrative)
            print(f"LLM summary saved to {summary_path} and {json_path}")
        except Exception as e:
            print(f"LLM summary failed (ensure LM Studio running on {args.lm_host}:1234 with model '{args.lm_model}' loaded): {e}")
            print("Skipping summary generation.")

    return 0

def plan_pool(args):
    """Select top-quality zones to form a dense bit pool based on density and noise."""
    run_dir = get_output_dir(args.output_dir)
    with open(args.input, 'r') as f:
        sm = json.load(f)
    min_density = args.min_density
    top_percent = args.top_percent

    candidates = {0: [], 1: []}  # (track, density, variance)
    def side_entries(track_obj, side_int):
        if not isinstance(track_obj, dict):
            return []
        if side_int in track_obj:
            return track_obj.get(side_int, [])
        return track_obj.get(str(side_int), [])

    for track in sm:
        if track == 'global':
            continue
        t_int = int(track)
        for side in [0, 1]:
            for entry in side_entries(sm.get(track, {}), side):
                if not isinstance(entry, dict):
                    continue
                dens = entry.get('analysis', {}).get('density_estimate_bits_per_rev')
                var = entry.get('analysis', {}).get('noise_profile', {}).get('avg_variance') if 'analysis' in entry else None
                if isinstance(dens, (int, float)) and dens >= min_density:
                    candidates[side].append((t_int, float(dens), float(var) if isinstance(var, (int, float)) else None))

    def pick_pool(items):
        if not items:
            return []
        # Rank primarily by density desc, secondarily by variance asc when available
        def key_fn(x):
            _, d, v = x
            return (-d, float('inf') if v is None else v)
        items_sorted = sorted(items, key=key_fn)
        k = max(1, int(len(items_sorted) * top_percent))
        return items_sorted[:k]

    pool0 = pick_pool(candidates[0])
    pool1 = pick_pool(candidates[1])

    plan = {
        'input': args.input,
        'criteria': {
            'min_density': min_density,
            'top_percent': top_percent,
        },
        'selected': {
            '0': [{'track': t, 'density': d, 'avg_variance': v} for t, d, v in pool0],
            '1': [{'track': t, 'density': d, 'avg_variance': v} for t, d, v in pool1],
        },
        'summary': {
            'side0_selected': len(pool0),
            'side1_selected': len(pool1),
        }
    }

    out_path = run_dir / 'pool_plan.json'
    with open(out_path, 'w') as f:
        json.dump(plan, f, indent=2)
    print(f"Pool plan saved to {out_path}")
    return 0

def main():
    parser = argparse.ArgumentParser(
        description="FloppyAI: Flux Stream Analysis and Custom Encoding Tool"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a .raw stream file")
    analyze_parser.add_argument("input", help="Path to .raw file")
    analyze_parser.add_argument("--output-dir", help="Custom output directory (default: test_outputs/timestamp/)")
    analyze_parser.set_defaults(func=analyze_stream)

    # Read command
    read_parser = subparsers.add_parser("read", help="Read track from hardware to .raw")
    read_parser.add_argument("track", type=int, help="Track number (0-79)")
    read_parser.add_argument("side", type=int, choices=[0, 1], help="Side (0 or 1)")
    read_parser.add_argument("--revs", type=int, default=3, dest="revolutions", help="Revolutions to read (default: 3)")
    read_parser.add_argument("--simulate", action="store_true", help="Simulate (no hardware)")
    read_parser.add_argument("--analyze", action="store_true", help="Analyze output after reading")
    read_parser.add_argument("--output-dir", help="Custom output directory (default: test_outputs/timestamp/)")
    read_parser.set_defaults(func=read_track)

    # Write command
    write_parser = subparsers.add_parser("write", help="Write .raw to hardware track")
    write_parser.add_argument("input", help="Input .raw file path")
    write_parser.add_argument("track", type=int, help="Track number")
    write_parser.add_argument("side", type=int, choices=[0, 1], help="Side")
    write_parser.add_argument("--simulate", action="store_true", help="Simulate (no hardware)")
    write_parser.add_argument("--output-dir", help="Custom output directory (default: test_outputs/timestamp/)")
    write_parser.set_defaults(func=write_track)

    # Generate dummy command
    gen_parser = subparsers.add_parser("generate", help="Generate dummy .raw stream")
    gen_parser.add_argument("track", type=int, help="Track number (for naming)")
    gen_parser.add_argument("side", type=int, choices=[0, 1], help="Side (for naming)")
    gen_parser.add_argument("--revs", type=int, default=1, dest="revolutions", help="Revolutions (default: 1)")
    gen_parser.add_argument("--cell", type=int, default=4000, dest="cell_length", help="Nominal cell length ns (default: 4000)")
    gen_parser.add_argument("--analyze", action="store_true", help="Analyze after generating")
    gen_parser.add_argument("--output-dir", help="Custom output directory (default: test_outputs/timestamp/)")
    gen_parser.set_defaults(func=generate_dummy)

    # Encode command
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

    # Analyze Disk command
    analyze_disk_parser = subparsers.add_parser("analyze_disk", help="Batch analyze .raw streams for disk surface map")
    analyze_disk_parser.add_argument("input", nargs='?', default="../example_stream_data/", help="Directory or single .raw file to analyze (default: ../example_stream_data/)")
    analyze_disk_parser.add_argument("--track", type=int, help="Manual track number if not parsable from filename")
    analyze_disk_parser.add_argument("--side", type=int, choices=[0, 1], help="Manual side number if not parsable from filename")
    analyze_disk_parser.add_argument("--rpm", type=int, default=360, help="Known RPM for validation (default: 360 for these dumps)")
    analyze_disk_parser.add_argument("--lm-host", default="localhost:1234", help="LM Studio host (IP or IP:port, default: localhost:1234)")
    analyze_disk_parser.add_argument("--lm-model", default="local-model", help="LM model name to use (default: local-model)")
    analyze_disk_parser.add_argument("--lm-temperature", type=float, default=0.2, dest="lm_temperature", help="Temperature for LLM summary generation (default: 0.2)")
    analyze_disk_parser.add_argument("--summarize", action="store_true", help="Generate LLM-powered human-readable summary report")
    analyze_disk_parser.add_argument("--summary-format", choices=["json", "text"], default="json", dest="summary_format", help="Summary output format: 'json' also writes llm_summary.json and renders narrative to txt (default), 'text' writes only txt")
    analyze_disk_parser.add_argument("--output-dir", help="Custom output directory (default: test_outputs/timestamp/)")
    analyze_disk_parser.set_defaults(func=analyze_disk)

    # Analyze Corpus command
    corpus_parser = subparsers.add_parser("analyze_corpus", help="Aggregate multiple surface_map.json files for a corpus summary")
    corpus_parser.add_argument("inputs", help="Directory containing runs (searched recursively for surface_map.json) or a single surface_map.json path")
    corpus_parser.add_argument("--output-dir", help="Custom output directory (default: test_outputs/timestamp/)")
    corpus_parser.add_argument("--generate-missing", action="store_true", dest="generate_missing", help="Scan for .raw under inputs, generate surface_map.json via analyze_disk where missing before aggregating")
    corpus_parser.add_argument("--rpm", type=int, default=360, help="Known RPM for normalization when generating missing maps (default: 360)")
    corpus_parser.add_argument("--summarize", action="store_true", help="Generate LLM-powered corpus summary report")
    corpus_parser.add_argument("--lm-host", default="localhost:1234", help="LM Studio host (IP or IP:port, default: localhost:1234)")
    corpus_parser.add_argument("--lm-model", default="local-model", help="LM model name to use (default: local-model)")
    corpus_parser.add_argument("--lm-temperature", type=float, default=0.2, dest="lm_temperature", help="Temperature for LLM corpus summary generation (default: 0.2)")
    corpus_parser.set_defaults(func=analyze_corpus)

    # Classify Surface command
    classify_parser = subparsers.add_parser("classify_surface", help="Classify blank-like vs written-like for a surface_map.json")
    classify_parser.add_argument("input", help="Path to surface_map.json")
    classify_parser.add_argument("--blank-density-thresh", type=float, default=1000.0, dest="blank_density_thresh", help="Density threshold below which an entry is considered blank-like (default: 1000)")
    classify_parser.add_argument("--output-dir", help="Custom output directory (default: test_outputs/timestamp/)")
    classify_parser.set_defaults(func=classify_surface)

    # Plan Pool command
    pool_parser = subparsers.add_parser("plan_pool", help="Select top-quality tracks to form a dense bit pool")
    pool_parser.add_argument("input", help="Path to surface_map.json")
    pool_parser.add_argument("--min-density", type=float, default=2000.0, dest="min_density", help="Minimum density (bits/rev) for candidate selection (default: 2000)")
    pool_parser.add_argument("--top-percent", type=float, default=0.2, dest="top_percent", help="Top percentile of candidates to keep (0-1, default: 0.2)")
    pool_parser.add_argument("--output-dir", help="Custom output directory (default: test_outputs/timestamp/)")
    pool_parser.set_defaults(func=plan_pool)
    
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 1
    
    return args.func(args)

if __name__ == "__main__":
    # For direct execution from src/ directory
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    sys.exit(main())