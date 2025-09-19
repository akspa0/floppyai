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
        if re.search(r'(\d+)\.(\d+)\.raw$', filename):
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
        
        # Side-specific globals
        side0_protection = np.mean([a.get('protection_score', 0) for track in surface_map if track != 'global' for entry in surface_map[track].get(0, []) if isinstance(entry, dict) and 'protection_score' in entry])
        side1_protection = np.mean([a.get('protection_score', 0) for track in surface_map if track != 'global' for entry in surface_map[track].get(1, []) if isinstance(entry, dict) and 'protection_score' in entry])
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
    
    # Save map
    map_path = run_dir / "surface_map.json"
    with open(map_path, 'w') as f:
        json.dump(surface_map, f, indent=2, default=str)
    print(f"Surface map saved to {map_path}")
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
            protection_by_track = {0: [], 1: []}  # list of (track, score)
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
                                pscore = entry['analysis'].get('protection_score')
                                if isinstance(pscore, (int, float)):
                                    protection_by_track[side].append((track, float(pscore)))

            def density_stats(values):
                if not values:
                    return {"min": None, "max": None, "avg": None}
                return {
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "avg": float(np.mean(values)),
                }

            topN = 5
            top_tracks = {
                0: sorted(protection_by_track[0], key=lambda t: t[1], reverse=True)[:topN],
                1: sorted(protection_by_track[1], key=lambda t: t[1], reverse=True)[:topN],
            }

            global_stats = surface_map['global'].get('stats', {}) if isinstance(surface_map.get('global'), dict) else {}
            summary_data = {
                "counts": {
                    "found_total": found_total,
                    "expected_total": expected_total,
                    "num_tracks": total_tracks,
                },
                "rpm": {
                    "measured": global_stats.get('measured_rpm'),
                    "drift_pct": global_stats.get('rpm_drift_pct'),
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
                },
                "insights": surface_map['global'].get('insights', {}),
            }

            # Strict JSON-only output prompt
            schema_description = (
                "Respond ONLY with a JSON object matching this schema: {\n"
                "  counts: { found_total: number, expected_total: number, num_tracks: number },\n"
                "  rpm: { measured: number|null, drift_pct: number|null },\n"
                "  protection: { global_score: number, side0_avg: number, side1_avg: number, side_diff: number, top_tracks_by_side: { '0': [{track:number, score:number}], '1': [{track:number, score:number}] } },\n"
                "  density: { side0: {min:number|null, max:number|null, avg:number|null}, side1: {min:number|null, max:number|null, avg:number|null}, max_theoretical_global: number },\n"
                "  narrative: string  \n"
                "}. The 'narrative' field must be a concise 150-250 word factual summary derived ONLY from provided numbers. No extra keys, no text outside JSON."
            )

            system_msg = (
                "You are an expert in floppy disk magnetic flux analysis."
                " Output strictly valid JSON per the given schema."
                " Do NOT include any extra text, code fences, explanations, or hidden reasoning."
                " If a value is unavailable, use null. Do not invent values."
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
                    "protection": summary_data["protection"],
                    "density": summary_data["density"],
                    "narrative": (
                        "Deterministic summary: Analyzed {found}/{expected} files across {tracks} tracks. "
                        "Measured RPM ~{rpm:.1f} with drift {drift}%. Protection averages — side0 {s0:.2f}, side1 {s1:.2f} (diff {sd:.2f}). "
                        "Density (bits/rev) — side0 avg {d0avg}, side1 avg {d1avg}."
                    ).format(
                        found=summary_data["counts"]["found_total"],
                        expected=summary_data["counts"]["expected_total"],
                        tracks=summary_data["counts"]["num_tracks"],
                        rpm=summary_data["rpm"].get("measured") or 0.0,
                        drift=(summary_data["rpm"].get("drift_pct") if summary_data["rpm"].get("drift_pct") is not None else 0.0),
                        s0=summary_data["protection"]["side0_avg"],
                        s1=summary_data["protection"]["side1_avg"],
                        sd=summary_data["protection"]["side_diff"],
                        d0avg=(None if summary_data["density"]["side0"]["avg"] is None else round(summary_data["density"]["side0"]["avg"], 1)),
                        d1avg=(None if summary_data["density"]["side1"]["avg"] is None else round(summary_data["density"]["side1"]["avg"], 1)),
                    )
                }

            # Save JSON
            json_path = run_dir / "llm_summary.json"
            with open(json_path, 'w') as jf:
                json.dump(parsed_json, jf, indent=2)

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