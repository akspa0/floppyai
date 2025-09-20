#!/usr/bin/env python3
"""
FloppyAI Disk Analysis Module
Analyzes multiple .raw streams for a disk surface map.
"""
import datetime
import json
import re
from pathlib import Path
from typing import Dict, Any

import numpy as np

from flux_analyzer import FluxAnalyzer
from overlay_detection import (
    detect_side_overlay_mfm,
    detect_side_overlay_gcr,
)
from rendering import (
    render_disk_surface,
    render_instability_map,
)
from utils.json_io import dump_json


def run(args):
    """Analyze multiple .raw streams for a disk surface map.

    This function analyzes all .raw files in the input directory or processes a single .raw file,
    generating per-track/side analysis with surface visualizations and metrics.
    """
    run_dir = get_output_dir(args.output_dir)
    input_path = Path(args.input)

    # Find all .raw files
    raw_files = []
    if input_path.is_dir():
        raw_files = list(input_path.rglob("*.raw"))
        if not raw_files:
            print(f"No .raw files found in {input_path}")
            return 1
    elif input_path.is_file() and input_path.suffix == ".raw":
        raw_files = [input_path]
    else:
        print(f"Input must be a directory containing .raw files or a single .raw file: {input_path}")
        return 1

    print(f"Found {len(raw_files)} .raw files to analyze")

    # Resolve effective RPM
    rpm_profile_map = {
        '35HD': 300.0,   # 3.5" 1.44MB
        '35DD': 300.0,   # 3.5" 720KB
        '525HD': 360.0,  # 5.25" 1.2MB
        '525DD': 300.0,  # 5.25" 360KB
    }
    try:
        profile = getattr(args, 'profile', None)
    except Exception:
        profile = None
    effective_rpm = (
        float(getattr(args, 'rpm', None)) if getattr(args, 'rpm', None) is not None
        else rpm_profile_map.get(profile, 360.0)
    )

    # Global surface map structure
    surface_map = {
        'global': {
            'input_path': str(input_path),
            'effective_rpm': effective_rpm,
            'media_type': getattr(args, 'media_type', None),
            'analysis_timestamp': datetime.datetime.now().isoformat(),
            'insights': {
                'overlay': {
                    'format_overlay_enabled': getattr(args, 'format_overlay', False),
                    'angular_bins': getattr(args, 'angular_bins', 0),
                    'overlay_mode': getattr(args, 'overlay_mode', 'mfm'),
                    'gcr_candidates': getattr(args, 'gcr_candidates', '10,12,8,9,11,13'),
                }
            }
        }
    }

    # Process each raw file
    processed_count = 0
    for raw_file in sorted(raw_files):
        try:
            print(f"Processing {raw_file}...")

            # Extract track/side from filename if possible
            track_num = getattr(args, 'track', None)
            side_num = getattr(args, 'side', None)

            if track_num is None or side_num is None:
                # Try to parse from filename
                filename = raw_file.name
                track_match = re.search(r't(\d+)', filename, re.IGNORECASE)
                side_match = re.search(r's(\d)', filename, re.IGNORECASE)

                if track_match:
                    track_num = int(track_match.group(1))
                if side_match:
                    side_num = int(side_match.group(1))

            if track_num is None or side_num is None:
                print(f"Warning: Could not determine track/side for {raw_file}, skipping")
                continue

            # Analyze the stream
            analyzer = FluxAnalyzer()
            parsed = analyzer.parse(str(raw_file))

            if not parsed.get('stats', {}):
                print(f"Warning: No flux data in {raw_file}, skipping")
                continue

            # Perform analysis
            analysis = analyzer.analyze()

            # Create track entry
            track_key = str(track_num)
            if track_key not in surface_map:
                surface_map[track_key] = {}

            side_key = str(side_num)
            side_data = {
                'file': str(raw_file),
                'track': track_num,
                'side': side_num,
                'analysis': analysis,
                'stats': parsed.get('stats', {})
            }

            # Add overlay information if enabled
            if getattr(args, 'format_overlay', False):
                try:
                    if getattr(args, 'overlay_mode', 'mfm') == 'mfm':
                        overlay = detect_side_overlay_mfm(
                            analyzer, angular_bins=getattr(args, 'angular_bins', 0)
                        )
                    else:  # gcr mode
                        overlay = detect_side_overlay_gcr(
                            analyzer, angular_bins=getattr(args, 'angular_bins', 0)
                        )

                    if overlay:
                        side_data['overlay'] = overlay
                except Exception as e:
                    print(f"Warning: Overlay detection failed for {raw_file}: {e}")

            surface_map[track_key][side_key] = side_data
            processed_count += 1

        except Exception as e:
            print(f"Error processing {raw_file}: {e}")
            continue

    if processed_count == 0:
        print("No files were successfully processed")
        return 1

    # Generate visualizations
    try:
        print("Generating visualizations...")

        # Create composite report
        base_name = str(run_dir / input_path.name)
        render_disk_surface(surface_map, base_name, "composite_report")

        # Create polar surface maps
        render_disk_surface(surface_map, base_name, "surface_disk_surface")

        # Create individual side maps
        for side in [0, 1]:
            side_name = f"side{side}"
            render_disk_surface(surface_map, base_name, f"surface_{side_name}")

        # Create instability map
        render_instability_map(surface_map, base_name)

        print(f"Visualizations saved in {run_dir}")

    except Exception as e:
        print(f"Warning: Visualization generation failed: {e}")

    # Save surface map
    surface_map_path = run_dir / 'surface_map.json'
    dump_json(surface_map_path, surface_map)
    print(f"Surface map saved to {surface_map_path}")

    # Optional LLM summary
    if getattr(args, 'summarize', False):
        try:
            print("Generating LLM summary...")
            summarize_disk_analysis(surface_map, run_dir, args)
        except Exception as e:
            print(f"Warning: LLM summary failed: {e}")

    return 0


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


def summarize_disk_analysis(surface_map, output_dir, args):
    """Generate LLM-powered summary of disk analysis results."""
    try:
        import openai

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


if __name__ == "__main__":
    import sys
    print("This module is meant to be imported, not run directly.")
    sys.exit(1)
