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
    render_single_track_detail,
    render_side_report,
)
from utils.json_io import dump_json


def run(args):
    """Analyze multiple .raw streams for a disk surface map.

    This function analyzes all .raw files in the input directory or processes a single .raw file,
    generating per-track/side analysis with surface visualizations and metrics.
    """
    run_dir = get_output_dir(args.output_dir)
    # Prepare run logging (collect and write at end)
    log_lines = []
    def log(msg: str):
        try:
            print(msg)
        except Exception:
            pass
        try:
            log_lines.append(str(msg))
        except Exception:
            pass
    input_path = Path(args.input)

    # Find all .raw files
    raw_files = []
    if input_path.is_dir():
        # Case-insensitive discovery: include *.raw and *.RAW
        raw_files = list(input_path.rglob("*.raw")) + list(input_path.rglob("*.RAW"))
        # Deduplicate and sort
        raw_files = sorted({p.resolve() for p in raw_files})
        if not raw_files:
            print(f"No .raw files found in {input_path}")
            return 1
    elif input_path.is_file() and input_path.suffix == ".raw":
        raw_files = [input_path]
    else:
        print(f"Input must be a directory containing .raw files or a single .raw file: {input_path}")
        return 1

    log(f"Found {len(raw_files)} .raw files to analyze")
    try:
        preview = [str(p) for p in raw_files[:5]]
        if len(raw_files) > 5:
            preview.append("...")
            preview.extend([str(p) for p in raw_files[-3:]])
        log("  Files preview:")
        for line in preview:
            log(f"    {line}")
    except Exception:
        pass

    # Resolve effective RPM (infer from profile or input path when possible)
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
    # Heuristic: infer 3.5" HD (300 RPM) from input path if profile not given
    inferred_rpm = None
    try:
        ip_str = str(Path(getattr(args, 'input', '')).as_posix()).lower()
        if any(k in ip_str for k in ['1.44', '35hd', '3.5']):
            inferred_rpm = 300.0
    except Exception:
        pass
    effective_rpm = (
        float(getattr(args, 'rpm', None)) if getattr(args, 'rpm', None) is not None
        else rpm_profile_map.get(profile, inferred_rpm if inferred_rpm is not None else 360.0)
    )

    # Global surface map structure
    surface_map = {
        'global': {
            'input_path': str(input_path),
            'effective_rpm': effective_rpm,
            'media_type': getattr(args, 'media_type', None),
            'analysis_timestamp': datetime.datetime.now().isoformat(),
            'insights': {
                'overlay': {}
            },
            # In directory mode, include manifest of inputs for transparency
            'inputs': [str(p) for p in raw_files] if input_path.is_dir() else [str(input_path)]
        }
    }

    # Populate overlay config with sensible defaults
    overlay_cfg = surface_map['global']['insights']['overlay']
    fmt_enabled = getattr(args, 'format_overlay', False)
    overlay_cfg['format_overlay_enabled'] = fmt_enabled
    overlay_cfg['angular_bins'] = getattr(args, 'angular_bins', 0)
    # If profile suggests MFM, default overlay mode to mfm
    default_overlay_mode = 'mfm' if (profile in ['35HD', '35DD', '525HD', '525DD'] or inferred_rpm == 300.0) else 'mfm'
    overlay_mode = getattr(args, 'overlay_mode', default_overlay_mode)
    overlay_cfg['overlay_mode'] = overlay_mode
    # Only include gcr_candidates for gcr/auto modes to avoid confusion on MFM media
    if overlay_mode in ['gcr', 'auto']:
        overlay_cfg['gcr_candidates'] = getattr(args, 'gcr_candidates', '10,12,8,9,11,13')

    # Process each raw file
    processed_count = 0
    for raw_file in sorted(raw_files):
        try:
            log(f"Processing {raw_file}...")

            # Extract track/side from filename if possible
            track_num = getattr(args, 'track', None)
            side_num = getattr(args, 'side', None)

            if track_num is None or side_num is None:
                # Try to parse from filename
                filename = raw_file.name
                # Supported patterns:
                #  - ...t<digits>...s<digit>... (existing)
                #  - NN.S.raw (e.g., 00.0.raw) from generate_disk
                track_match = re.search(r't(\d+)', filename, re.IGNORECASE)
                side_match = re.search(r's(\d)', filename, re.IGNORECASE)

                if track_match:
                    track_num = int(track_match.group(1))
                if side_match:
                    side_num = int(side_match.group(1))

                if track_num is None or side_num is None:
                    nn_s_match = re.match(r'^(\d{2})\.(\d)\.raw$', filename, re.IGNORECASE)
                    if nn_s_match:
                        track_num = int(nn_s_match.group(1))
                        side_num = int(nn_s_match.group(2))

                # Fallback: extract trailing NN.S before .raw anywhere in the name
                if track_num is None or side_num is None:
                    tail_match = re.search(r'(\d{1,2})\.(\d)\.raw$', filename, re.IGNORECASE)
                    if tail_match:
                        track_num = int(tail_match.group(1))
                        side_num = int(tail_match.group(2))

            if track_num is None or side_num is None:
                print(f"Warning: Could not determine track/side for {raw_file}, skipping")
                continue

            # Analyze the stream
            analyzer = FluxAnalyzer()
            parsed = analyzer.parse(str(raw_file))
            stats = parsed.get('stats', {}) if isinstance(parsed, dict) else {}
            if stats:
                log(f"  Parsed ok: total_fluxes={stats.get('total_fluxes')}, num_revolutions={stats.get('num_revolutions')}, mean_interval_ns={stats.get('mean_interval_ns'):.2f}")
                if 'decoder_sck_hz' in stats:
                    log(f"  decoder_sck_hz={stats.get('decoder_sck_hz'):.3f} Hz, decoder_oob_index_count={stats.get('decoder_oob_index_count', 0)}, decoder_total_samples={stats.get('decoder_total_samples', 0)}")

            if not parsed.get('stats', {}):
                log(f"Warning: No flux data in {raw_file}, skipping")
                continue

            # Perform analysis (compute angular histogram if bins specified)
            ang_bins = int(getattr(args, 'angular_bins', 0) or 0)
            analysis = analyzer.analyze(ang_bins if ang_bins > 0 else None)

            # Per-file plots disabled by default to reduce output volume and speed up runs.
            # If needed, we can gate this behind a CLI flag in the future.

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

            # Add overlay information if enabled (populate global.insights.overlay.by_side)
            if getattr(args, 'format_overlay', False):
                try:
                    bins = int(getattr(args, 'angular_bins', 720) or 720)
                    mode = str(getattr(args, 'overlay_mode', 'mfm')).lower()
                    files = [str(raw_file)]
                    k = None; bdeg = []; conf = 0.0
                    hint_k = getattr(args, 'overlay_sectors_hint', None)
                    if isinstance(hint_k, int) and hint_k and hint_k > 1:
                        # Use explicit hint: equally spaced boundaries
                        k = int(hint_k)
                        step = 360.0 / float(k)
                        bdeg = [step * i for i in range(k)]
                        conf = 1.0
                    elif mode == 'mfm':
                        k, bdeg, conf = detect_side_overlay_mfm(files, bins)
                    elif mode == 'gcr':
                        gc_raw = getattr(args, 'gcr_candidates', '10,12,8,9,11,13')
                        cand = []
                        for tok in str(gc_raw).replace(' ', '').split(','):
                            try:
                                cand.append(int(tok))
                            except Exception:
                                continue
                        k, bdeg, conf = detect_side_overlay_gcr(files, bins, cand)
                    elif mode == 'auto':
                        # Try MFM then GCR, keep higher confidence
                        k1, b1, c1 = detect_side_overlay_mfm(files, bins)
                        gc_raw = getattr(args, 'gcr_candidates', '10,12,8,9,11,13')
                        cand = []
                        for tok in str(gc_raw).replace(' ', '').split(','):
                            try:
                                cand.append(int(tok))
                            except Exception:
                                continue
                        k2, b2, c2 = detect_side_overlay_gcr(files, bins, cand)
                        if (c2 or 0) > (c1 or 0):
                            k, bdeg, conf = k2, b2, c2
                        else:
                            k, bdeg, conf = k1, b1, c1

                    if k is not None and bdeg:
                        overlay_cfg = surface_map['global']['insights']['overlay']
                        by_side = overlay_cfg.get('by_side', {})
                        entry = {
                            'sector_count': int(k),
                            'boundaries_deg': [float(x) for x in bdeg],
                            'confidence': float(conf),
                        }
                        if isinstance(hint_k, int) and hint_k and hint_k > 1:
                            entry['hint_used'] = True
                        by_side[str(side_num)] = entry
                        overlay_cfg['by_side'] = by_side
                except Exception as e:
                    print(f"Warning: Overlay detection failed for {raw_file}: {e}")

            surface_map[track_key][side_key] = side_data
            processed_count += 1

        except Exception as e:
            log(f"Error processing {raw_file}: {e}")
            continue

    if processed_count == 0:
        print("No files were successfully processed")
        return 1

    # Generate visualizations
    try:
        log("Generating visualizations...")

        # Output base prefix
        base_name = str(run_dir / Path(input_path.name).stem)

        # Create polar surface maps (pass args so overlay flags propagate)
        render_disk_surface(surface_map, base_name, args)
        # If only one track is present, render a single-track angular detail helper
        track_keys = [k for k in surface_map.keys() if k != 'global']
        if len(track_keys) == 1:
            render_single_track_detail(surface_map, base_name)

        # Build and render instability map
        # Compute simple instability scores from noise variance normalized per side
        instab_scores = {0: {}, 1: {}}
        max_track = 0
        side_vars = {0: [], 1: []}
        for tk in surface_map:
            if tk == 'global':
                continue
            try:
                ti = int(tk); max_track = max(max_track, ti)
            except Exception:
                continue
            for s in [0, 1]:
                entry_list = surface_map[tk].get(str(s), []) if isinstance(surface_map[tk], dict) else []
                # Normalize: if it's a dict, wrap into a list
                if isinstance(entry_list, dict):
                    entry_list = [entry_list]
                for ent in entry_list:
                    var = None
                    try:
                        var = ent.get('analysis', {}).get('noise_profile', {}).get('avg_variance')
                    except Exception:
                        var = None
                    if isinstance(var, (int, float)):
                        side_vars[s].append(float(var))
                        instab_scores[s][ti] = float(var)
        # Normalize per side
        for s in [0, 1]:
            vmax = max(side_vars[s]) if side_vars[s] else 1.0
            if vmax <= 0:
                vmax = 1.0
            for ti, v in list(instab_scores[s].items()):
                instab_scores[s][ti] = float(min(1.0, max(0.0, v / vmax)))

        render_instability_map(instab_scores, max_track + 1, base_name)

        # Per-side composite reports (one PNG per side)
        try:
            for s in [0, 1]:
                render_side_report(surface_map, instab_scores, s, base_name, args)
                log(f"Saved side report for side {s}")
        except Exception as e:
            log(f"Warning: Side report rendering failed: {e}")

        log(f"Visualizations saved in {run_dir}")

    except Exception as e:
        log(f"Warning: Visualization generation failed: {e}")

    # Save surface map
    surface_map_path = run_dir / 'surface_map.json'
    dump_json(surface_map_path, surface_map)
    log(f"Surface map saved to {surface_map_path}")

    # Optional LLM summary
    if getattr(args, 'summarize', False):
        try:
            log("Generating LLM summary...")
            summarize_disk_analysis(surface_map, run_dir, args)
        except Exception as e:
            log(f"Warning: LLM summary failed: {e}")

    # Write run log
    try:
        rl = run_dir / 'run.log'
        with open(rl, 'w', encoding='utf-8') as f:
            f.write("\n".join(log_lines) + "\n")
        print(f"Log saved to {rl}")
    except Exception:
        pass

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
