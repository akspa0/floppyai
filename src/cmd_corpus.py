import argparse
import datetime
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import openai

from utils.io_paths import get_output_dir
from utils.json_io import dump_json


def analyze_corpus(args):
    """Aggregate multiple surface_map.json files to produce a corpus-level summary.
    Also optionally generate missing per-disk surface maps by invoking analyze_disk.
    """
    run_dir = get_output_dir(args.output_dir)
    inputs = []
    base = Path(args.inputs)
    # Ensure dedicated corpus folder for all corpus-level outputs
    corpus_dir = run_dir / 'corpus'
    corpus_dir.mkdir(parents=True, exist_ok=True)

    # Resolve effective RPM from profile or explicit value
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
                # Invoke analyze_disk via subprocess to avoid import/package issues
                import sys as _sys
                import subprocess as _sp
                cmd = [
                    _sys.executable, "-m", "FloppyAI.src.main", "analyze_disk", str(d),
                    "--rpm", str(effective_rpm),
                    "--output-dir", str(disk_dir),
                ]
                # Media override
                mt = getattr(args, 'media_type', None)
                if mt:
                    cmd += ["--media-type", str(mt)]
                # LLM flags
                if getattr(args, 'summarize', False):
                    cmd += ["--summarize"]
                cmd += ["--lm-host", str(getattr(args, 'lm_host', 'localhost:1234'))]
                cmd += ["--lm-model", str(getattr(args, 'lm_model', 'local-model'))]
                cmd += ["--lm-temperature", str(getattr(args, 'lm_temperature', 0.2))]
                # Overlay flags propagated
                if getattr(args, 'format_overlay', False):
                    cmd += ["--format-overlay"]
                ab = getattr(args, 'angular_bins', 0)
                if ab:
                    cmd += ["--angular-bins", str(ab)]
                oc = getattr(args, 'overlay_color', None)
                if oc:
                    cmd += ["--overlay-color", str(oc)]
                oa = getattr(args, 'overlay_alpha', None)
                if oa is not None:
                    cmd += ["--overlay-alpha", str(oa)]
                om = getattr(args, 'overlay_mode', None)
                if om:
                    cmd += ["--overlay-mode", str(om)]
                gc = getattr(args, 'gcr_candidates', None)
                if gc:
                    cmd += ["--gcr-candidates", str(gc)]
                osh = getattr(args, 'overlay_sectors_hint', None)
                if osh:
                    cmd += ["--overlay-sectors-hint", str(osh)]
                try:
                    print("Running:", " ".join(cmd))
                    # Ensure module path is resolvable by setting CWD to repo root (parent of 'FloppyAI')
                    repo_root = Path(__file__).resolve().parents[2]
                    _sp.run(cmd, check=False, cwd=str(repo_root))
                except Exception as _e:
                    print(f"Failed to run analyze_disk for {d}: {_e}")
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

    dump_json(corpus_dir / 'corpus_summary.json', corpus_summary)
    print(f"Corpus summary saved to {corpus_dir / 'corpus_summary.json'}")

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

        # Density overlays per disk (approximate via normalized histograms)
        if any(len(x) for x in plot_side0_by_disk):
            plt.figure(figsize=(9,5))
            for dens, label in zip(plot_side0_by_disk, labels_side0):
                if not dens:
                    continue
                counts, bins = np.histogram(dens, bins=min(40, max(10, int(len(dens)/5))), density=True)
                centers = (bins[1:] + bins[:-1]) / 2.0
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
                _, axs = plt.subplots(1, 1, figsize=(7,4))
                axs.scatter(xs, ys, s=10, alpha=0.7, color='steelblue')
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
                    except Exception:
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
                dump_json(corpus_dir / 'llm_corpus_summary.json', parsed)
                with open(corpus_dir / 'llm_corpus_summary.txt', 'w') as tf:
                    tf.write(f"FloppyAI LLM Corpus Summary - Generated on {datetime.datetime.now().isoformat()}\n\n")
                    tf.write(parsed.get('narrative', ''))
                print(f"LLM corpus summary saved to {corpus_dir / 'llm_corpus_summary.txt'} and {corpus_dir / 'llm_corpus_summary.json'}")
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
                    max_tokens=300,
                    temperature=getattr(args, 'lm_temperature', 0.2),
                )
                parsed = try_parse_json(response2.choices[0].message.content or "{}")

            # Sanitize floats for strict JSON
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

            parsed_json = _sanitize(parsed)
            dump_json(corpus_dir / 'llm_corpus_summary.json', parsed_json, allow_nan=False)
            with open(corpus_dir / 'llm_corpus_summary.txt', 'w') as tf:
                tf.write(f"FloppyAI LLM Corpus Summary - Generated on {datetime.datetime.now().isoformat()}\n\n")
                tf.write(parsed_json.get('narrative', ''))
            print(f"LLM corpus summary saved to {corpus_dir / 'llm_corpus_summary.txt'} and {corpus_dir / 'llm_corpus_summary.json'}")
        except Exception as e:
            print(f"LLM corpus summary failed: {e}")
            print("Skipping corpus summary generation.")

    return 0
