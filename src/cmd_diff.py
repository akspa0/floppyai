from pathlib import Path
import json
import numpy as np
from utils.io_paths import get_output_dir
from utils.json_io import dump_json

def compare_reads(args):
    """Compare multiple reads (>=2) of the same disk.
    Inputs may be surface_map.json paths or directories containing them.
    Outputs a summary JSON and per-track density CSV under run_dir/diff.
    """
    run_dir = get_output_dir(args.output_dir)
    diff_dir = run_dir / 'diff'
    diff_dir.mkdir(parents=True, exist_ok=True)

    # Resolve surface_map.json files from inputs
    maps = []  # list of (label, path, surface_map)
    for inp in args.inputs:
        p = Path(inp)
        sm_path = None
        if p.is_dir():
            hits = list(p.rglob('surface_map.json'))
            if hits:
                sm_path = hits[0]
        elif p.is_file() and p.name.endswith('.json'):
            sm_path = p
        if sm_path is None:
            print(f"No surface_map.json found under {inp}; skipping")
            continue
        try:
            with open(sm_path, 'r') as f:
                sm = json.load(f)
            label = p.name if p.is_dir() else p.parent.name
            maps.append((label, sm_path, sm))
            print(f"Loaded {sm_path}")
        except Exception as e:
            print(f"Failed to load {sm_path}: {e}")

    if len(maps) < 2:
        print("Need at least 2 inputs for comparison")
        return 1

    # Helper to gather per-track mean density per side from a surface_map
    def mean_density_by_track(sm: dict):
        res = {0: {}, 1: {}}
        for tk in sm:
            if tk == 'global':
                continue
            try:
                ti = int(tk)
            except Exception:
                continue
            for side in [0, 1]:
                vals = []
                for e in sm.get(tk, {}).get(side, []):
                    if isinstance(e, dict):
                        d = e.get('analysis', {}).get('density_estimate_bits_per_rev')
                        if isinstance(d, (int, float)):
                            vals.append(float(d))
                if vals:
                    res[side][ti] = float(np.mean(vals))
        return res

    # Helper to compute overlay info by side
    def overlay_info_by_side(sm: dict):
        ov = sm.get('global', {}).get('insights', {}).get('overlay', {}) if isinstance(sm.get('global'), dict) else {}
        bside = ov.get('by_side', {}) if isinstance(ov, dict) else {}
        out = {}
        for s in ['0','1']:
            info = bside.get(s, {})
            out[s] = {
                'sector_count': info.get('sector_count'),
                'boundaries_deg': info.get('boundaries_deg') or []
            }
        return out

    # Collect per-read data
    reads = []
    for label, path, sm in maps:
        reads.append({
            'label': label,
            'path': str(path),
            'dens': mean_density_by_track(sm),
            'overlay': overlay_info_by_side(sm),
        })

    # Compute intersection of tracks per side
    common_tracks = {0: None, 1: None}
    for side in [0, 1]:
        track_sets = []
        for r in reads:
            track_sets.append(set(r['dens'][side].keys()))
        common = set.intersection(*track_sets) if track_sets else set()
        common_tracks[side] = sorted(common)

    # Write per-track density table CSV
    import csv
    csv_path = diff_dir / 'diff_densities.csv'
    with open(csv_path, 'w', newline='') as cf:
        writer = csv.writer(cf)
        header = ['track','side'] + [f"{r['label']}_dens" for r in reads]
        writer.writerow(header)
        for side in [0, 1]:
            for ti in common_tracks[side]:
                row = [ti, side]
                for r in reads:
                    row.append(r['dens'][side].get(ti))
                writer.writerow(row)
    print(f"Per-track density diff CSV saved to {csv_path}")

    # Simple global stats per side for each read
    def stats(vals):
        if not vals:
            return {"min": None, "max": None, "avg": None, "median": None, "std": None}
        arr = np.array(vals, dtype=float)
        return {
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'avg': float(np.mean(arr)),
            'median': float(np.median(arr)),
            'std': float(np.std(arr)),
        }

    global_summary = { 'side0': [], 'side1': [] }
    for r in reads:
        for side_key, side in [('side0',0),('side1',1)]:
            vals = list(r['dens'][side].values())
            global_summary[side_key].append({ 'label': r['label'], **stats(vals) })

    # Overlay differences (global per side): sector counts and boundary alignment error
    def boundary_alignment_error(bases, comps):
        # Return minimal mean abs difference (deg) between two lists by circular shift
        if not bases or not comps or len(bases) != len(comps):
            return None
        bases = sorted(bases)
        comps = sorted(comps)
        n = len(bases)
        best = None
        for shift in range(n):
            errs = []
            for i in range(n):
                a = bases[i]
                b = comps[(i+shift) % n]
                d = abs(a - b) % 360.0
                d = min(d, 360.0 - d)
                errs.append(d)
            m = float(np.mean(errs))
            best = m if best is None else min(best, m)
        return best

    overlay_cmp = { 'side0': {}, 'side1': {} }
    for side_key, s in [('side0','0'),('side1','1')]:
        sector_counts = [r['overlay'][s].get('sector_count') for r in reads]
        overlay_cmp[side_key]['sector_counts'] = sector_counts
        # Pairwise alignment against first read as baseline
        base = reads[0]['overlay'][s]
        base_b = base.get('boundaries_deg') or []
        diffs = []
        for r in reads[1:]:
            comp_b = r['overlay'][s].get('boundaries_deg') or []
            diffs.append(boundary_alignment_error(base_b, comp_b))
        overlay_cmp[side_key]['boundary_avg_abs_diff_deg_vs_first'] = diffs

    summary = {
        'reads': [ {'label': r['label'], 'path': r['path']} for r in reads ],
        'global_density_stats': global_summary,
        'overlay_comparison': overlay_cmp,
        'common_tracks': { 'side0': common_tracks[0], 'side1': common_tracks[1] },
    }
    dump_json(diff_dir / 'diff_summary.json', summary)
    print(f"Diff summary saved to {diff_dir / 'diff_summary.json'}")
    return 0
