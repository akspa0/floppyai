from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def resolve_render_params(args):
    """Resolve rendering fidelity parameters from args with sensible defaults.

    Returns (theta_samples, radial_upsample, dpi)
    """
    # Presets
    quality = str(getattr(args, 'quality', 'ultra') or 'ultra').lower()
    if quality == 'normal':
        theta_samples = 1440
        radial_upsample = 4
        dpi = 180
    elif quality == 'high':
        theta_samples = 2880
        radial_upsample = 6
        dpi = 220
    else:  # ultra
        theta_samples = 4096
        radial_upsample = 8
        dpi = 260

    # Overrides
    ts = getattr(args, 'theta_samples', None)
    if isinstance(ts, int) and ts > 0:
        theta_samples = ts
    ru = getattr(args, 'radial_upsample', None)
    if isinstance(ru, int) and ru > 0:
        radial_upsample = ru
    rd = getattr(args, 'render_dpi', None)
    if isinstance(rd, int) and rd > 0:
        dpi = rd
    return int(theta_samples), int(radial_upsample), int(dpi)


def _theta_offset_for_side(sm: dict, args, side: int) -> float:
    """Return radians to rotate axis so first detected sector boundary is at 0°.
    Uses overlay insights if available. Honors args.align_to_sectors.
    """
    try:
        mode = str(getattr(args, 'align_to_sectors', 'off') or 'off').lower()
    except Exception:
        mode = 'off'
    if mode == 'off':
        return 0.0
    # Fallback strategy: we only have side-level boundaries; track-level uses side-level too.
    try:
        overlay = sm.get('global', {}).get('insights', {}).get('overlay', {})
        by_side = overlay.get('by_side', {}) if isinstance(overlay, dict) else {}
        bdeg = by_side.get(str(side), {}).get('boundaries_deg', []) if isinstance(by_side, dict) else []
        if isinstance(bdeg, list) and bdeg:
            first = float(bdeg[0])
            return float(-np.deg2rad(first))
    except Exception:
        pass
    # Heuristic fallback: align strongest angular peak to 0° if periodicity is present
    try:
        if mode in ('auto', 'side'):
            hist, bins = _aggregate_side_hist(sm, side)
            if hist is not None and isinstance(bins, int) and bins > 0:
                H = np.abs(np.fft.rfft(hist))
                if H.size > 1:
                    H[0] = 0.0
                median_spec = float(np.median(H[1:])) if H.size > 1 else 1.0
                k_best = int(np.argmax(H)) if H.size > 0 else 0
                v_best = float(H[k_best]) if H.size > 0 else 0.0
                ratio = (v_best / max(1e-9, median_spec)) if median_spec > 0 else 0.0
                if ratio >= 1.3:
                    peak_bin = int(np.argmax(hist))
                    peak_theta = (peak_bin + 0.5) * (2 * np.pi / float(bins))
                    return float(-peak_theta)
    except Exception:
        pass
    return 0.0


def _aggregate_side_hist(sm: dict, side: int):
    """Aggregate angular_hist across tracks for a side. Returns (hist, bins) or (None, 0)."""
    try:
        bins = 0
        for tk in sm:
            if tk == 'global':
                continue
            lst = sm[tk].get(str(side), []) if isinstance(sm[tk], dict) else []
            if isinstance(lst, dict):
                lst = [lst]
            if not lst:
                continue
            ent = lst[0]
            b = ent.get('analysis', {}).get('angular_bins') if isinstance(ent, dict) else None
            if isinstance(b, int) and b > bins:
                bins = b
        if not bins:
            return None, 0
        agg = np.zeros(bins, dtype=float)
        used = 0
        for tk in sm:
            if tk == 'global':
                continue
            lst = sm[tk].get(str(side), []) if isinstance(sm[tk], dict) else []
            if isinstance(lst, dict):
                lst = [lst]
            if not lst:
                continue
            ent = lst[0]
            ah = ent.get('analysis', {}).get('angular_hist') if isinstance(ent, dict) else None
            b = ent.get('analysis', {}).get('angular_bins') if isinstance(ent, dict) else None
            if isinstance(ah, list) and isinstance(b, int) and b == bins and b > 0:
                agg[:b] += np.array(ah[:b], dtype=float)
                used += 1
        if used > 0 and np.max(agg) > 0:
            agg = agg / float(np.max(agg))
        return (agg if used > 0 else None), bins
    except Exception:
        return None, 0


def _intra_wedge_peak_angles(hist: np.ndarray, bins: int, wedge_angles: list[float]) -> list[float]:
    """For each wedge interval between successive wedge_angles, return the angle of the max hist bin.
    Wedge angles should be sorted in [0, 2pi)."""
    if hist is None or not isinstance(bins, int) or bins <= 0 or not wedge_angles:
        return []
    th_centers = (np.arange(bins) + 0.5) * (2 * np.pi / float(bins))
    peaks = []
    k = len(wedge_angles)
    for i in range(k):
        a0 = wedge_angles[i]
        a1 = wedge_angles[(i + 1) % k]
        if a1 <= a0:
            a1 += 2 * np.pi
        # Select centers in [a0, a1)
        mask = (th_centers >= a0) & (th_centers < a1)
        # Also consider wrap by adding 2pi to centers < a0
        centers_ext = np.where(th_centers < a0, th_centers + 2 * np.pi, th_centers)
        mask = (centers_ext >= a0) & (centers_ext < a1)
        if not np.any(mask):
            peaks.append((a0 + a1) * 0.5 % (2 * np.pi))
            continue
        idx = np.argmax(hist[mask])
        sel_angles = centers_ext[mask]
        th = float(sel_angles[idx])
        peaks.append(th % (2 * np.pi))
    return peaks


def _draw_intra_wedge_peaks(ax, T: int, peaks_rad: list[float], color: str = '#ffaa00', alpha: float = 0.9):
    """Draw small tick markers near the outer radius at intra-wedge peak angles."""
    try:
        if not peaks_rad:
            return
        r = max(1.0, T - 1) * 0.95
        ax.scatter(peaks_rad, [r] * len(peaks_rad), s=6, c=color, alpha=alpha, marker='|', linewidths=0.8)
    except Exception:
        pass


def _formatted_badge(sm: dict, side: int) -> str:
    """Build a short badge string for titles based on formatted detection insights."""
    try:
        insights = sm.get('global', {}).get('insights', {})
        fmt = insights.get('formatted', {}) if isinstance(insights, dict) else {}
        entry = fmt.get('by_side', {}).get(str(side), {}) if isinstance(fmt, dict) else {}
        if not isinstance(entry, dict):
            return ''
        if entry.get('formatted') is True:
            mode = entry.get('mode', 'mfm')
            k = entry.get('sector_count')
            conf = entry.get('confidence')
            part_k = f", k={int(k)}" if isinstance(k, int) else ""
            part_c = f", conf={conf:.2f}" if isinstance(conf, (int, float)) else ""
            return f"Formatted ({mode}{part_k}{part_c})"
        if entry.get('formatted') is False:
            conf = entry.get('confidence')
            part_c = f" conf={conf:.2f}" if isinstance(conf, (int, float)) else ""
            return f"Unformatted{part_c}"
    except Exception:
        pass
    return 'Format: unknown'


def _wedge_angles_for_side(sm: dict, args, side: int):
    """Compute wedge boundary angles (radians) for a side.
    Priority: overlay boundaries -> formatted sector_count with equal spacing.
    Returns (angles: list[float], k: int|None)
    """
    try:
        insights = sm.get('global', {}).get('insights', {})
        # Prefer overlay boundaries if present
        ov = insights.get('overlay', {}) if isinstance(insights, dict) else {}
        by_side = ov.get('by_side', {}) if isinstance(ov, dict) else {}
        bdeg = by_side.get(str(side), {}).get('boundaries_deg', []) if isinstance(by_side, dict) else []
        if isinstance(bdeg, list) and bdeg:
            return [np.deg2rad(float(x)) for x in bdeg], len(bdeg)
        # Fallback to formatted k
        fmt = insights.get('formatted', {}) if isinstance(insights, dict) else {}
        entry = fmt.get('by_side', {}).get(str(side), {}) if isinstance(fmt, dict) else {}
        k = entry.get('sector_count') if isinstance(entry, dict) else None
        if isinstance(k, int) and k >= 2:
            step = (2 * np.pi) / float(k)
            # With axis offset applied separately, 0, step, 2*step ... are fine
            return [m * step for m in range(k)], k
    except Exception:
        pass
    return [], None


def _draw_wedges(ax, T: int, angles_rad: list[float], label: bool, color: str = '#ff3333', alpha: float = 0.6):
    """Draw radial wedge lines and optional labels on a polar axis."""
    try:
        for i, th in enumerate(angles_rad):
            ax.plot([th, th], [0, T], color=color, alpha=alpha, linewidth=0.9)
        if label and len(angles_rad) >= 2:
            # Label at mid-angle between consecutive boundaries
            k = len(angles_rad)
            for i in range(k):
                th0 = angles_rad[i]
                th1 = angles_rad[(i + 1) % k]
                # Handle wrap-around
                if th1 < th0:
                    th1 += 2 * np.pi
                th_mid = (th0 + th1) * 0.5
                th_mid = (th_mid + 2 * np.pi) % (2 * np.pi)
                ax.text(th_mid, max(0.0, T - 1) * 0.92, f"{i+1}", ha='center', va='center', fontsize=7, color=color, alpha=alpha)
    except Exception:
        pass


def render_disk_surface(sm: dict, out_prefix: Path, args) -> None:
    """Render combined polar disk surface and per-side overlays using data in sm.
    - sm: surface_map dict with 'global' and per-track entries
    - out_prefix: Path prefix (without suffix)
    - args: argparse Namespace providing overlay flags (format_overlay, overlay_alpha, overlay_color)
    """
    def side_entries(track_obj, side_int):
        if not isinstance(track_obj, dict):
            return []
        # Prefer int key then string key
        val = None
        if side_int in track_obj:
            val = track_obj.get(side_int, [])
        else:
            val = track_obj.get(str(side_int), [])
        # Normalize to list of entries
        if isinstance(val, dict):
            return [val]
        return val if isinstance(val, list) else []

    try:
        max_track = max([int(k) for k in sm.keys() if k != 'global'], default=83)
    except Exception:
        max_track = 83
    T = max(max_track + 1, 1)

    radials = {}
    masks = {}
    counts = {}
    for side in [0, 1]:
        radial = np.zeros(T)
        has = np.zeros(T, dtype=bool)
        c = 0
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
                c += 1
        radials[side] = radial
        masks[side] = has
        counts[side] = c

    # Compute a shared color scale across both sides
    vals = []
    for s in [0, 1]:
        if masks[s].any():
            vals.extend(radials[s][masks[s]].tolist())
    if len(vals) >= 2:
        vmin = float(np.percentile(vals, 5))
        vmax = float(np.percentile(vals, 95))
        if vmax <= vmin:
            vmax = vmin + 1.0
    else:
        vmin, vmax = 0.0, 1.0

    # Resolve fidelity
    theta_samples, radial_upsample, dpi = resolve_render_params(args)

    # Upsample radial resolution for smoother rendering
    # Use [0, T] so a single track (T=1) still has non-zero radial extent
    up_factor = radial_upsample
    Tu = max(T * up_factor, T)
    r_up = np.linspace(0, T, Tu)

    def upsample(radial, has):
        if has.sum() >= 2:
            xs = np.where(has)[0]
            ys = radial[has]
            return np.interp(r_up, xs, ys)
        fill = float(radial[has][0]) if has.any() else 0.0
        return np.full_like(r_up, fill, dtype=float)

    radial_up = {s: upsample(radials[s], masks[s]) for s in [0, 1]}

    # Build high-fidelity grids
    theta = np.linspace(0, 2 * np.pi, int(theta_samples))
    TH, R = np.meshgrid(theta, r_up)

    # Build angular templates per track per side from aggregated angular histograms
    def interp_hist_to_theta(hist: np.ndarray, bins: int) -> np.ndarray:
        th_centers = (np.arange(bins) + 0.5) * (2 * np.pi / float(bins))
        # Wrap theta to [0, 2pi)
        th = theta % (2 * np.pi)
        # Interpolate with wrap-around by extending
        x = np.concatenate([th_centers - 2 * np.pi, th_centers, th_centers + 2 * np.pi])
        y = np.concatenate([hist, hist, hist])
        return np.interp(th, x, y)

    def build_side_templates(side: int):
        templates = [None] * T
        bins_max = 0
        # First pass: find max bins available
        for tk in sm:
            if tk == 'global':
                continue
            try:
                ti = int(tk)
            except Exception:
                continue
            lst = sm[tk].get(str(side), []) if isinstance(sm[tk], dict) else []
            if isinstance(lst, dict):
                lst = [lst]
            if not lst:
                continue
            ent = lst[0]
            b = ent.get('analysis', {}).get('angular_bins') if isinstance(ent, dict) else None
            if isinstance(b, int) and b > bins_max:
                bins_max = b
        # Second pass: build per-track template
        for tk in sm:
            if tk == 'global':
                continue
            try:
                ti = int(tk)
            except Exception:
                continue
            lst = sm[tk].get(str(side), []) if isinstance(sm[tk], dict) else []
            if isinstance(lst, dict):
                lst = [lst]
            template = np.ones_like(theta)
            if lst:
                ent = lst[0]
                ah = ent.get('analysis', {}).get('angular_hist') if isinstance(ent, dict) else None
                b = ent.get('analysis', {}).get('angular_bins') if isinstance(ent, dict) else None
                if isinstance(ah, list) and isinstance(b, int) and b > 1:
                    h = np.array(ah[:b], dtype=float)
                    m = float(np.max(h))
                    if m > 0:
                        h = h / m
                    # Per-track alignment: roll histogram so its dominant peak maps to 0° if requested
                    try:
                        mode = str(getattr(args, 'align_to_sectors', 'off') or 'off').lower()
                    except Exception:
                        mode = 'off'
                    if mode == 'track':
                        peak = int(np.argmax(h))
                        if b > 0:
                            h = np.roll(h, -peak)
                    template = interp_hist_to_theta(h, b)
            templates[ti] = template
        # Fill any None with ones
        for i in range(T):
            if templates[i] is None:
                templates[i] = np.ones_like(theta)
        return templates

    templates = {s: build_side_templates(s) for s in [0, 1]}
    # Normalize densities per side for angular-weighted visualization
    dens_norm = {}
    for s in [0, 1]:
        den = radials[s]
        m = float(np.max(den[masks[s]])) if masks[s].any() else 0.0
        if m <= 0:
            dens_norm[s] = np.zeros_like(den)
        else:
            dens_norm[s] = den / m

    # Layout: side0 | side1 | colorbar
    fig = plt.figure(figsize=(13, 5.5))
    gs = GridSpec(1, 3, width_ratios=[1, 1, 0.05], figure=fig)
    ax0 = fig.add_subplot(gs[0, 0], projection='polar')
    ax1 = fig.add_subplot(gs[0, 1], projection='polar')
    cax = fig.add_subplot(gs[0, 2])

    pcm = None
    for ax, side in [(ax0, 0), (ax1, 1)]:
        # Apply sector alignment offset if requested
        try:
            off = _theta_offset_for_side(sm, args, side)
            if abs(off) > 1e-9:
                ax.set_theta_offset(off)
        except Exception:
            pass
        if counts[side] > 0 and masks[side].any():
            # Build angular-weighted Z using nearest-track template × normalized density
            Z = np.zeros((r_up.shape[0], theta.shape[0]), dtype=float)
            for ri, rv in enumerate(r_up):
                ti = int(np.clip(int(round(rv)), 0, T - 1))
                Z[ri, :] = templates[side][ti] * float(dens_norm[side][ti])
            pcm = ax.pcolormesh(TH, R, Z, cmap='viridis', vmin=0.0, vmax=1.0, shading='auto')
            ax.set_ylim(0, T)
            ax.set_yticks([0, T // 4, T // 2, 3 * T // 4, T - 1])
            ax.set_yticklabels(["0", str(T // 4), str(T // 2), str(3 * T // 4), str(T - 1)])
            badge = _formatted_badge(sm, side)
            ax.set_title(f"Side {side} – {badge}")
            # Draw wedge spokes and optional labels
            try:
                color = getattr(args, 'overlay_color', '#ff3333')
                alpha = float(getattr(args, 'overlay_alpha', 0.8))
                label = bool(getattr(args, 'label_sectors', False))
                angles, _k = _wedge_angles_for_side(sm, args, side)
                if angles:
                    _draw_wedges(ax, T, angles, label, color=color, alpha=alpha)
                    # Intra-wedge peak ticks from aggregated angular histogram
                    hist, bins = _aggregate_side_hist(sm, side)
                    peaks = _intra_wedge_peak_angles(hist, bins, angles)
                    _draw_intra_wedge_peaks(ax, T, peaks)
            except Exception:
                pass
        else:
            ax.set_title(f"Side {side} (no data)")
            ax.set_ylim(0, T)
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center')

    if pcm is not None:
        cbar = fig.colorbar(pcm, cax=cax, orientation='vertical')
        cbar.set_label('Angular-Weighted Density (norm)')

    plt.tight_layout()
    # Support export format (png|svg|both); default png
    fmt = str(getattr(args, 'export_format', 'png') or 'png').lower()
    base = Path(str(out_prefix) + "_disk_surface")
    try:
        if fmt in ('png', 'both'):
            plt.savefig(str(base.with_suffix('.png')), dpi=dpi)
        if fmt in ('svg', 'both'):
            # Rasterize pcolormesh where possible
            try:
                for ax in [ax0, ax1]:
                    for coll in ax.collections:
                        try:
                            coll.set_rasterized(True)
                        except Exception:
                            pass
            except Exception:
                pass
            plt.savefig(str(base.with_suffix('.svg')), dpi=dpi, format='svg')
    finally:
        plt.close()


def render_side_report(sm: dict, instab_scores: dict, side: int, out_prefix: Path, args) -> None:
    """Render a single composite report for one side.
    Layout:
      - Top: Polar disk surface for this side (radial = density per track), with optional overlay sector lines
      - Middle: Instability polar for this side (from instab_scores)
      - Bottom: (L) Track vs Density scatter, (R) Density histogram
    """
    # Helpers
    def side_entries(track_obj, side_int):
        if not isinstance(track_obj, dict):
            return []
        v = track_obj.get(side_int)
        if v is None:
            v = track_obj.get(str(side_int))
        if isinstance(v, dict):
            return [v]
        return v if isinstance(v, list) else []

    try:
        max_track = max([int(k) for k in sm.keys() if k != 'global'], default=83)
    except Exception:
        max_track = 83
    T = max(max_track + 1, 1)

    # Densities per track for this side
    radial = np.zeros(T)
    has = np.zeros(T, dtype=bool)
    dens_all = []
    tvd_x = []
    tvd_y = []
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
            dmean = float(np.mean(dens_vals))
            radial[ti] = dmean
            has[ti] = True
            dens_all.append(dmean)
            tvd_x.append(ti)
            tvd_y.append(dmean)

    if not has.any():
        # Nothing to plot for this side
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(1, 1, 1)
        ax.axis('off')
        ax.text(0.5, 0.5, f"Side {side}: no data", ha='center', va='center')
        plt.tight_layout()
        fmt = str(getattr(args, 'export_format', 'png') or 'png').lower()
        base = Path(str(out_prefix) + f"_side{side}_report")
        # Resolve dpi for consistent export
        try:
            _, _, dpi = resolve_render_params(args)
        except Exception:
            dpi = 180
        try:
            if fmt in ('png', 'both'):
                plt.savefig(str(base.with_suffix('.png')), dpi=dpi)
            if fmt in ('svg', 'both'):
                plt.savefig(str(base.with_suffix('.svg')), dpi=dpi, format='svg')
        finally:
            plt.close()
        return

    # Color scale from available densities
    vmin = float(np.percentile(dens_all, 5)) if len(dens_all) >= 2 else float(min(dens_all + [0.0]))
    vmax = float(np.percentile(dens_all, 95)) if len(dens_all) >= 2 else float(max(dens_all + [1.0]))
    if vmax <= vmin:
        vmax = vmin + 1.0

    # Resolve fidelity
    theta_samples, radial_upsample, dpi = resolve_render_params(args)

    # Upsample radial
    up_factor = radial_upsample
    Tu = max(T * up_factor, T)
    r_up = np.linspace(0, T, Tu)
    if has.sum() >= 2:
        xs = np.where(has)[0]
        ys = radial[has]
        radial_up = np.interp(r_up, xs, ys)
    else:
        fill = float(radial[has][0]) if has.any() else 0.0
        radial_up = np.full_like(r_up, fill, dtype=float)
    theta = np.linspace(0, 2 * np.pi, int(theta_samples))
    TH, R = np.meshgrid(theta, r_up)

    # Overlay lines for this side
    thetas_overlay = []
    try:
        ov_enabled = bool(getattr(args, 'format_overlay', False))
        overlay_info = sm.get('global', {}).get('insights', {}).get('overlay', {}) if isinstance(sm.get('global'), dict) else {}
        by_side = overlay_info.get('by_side', {}) if isinstance(overlay_info, dict) else {}
        bdeg = by_side.get(str(side), {}).get('boundaries_deg', [])
        thetas_overlay = [np.deg2rad(x) for x in bdeg] if ov_enabled and bdeg else []
    except Exception:
        thetas_overlay = []

    # Prepare angular heatmap (track x angle) and aggregated interval histogram for this side
    # Collect angular histograms by track
    ang_bins = 0
    for tk in sm:
        if tk == 'global':
            continue
        entry_list = side_entries(sm[tk], side)
        if not entry_list:
            continue
        ent = entry_list[0] if isinstance(entry_list, list) else entry_list
        bins = None
        try:
            bins = ent.get('analysis', {}).get('angular_bins')
        except Exception:
            bins = None
        if isinstance(bins, int) and bins > ang_bins:
            ang_bins = bins
    # Build heatmap matrix T x ang_bins
    heatmap = None
    if ang_bins and ang_bins > 0:
        heatmap = np.zeros((T, ang_bins), dtype=float)
        present_rows = np.zeros(T, dtype=bool)
        for tk in sm:
            if tk == 'global':
                continue
            try:
                ti = int(tk)
            except Exception:
                continue
            entry_list = side_entries(sm[tk], side)
            if not entry_list:
                continue
            ent = entry_list[0] if isinstance(entry_list, list) else entry_list
            ah = None
            try:
                ah = ent.get('analysis', {}).get('angular_hist')
            except Exception:
                ah = None
            if isinstance(ah, list) and len(ah) == ang_bins:
                heatmap[ti, :] = np.array(ah, dtype=float)
                present_rows[ti] = True
        # Mask absent rows to NaN to render as blank
        for i in range(T):
            if not present_rows[i]:
                heatmap[i, :] = np.nan

    # Aggregated interval histogram across tracks for this side (average of normalized histograms)
    agg_hist = None; agg_edges = None
    acc = None; count = 0
    first_edges = None
    for tk in sm:
        if tk == 'global':
            continue
        entry_list = side_entries(sm[tk], side)
        if not entry_list:
            continue
        ent = entry_list[0] if isinstance(entry_list, list) else entry_list
        ih = None; ih_bins = None; ih_range = None
        try:
            an = ent.get('analysis', {})
            ih = an.get('interval_hist')
            ih_bins = an.get('interval_hist_bins')
            ih_range = an.get('interval_hist_range_ns')
        except Exception:
            ih = None
        if isinstance(ih, list) and isinstance(ih_bins, int) and ih_bins > 0 and isinstance(ih_range, list) and len(ih_range) == 2:
            if first_edges is None:
                mn, mx = float(ih_range[0]), float(ih_range[1])
                first_edges = np.logspace(np.log10(max(mn, 1.0)), np.log10(max(mx, mn + 1.0)), ih_bins + 1)
            vec = np.array(ih, dtype=float)
            if acc is None:
                acc = vec.copy()
            else:
                # Average later to keep scale 0..1
                acc += vec
            count += 1
    if acc is not None and count > 0:
        agg_hist = (acc / float(count))
        agg_edges = first_edges

    # Compose figure
    fig = plt.figure(figsize=(12, 10))
    # Grid: 3 rows, 2 columns (top polar across 2 columns; middle-left instability polar, middle-right heatmap; bottom-left track vs density; bottom-right interval histogram)
    gs = GridSpec(3, 2, height_ratios=[2.0, 1.6, 1.2], width_ratios=[1.0, 1.0], figure=fig)
    # Top: surface polar
    ax_surf = fig.add_subplot(gs[0, :], projection='polar')
    # Apply sector alignment and formatted badge
    try:
        off_side = _theta_offset_for_side(sm, args, side)
        if abs(off_side) > 1e-9:
            ax_surf.set_theta_offset(off_side)
    except Exception:
        pass
    # Build angular-aware template per track for this side
    def _template_for_track(ti: int):
        try:
            lst = sm.get(str(ti), {}).get(str(side), []) if isinstance(sm.get(str(ti)), dict) else []
            if isinstance(lst, dict):
                lst = [lst]
            if lst:
                ent = lst[0]
                ah = ent.get('analysis', {}).get('angular_hist') if isinstance(ent, dict) else None
                b = ent.get('analysis', {}).get('angular_bins') if isinstance(ent, dict) else None
                if isinstance(ah, list) and isinstance(b, int) and b > 1:
                    h = np.array(ah[:b], dtype=float)
                    m = float(np.max(h))
                    if m > 0:
                        h = h / m
                    # Interpolate onto theta grid
                    th_centers = (np.arange(b) + 0.5) * (2 * np.pi / float(b))
                    x = np.concatenate([th_centers - 2*np.pi, th_centers, th_centers + 2*np.pi])
                    y = np.concatenate([h, h, h])
                    return np.interp(theta % (2*np.pi), x, y)
        except Exception:
            pass
        return np.ones_like(theta)

    # Normalize density per track for this side
    dmax = float(np.max(radial[has])) if has.any() else 0.0
    dnorm = (radial / dmax) if dmax > 0 else np.zeros_like(radial)
    # Build angular-weighted Z
    Z = np.zeros((r_up.shape[0], theta.shape[0]), dtype=float)
    for ri, rv in enumerate(r_up):
        ti = int(np.clip(int(round(rv)), 0, T - 1))
        tpl = _template_for_track(ti)
        Z[ri, :] = tpl * float(dnorm[ti])
    pcm = ax_surf.pcolormesh(TH, R, Z, cmap='viridis', vmin=0.0, vmax=1.0, shading='auto')
    ax_surf.set_ylim(0, T)
    ax_surf.set_yticks([0, T // 4, T // 2, 3 * T // 4, T - 1])
    ax_surf.set_yticklabels(["0", str(T // 4), str(T // 2), str(3 * T // 4), str(T - 1)])
    badge = _formatted_badge(sm, side)
    ax_surf.set_title(f"Side {side} – {badge} – Angular-Weighted Density")
    if thetas_overlay:
        ov_color = getattr(args, 'overlay_color', '#ff3333')
        ov_alpha = float(getattr(args, 'overlay_alpha', 0.8))
        for th in thetas_overlay:
            ax_surf.plot([th, th], [0, T], color=ov_color, alpha=ov_alpha, linewidth=0.9)
    # If no explicit overlay, attempt wedge lines/labels and intra-wedge ticks
    try:
        if not thetas_overlay:
            color = getattr(args, 'overlay_color', '#ff3333')
            alpha = float(getattr(args, 'overlay_alpha', 0.8))
            label = bool(getattr(args, 'label_sectors', False))
            angles, _k = _wedge_angles_for_side(sm, args, side)
            if angles:
                _draw_wedges(ax_surf, T, angles, label, color=color, alpha=alpha)
                hist, bins = _aggregate_side_hist(sm, side)
                peaks = _intra_wedge_peak_angles(hist, bins, angles)
                _draw_intra_wedge_peaks(ax_surf, T, peaks)
    except Exception:
        pass
    cbar_sr = fig.colorbar(pcm, ax=ax_surf, orientation='vertical', pad=0.1)
    try:
        cbar_sr.set_label('Angular-Weighted Density (norm)')
    except Exception:
        pass
    # Rasterize heavy mesh for SVG friendliness while keeping labels/ticks vector
    try:
        pcm.set_rasterized(True)
    except Exception:
        pass

    # Middle-left: instability polar for this side
    ax_inst = fig.add_subplot(gs[1, 0], projection='polar')
    try:
        if abs(off_side) > 1e-9:
            ax_inst.set_theta_offset(off_side)
    except Exception:
        pass
    # Build instability radial for this side
    inst_rad = np.zeros(T)
    for ti in range(T):
        inst_rad[ti] = float(instab_scores.get(side, {}).get(ti, 0.0))
    Zr = np.interp(r_up, np.arange(T), inst_rad)
    th_prof_theta, th_prof = _aggregate_instability_theta(sm, side)
    if th_prof is not None:
        prof = np.interp(theta % (2*np.pi), th_prof_theta, th_prof)
    else:
        prof = np.ones_like(theta)
    Z2 = np.outer(Zr, prof)
    Z2 = _contrast_stretch(Z2, pct=97.0)
    pcm2 = ax_inst.pcolormesh(TH, R, Z2, cmap='magma_r', vmin=0.0, vmax=1.0, shading='auto')
    ax_inst.set_ylim(0, T)
    ax_inst.set_yticks([0, T // 4, T // 2, 3 * T // 4, T - 1])
    ax_inst.set_yticklabels(["0", str(T // 4), str(T // 2), str(3 * T // 4), str(T - 1)])
    ax_inst.set_title(f"Side {side} – Instability")
    try:
        pcm2.set_rasterized(True)
    except Exception:
        pass

    # Middle-right: angular heatmap (track x angle)
    ax_hm = fig.add_subplot(gs[1, 1])
    if heatmap is not None:
        im = ax_hm.imshow(heatmap, aspect='auto', origin='lower', interpolation='nearest', cmap='viridis')
        ax_hm.set_xlabel('Angle bin')
        ax_hm.set_ylabel('Track')
        ax_hm.set_title('Angular Activity (track × angle)')
        plt.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04)
        try:
            im.set_rasterized(True)
        except Exception:
            pass
    else:
        ax_hm.axis('off')
        ax_hm.text(0.5, 0.5, 'No angular histograms', ha='center', va='center')

    # Bottom left: Track vs Density scatter
    ax_sc = fig.add_subplot(gs[2, 0])
    if tvd_x and tvd_y:
        ax_sc.scatter(tvd_x, tvd_y, s=10, alpha=0.8, color='steelblue')
    ax_sc.set_xlabel('Track')
    ax_sc.set_ylabel('Bits per Revolution')
    ax_sc.set_title('Track vs Density')

    # Bottom right: Aggregated interval histogram (log ns)
    ax_h = fig.add_subplot(gs[2, 1])
    if agg_hist is not None and agg_edges is not None:
        centers = np.sqrt(agg_edges[:-1] * agg_edges[1:])
        ax_h.bar(centers, agg_hist, width=np.diff(agg_edges), align='center', alpha=0.85, color='steelblue')
        ax_h.set_xscale('log')
        ax_h.set_xlim(agg_edges[0], agg_edges[-1])
        ax_h.set_xlabel('Interval (ns, log scale)')
        ax_h.set_ylabel('Normalized density')
        ax_h.set_title('Flux Interval Histogram (aggregated)')
    else:
        ax_h.axis('off')
        ax_h.text(0.5, 0.5, 'No interval histogram', ha='center', va='center')

    plt.tight_layout()
    # Export format control: default png; allow svg or both via args.export_format
    fmt = str(getattr(args, 'export_format', 'png') or 'png').lower()
    base = Path(str(out_prefix) + f"_side{side}_report")
    try:
        if fmt in ('png', 'both'):
            plt.savefig(str(base.with_suffix('.png')), dpi=dpi)
        if fmt in ('svg', 'both'):
            plt.savefig(str(base.with_suffix('.svg')), dpi=dpi, format='svg')
    finally:
        plt.close()


def render_single_track_detail(sm: dict, out_prefix: Path) -> None:
    """Render a single-track polar histogram using angular_hist (if present).
    Falls back to a neutral notice if histogram is unavailable.
    Saves to <out_prefix>_single_track_detail.png.
    """
    # Find the only track present
    track_keys = [k for k in sm.keys() if k != 'global']
    if len(track_keys) != 1:
        return
    tk = track_keys[0]
    side0 = sm[tk].get('0') if isinstance(sm[tk], dict) else None
    side1 = sm[tk].get('1') if isinstance(sm[tk], dict) else None

    # Prefer side0 then side1
    entry = None
    side = 0
    if isinstance(side0, dict):
        entry = side0
        side = 0
    elif isinstance(side1, dict):
        entry = side1
        side = 1
    elif isinstance(side0, list) and side0:
        entry = side0[0]
        side = 0
    elif isinstance(side1, list) and side1:
        entry = side1[0]
        side = 1

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(1, 1, 1, projection='polar')
    title = f"Track {tk}, Side {side}"
    try:
        ah = entry.get('analysis', {}).get('angular_hist') if isinstance(entry, dict) else None
        bins = entry.get('analysis', {}).get('angular_bins') if isinstance(entry, dict) else None
        if ah and bins:
            ah = np.array(ah, dtype=float)
            theta = np.linspace(0, 2 * np.pi, int(bins), endpoint=False)
            r = np.ones_like(theta)
            # Build a simple colored ring with intensity modulated by angular histogram
            TH, R = np.meshgrid(np.linspace(0, 2 * np.pi, 1440), np.linspace(0.0, 1.0, 80))
            # Interp hist onto fine theta grid
            ah_fine = np.interp((TH[0] % (2*np.pi)), theta, ah)
            Z = np.repeat(ah_fine[np.newaxis, :], R.shape[0], axis=0)
            pcm = ax.pcolormesh(TH, R, Z, cmap='viridis', vmin=0.0, vmax=1.0, shading='auto')
            ax.set_yticks([0.0, 0.5, 1.0])
            ax.set_yticklabels(["0","","1.0"])
            ax.set_title(title + " – angular activity")
            cbar = fig.colorbar(pcm, ax=ax, orientation='vertical', pad=0.1)
            cbar.set_label('Normalized Angular Density')
        else:
            ax.set_title(title + " – no angular histogram available")
            ax.text(0.5, 0.5, 'No angular histogram', transform=ax.transAxes, ha='center', va='center')
    except Exception as e:
        ax.set_title(title + " – error rendering")
        ax.text(0.5, 0.5, str(e), transform=ax.transAxes, ha='center', va='center')
    plt.tight_layout()
    outfile = Path(str(out_prefix) + "_single_track_detail.png")
    plt.savefig(str(outfile), dpi=220)
    plt.close()

    


def _aggregate_instability_theta(sm: dict, side: int):
    """Aggregate angular-resolved instability profiles across tracks for a side.
    Returns (theta_samples array, profile array in 0..1) or (None, None) if unavailable."""
    try:
        bins = 0
        # Determine maximum angular_bins present
        for tk in sm:
            if tk == 'global':
                continue
            v = sm[tk]
            lst = v.get(str(side), []) if isinstance(v, dict) else []
            if isinstance(lst, dict):
                lst = [lst]
            if not lst:
                continue
            ent = lst[0]
            b = ent.get('analysis', {}).get('angular_bins') if isinstance(ent, dict) else None
            if isinstance(b, int) and b > bins:
                bins = b
        if not bins:
            return None, None
        acc = np.zeros(bins, dtype=float)
        used = 0
        for tk in sm:
            if tk == 'global':
                continue
            v = sm[tk]
            lst = v.get(str(side), []) if isinstance(v, dict) else []
            if isinstance(lst, dict):
                lst = [lst]
            if not lst:
                continue
            ent = lst[0]
            prof = ent.get('analysis', {}).get('instability_theta') if isinstance(ent, dict) else None
            ab = ent.get('analysis', {}).get('angular_bins') if isinstance(ent, dict) else None
            if isinstance(prof, list) and isinstance(ab, int) and ab == bins and len(prof) == bins:
                acc += np.array(prof, dtype=float)
                used += 1
        if used == 0:
            return None, None
        acc = acc / float(np.max(acc)) if np.max(acc) > 0 else acc
        theta = np.linspace(0, 2 * np.pi, bins, endpoint=False)
        return theta, acc
    except Exception:
        return None, None


def _contrast_stretch(Z: np.ndarray, pct: float = 95.0) -> np.ndarray:
    """Apply simple percentile-based contrast stretch and clamp to 0..1.
    Uses the given percentile of positive/finite values as the 1.0 reference.
    """
    try:
        finite = Z[np.isfinite(Z)]
        vec = finite[finite > 0]
        if vec.size == 0:
            vec = finite
        p = float(np.percentile(vec, pct)) if vec.size else 0.0
        if p > 0.0:
            Z = Z / p
        return np.clip(Z, 0.0, 1.0)
    except Exception:
        return np.clip(Z, 0.0, 1.0)


def render_instability_map(sm: dict, instab_scores: dict, T: int, out_prefix: Path) -> None:
    """Render an instability polar map for both sides.
    Intensity is angular-resolved when available: Z(r,theta) = radial_instab(t) * side_profile(theta)."""
    # Build per-side radial arrays
    radials = {}
    for side in [0, 1]:
        radial = np.zeros(T)
        for ti in range(T):
            radial[ti] = float(instab_scores.get(side, {}).get(ti, 0.0))
        radials[side] = radial
    vmin, vmax = 0.0, 1.0

    # Upsample radial for smoother rings
    up_factor = 4
    Tu = max(T * up_factor, T)
    r_up = np.linspace(0, T - 1, Tu)
    theta = np.linspace(0, 2 * np.pi, 1440)
    TH, R = np.meshgrid(theta, r_up)

    fig = plt.figure(figsize=(12, 5))
    gs = GridSpec(1, 3, width_ratios=[1, 1, 0.05], figure=fig)
    ax0 = fig.add_subplot(gs[0, 0], projection='polar')
    ax1 = fig.add_subplot(gs[0, 1], projection='polar')
    cax = fig.add_subplot(gs[0, 2])
    pcm = None
    for ax, side in [(ax0, 0), (ax1, 1)]:
        Zr = np.interp(r_up, np.arange(T), radials[side])
        # Angular profile for this side (fallback to ones)
        th_prof_theta, th_prof = _aggregate_instability_theta(sm, side)
        if th_prof is not None:
            # Interpolate onto theta grid
            prof = np.interp(theta % (2*np.pi), th_prof_theta, th_prof)
        else:
            prof = np.ones_like(theta)
        Z = np.outer(Zr, prof)
        Z = _contrast_stretch(Z, pct=97.0)
        pcm = ax.pcolormesh(TH, R, Z, cmap='magma_r', vmin=vmin, vmax=vmax, shading='auto')
        ax.set_ylim(0, T)
        ax.set_yticks([0, T // 4, T // 2, 3 * T // 4, T - 1])
        ax.set_yticklabels(["0", str(T // 4), str(T // 2), str(3 * T // 4), str(T - 1)])
        ax.set_title(f"Side {side}")
    if pcm is not None:
        cbar = plt.colorbar(pcm, cax=cax, orientation='vertical')
        cbar.set_label('Instability Score (0-1)')
    plt.tight_layout()
    # Add rasterization for SVG friendliness
    try:
        for ax in [ax0, ax1]:
            for coll in ax.collections:
                try:
                    coll.set_rasterized(True)
                except Exception:
                    pass
    except Exception:
        pass
    # Backward compatibility: no args here, so default to PNG only
    outfile = Path(str(out_prefix) + "_instability_map.png")
    plt.savefig(str(outfile), dpi=220)
    plt.close()


def render_disk_dashboard(sm: dict, instab_scores: dict, out_prefix: Path, args) -> None:
    """Render a whole-disk dashboard combining both sides:
    Row1: Side0 density polar | Side1 density polar
    Row2: Side0 instability   | Side1 instability
    Row3: Side0 ang heatmap   | Side1 ang heatmap
    Row4: Side0 interval hist | Side1 interval hist
    """
    # Helper to gather per-side data
    def side_entries(track_obj, side_int):
        if not isinstance(track_obj, dict):
            return []
        v = track_obj.get(side_int)
        if v is None:
            v = track_obj.get(str(side_int))
        if isinstance(v, dict):
            return [v]
        return v if isinstance(v, list) else []

    try:
        max_track = max([int(k) for k in sm.keys() if k != 'global'], default=83)
    except Exception:
        max_track = 83
    T = max(max_track + 1, 1)

    # Resolve fidelity
    theta_samples, radial_upsample, dpi = resolve_render_params(args)

    # Common angular grid
    theta = np.linspace(0, 2 * np.pi, int(theta_samples))

    # Build density radials and color scale across sides
    radials = {}
    masks = {}
    dens_vals_all = []
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
            dvals = []
            for ent in side_entries(sm[tk], side):
                if isinstance(ent, dict):
                    d = ent.get('analysis', {}).get('density_estimate_bits_per_rev')
                    if isinstance(d, (int, float)):
                        dvals.append(float(d))
            if dvals:
                val = float(np.mean(dvals))
                radial[ti] = val
                has[ti] = True
                dens_vals_all.append(val)
        radials[side] = radial
        masks[side] = has

    if len(dens_vals_all) >= 2:
        vmin = float(np.percentile(dens_vals_all, 5))
        vmax = float(np.percentile(dens_vals_all, 95))
        if vmax <= vmin:
            vmax = vmin + 1.0
    else:
        vmin, vmax = 0.0, 1.0

    # Upsample radial
    up_factor = radial_upsample
    Tu = max(T * up_factor, T)
    r_up = np.linspace(0, T, Tu)
    def upsample(radial, has):
        if has.sum() >= 2:
            xs = np.where(has)[0]
            ys = radial[has]
            return np.interp(r_up, xs, ys)
        fill = float(radial[has][0]) if has.any() else 0.0
        return np.full_like(r_up, fill, dtype=float)
    radial_up = {s: upsample(radials[s], masks[s]) for s in [0, 1]}
    TH, R = np.meshgrid(theta, r_up)

    # Build angular heatmaps and interval histograms per side
    def build_ang_heat_and_hist(side: int):
        # Determine ang_bins
        ang_bins = 0
        for tk in sm:
            if tk == 'global':
                continue
            lst = side_entries(sm[tk], side)
            if not lst:
                continue
            ent = lst[0] if isinstance(lst, list) else lst
            bins = ent.get('analysis', {}).get('angular_bins') if isinstance(ent, dict) else None
            if isinstance(bins, int) and bins > ang_bins:
                ang_bins = bins
        heatmap = None
        if ang_bins and ang_bins > 0:
            heatmap = np.zeros((T, ang_bins), dtype=float)
            present = np.zeros(T, dtype=bool)
            for tk in sm:
                if tk == 'global':
                    continue
                try:
                    ti = int(tk)
                except Exception:
                    continue
                lst = side_entries(sm[tk], side)
                if not lst:
                    continue
                ent = lst[0] if isinstance(lst, list) else lst
                ah = ent.get('analysis', {}).get('angular_hist') if isinstance(ent, dict) else None
                if isinstance(ah, list) and len(ah) == ang_bins:
                    heatmap[ti, :] = np.array(ah, dtype=float)
                    present[ti] = True
            for i in range(T):
                if not present[i]:
                    heatmap[i, :] = np.nan
        # Aggregated interval histogram
        agg = None; edges = None; acc = None; count = 0; first_edges = None
        for tk in sm:
            if tk == 'global':
                continue
            lst = side_entries(sm[tk], side)
            if not lst:
                continue
            ent = lst[0] if isinstance(lst, list) else lst
            if not isinstance(ent, dict):
                continue
            an = ent.get('analysis', {})
            ih = an.get('interval_hist'); ih_bins = an.get('interval_hist_bins'); ih_range = an.get('interval_hist_range_ns')
            if isinstance(ih, list) and isinstance(ih_bins, int) and ih_bins > 0 and isinstance(ih_range, list) and len(ih_range) == 2:
                if first_edges is None:
                    mn, mx = float(ih_range[0]), float(ih_range[1])
                    first_edges = np.logspace(np.log10(max(mn, 1.0)), np.log10(max(mx, mn + 1.0)), ih_bins + 1)
                vec = np.array(ih, dtype=float)
                acc = vec.copy() if acc is None else (acc + vec)
                count += 1
        if acc is not None and count > 0:
            agg = acc / float(count)
            edges = first_edges
        return heatmap, (agg, edges)

    hm0, (h0, e0) = build_ang_heat_and_hist(0)
    hm1, (h1, e1) = build_ang_heat_and_hist(1)

    # Compose figure (4 rows x 2 cols)
    fig = plt.figure(figsize=(13, 16))
    gs = GridSpec(4, 2, height_ratios=[1.6, 1.2, 1.2, 1.2], width_ratios=[1.0, 1.0], figure=fig)

    # Row 1: density polar side0 | side1
    for col, side in enumerate([0, 1]):
        ax = fig.add_subplot(gs[0, col], projection='polar')
        try:
            off = _theta_offset_for_side(sm, args, side)
            if abs(off) > 1e-9:
                ax.set_theta_offset(off)
        except Exception:
            pass
        # Build angular templates per track for this side
        def _tpl_for_track_dashboard(ti: int, side_: int):
            try:
                lst = sm.get(str(ti), {}).get(str(side_), []) if isinstance(sm.get(str(ti)), dict) else []
                if isinstance(lst, dict):
                    lst = [lst]
                if lst:
                    ent = lst[0]
                    ah = ent.get('analysis', {}).get('angular_hist') if isinstance(ent, dict) else None
                    b = ent.get('analysis', {}).get('angular_bins') if isinstance(ent, dict) else None
                    if isinstance(ah, list) and isinstance(b, int) and b > 1:
                        h = np.array(ah[:b], dtype=float)
                        m = float(np.max(h))
                        if m > 0:
                            h = h / m
                        th_centers = (np.arange(b) + 0.5) * (2 * np.pi / float(b))
                        x = np.concatenate([th_centers - 2*np.pi, th_centers, th_centers + 2*np.pi])
                        y = np.concatenate([h, h, h])
                        return np.interp(theta % (2*np.pi), x, y)
            except Exception:
                pass
            return np.ones_like(theta)
        # Normalize density per track for this side
        den = radials[side]
        mden = float(np.max(den[masks[side]])) if masks[side].any() else 0.0
        dnorm = (den / mden) if mden > 0 else np.zeros_like(den)
        # Build Z
        Z = np.zeros((r_up.shape[0], theta.shape[0]), dtype=float)
        for ri, rv in enumerate(r_up):
            ti = int(np.clip(int(round(rv)), 0, T - 1))
            tpl = _tpl_for_track_dashboard(ti, side)
            Z[ri, :] = tpl * float(dnorm[ti])
        pcm = ax.pcolormesh(TH, R, Z, cmap='viridis', vmin=0.0, vmax=1.0, shading='auto')
        ax.set_ylim(0, T)
        ax.set_yticks([0, T // 4, T // 2, 3 * T // 4, T - 1])
        ax.set_yticklabels(["0", str(T // 4), str(T // 2), str(3 * T // 4), str(T - 1)])
        badge = _formatted_badge(sm, side)
        ax.set_title(f"Side {side} – {badge} – Angular-Weighted Density")
        # Draw wedge spokes/labels if available
        try:
            color = getattr(args, 'overlay_color', '#ff3333')
            alpha = float(getattr(args, 'overlay_alpha', 0.8))
            label = bool(getattr(args, 'label_sectors', False))
            angles, _k = _wedge_angles_for_side(sm, args, side)
            if angles:
                _draw_wedges(ax, T, angles, label, color=color, alpha=alpha)
        except Exception:
            pass
        try:
            pcm.set_rasterized(True)
        except Exception:
            pass
        # Wedge lines and intra-wedge peak ticks
        try:
            color = getattr(args, 'overlay_color', '#ff3333')
            alpha = float(getattr(args, 'overlay_alpha', 0.8))
            label = bool(getattr(args, 'label_sectors', False))
            angles, _k = _wedge_angles_for_side(sm, args, side)
            if angles:
                _draw_wedges(ax, T, angles, label, color=color, alpha=alpha)
                hist, bins = _aggregate_side_hist(sm, side)
                peaks = _intra_wedge_peak_angles(hist, bins, angles)
                _draw_intra_wedge_peaks(ax, T, peaks)
        except Exception:
            pass

    # Add shared colorbar for row 1 (normalized 0..1)
    try:
        import matplotlib as _mpl
        cax = fig.add_axes([0.92, 0.76, 0.015, 0.18])
        m = _mpl.cm.ScalarMappable(cmap='viridis')
        m.set_clim(0.0, 1.0)
        cb = fig.colorbar(m, cax=cax)
        cb.set_label('Angular-Weighted Density (norm)')
    except Exception:
        pass

    # Row 2: instability polar side0 | side1
    for col, side in enumerate([0, 1]):
        ax = fig.add_subplot(gs[1, col], projection='polar')
        try:
            off = _theta_offset_for_side(sm, args, side)
            if abs(off) > 1e-9:
                ax.set_theta_offset(off)
        except Exception:
            pass
        inst_rad = np.array([float(instab_scores.get(side, {}).get(ti, 0.0)) for ti in range(T)])
        Zr = np.interp(r_up, np.arange(T), inst_rad)
        th_prof_theta, th_prof = _aggregate_instability_theta(sm, side)
        if th_prof is not None:
            prof = np.interp(theta % (2*np.pi), th_prof_theta, th_prof)
        else:
            prof = np.ones_like(theta)
        Z = np.outer(Zr, prof)
        Z = _contrast_stretch(Z, pct=97.0)
        pcm2 = ax.pcolormesh(TH, R, Z, cmap='magma_r', vmin=0.0, vmax=1.0, shading='auto')
        ax.set_ylim(0, T)
        ax.set_yticks([0, T // 4, T // 2, 3 * T // 4, T - 1])
        ax.set_yticklabels(["0", str(T // 4), str(T // 2), str(3 * T // 4), str(T - 1)])
        ax.set_title(f"Side {side} – Instability")
        try:
            pcm2.set_rasterized(True)
        except Exception:
            pass

    # Row 3: angular heatmaps
    for col, (hm, side) in enumerate([(hm0, 0), (hm1, 1)]):
        ax = fig.add_subplot(gs[2, col])
        if hm is not None:
            im = ax.imshow(hm, aspect='auto', origin='lower', interpolation='nearest', cmap='viridis')
            ax.set_title(f"Side {side} – Angular Activity (track × angle)")
            ax.set_xlabel('Angle bin')
            ax.set_ylabel('Track')
            try:
                im.set_rasterized(True)
            except Exception:
                pass
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.axis('off')
            ax.text(0.5, 0.5, 'No angular histograms', ha='center', va='center')

    # Row 4: aggregated interval histograms
    for col, (agg, edges, side) in enumerate([(h0, e0, 0), (h1, e1, 1)]):
        ax = fig.add_subplot(gs[3, col])
        if agg is not None and edges is not None:
            centers = np.sqrt(edges[:-1] * edges[1:])
            ax.bar(centers, agg, width=np.diff(edges), align='center', alpha=0.85, color='steelblue')
            ax.set_xscale('log')
            ax.set_xlim(edges[0], edges[-1])
            ax.set_xlabel('Interval (ns, log)')
            ax.set_ylabel('Norm density')
            ax.set_title(f"Side {side} – Flux Interval Histogram")
        else:
            ax.axis('off')
            ax.text(0.5, 0.5, 'No interval histogram', ha='center', va='center')

    plt.tight_layout()
    fmt = str(getattr(args, 'export_format', 'png') or 'png').lower()
    base = Path(str(out_prefix) + "_dashboard")
    try:
        if fmt in ('png', 'both'):
            plt.savefig(str(base.with_suffix('.png')), dpi=dpi)
        if fmt in ('svg', 'both'):
            plt.savefig(str(base.with_suffix('.svg')), dpi=dpi, format='svg')
    finally:
        plt.close()
