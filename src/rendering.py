from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


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

    # Upsample radial resolution for smoother rendering
    # Use [0, T] so a single track (T=1) still has non-zero radial extent
    up_factor = 4
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
    theta = np.linspace(0, 2 * np.pi, 1440)
    TH, R = np.meshgrid(theta, r_up)

    # Layout: side0 | side1 | colorbar
    fig = plt.figure(figsize=(13, 5.5))
    gs = GridSpec(1, 3, width_ratios=[1, 1, 0.05], figure=fig)
    ax0 = fig.add_subplot(gs[0, 0], projection='polar')
    ax1 = fig.add_subplot(gs[0, 1], projection='polar')
    cax = fig.add_subplot(gs[0, 2])

    pcm = None
    for ax, side in [(ax0, 0), (ax1, 1)]:
        if counts[side] > 0 and masks[side].any():
            Z = np.repeat(radial_up[side][:, None], theta.shape[0], axis=1)
            pcm = ax.pcolormesh(TH, R, Z, cmap='viridis', vmin=vmin, vmax=vmax, shading='auto')
            ax.set_ylim(0, T)
            ax.set_yticks([0, T // 4, T // 2, 3 * T // 4, T - 1])
            ax.set_yticklabels(["0", str(T // 4), str(T // 2), str(3 * T // 4), str(T - 1)])
            ax.set_title(f"Side {side}")
        else:
            ax.set_title(f"Side {side} (no data)")
            ax.set_ylim(0, T)
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center')

    if pcm is not None:
        cbar = fig.colorbar(pcm, cax=cax, orientation='vertical')
        cbar.set_label('Bits per Revolution')

    plt.tight_layout()
    outfile = Path(str(out_prefix) + "_disk_surface.png")
    plt.savefig(str(outfile), dpi=220)
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
        outfile = Path(str(out_prefix) + f"_side{side}_report.png")
        plt.savefig(str(outfile), dpi=180)
        plt.close()
        return

    # Color scale from available densities
    vmin = float(np.percentile(dens_all, 5)) if len(dens_all) >= 2 else float(min(dens_all + [0.0]))
    vmax = float(np.percentile(dens_all, 95)) if len(dens_all) >= 2 else float(max(dens_all + [1.0]))
    if vmax <= vmin:
        vmax = vmin + 1.0

    # Upsample radial
    up_factor = 4
    Tu = max(T * up_factor, T)
    r_up = np.linspace(0, T, Tu)
    if has.sum() >= 2:
        xs = np.where(has)[0]
        ys = radial[has]
        radial_up = np.interp(r_up, xs, ys)
    else:
        fill = float(radial[has][0]) if has.any() else 0.0
        radial_up = np.full_like(r_up, fill, dtype=float)
    theta = np.linspace(0, 2 * np.pi, 1440)
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
    Z = np.repeat(radial_up[:, None], theta.shape[0], axis=1)
    pcm = ax_surf.pcolormesh(TH, R, Z, cmap='viridis', vmin=vmin, vmax=vmax, shading='auto')
    ax_surf.set_ylim(0, T)
    ax_surf.set_yticks([0, T // 4, T // 2, 3 * T // 4, T - 1])
    ax_surf.set_yticklabels(["0", str(T // 4), str(T // 2), str(3 * T // 4), str(T - 1)])
    ax_surf.set_title(f"Side {side} – Density per Track (bits/rev)")
    if thetas_overlay:
        ov_color = getattr(args, 'overlay_color', '#ff3333')
        ov_alpha = float(getattr(args, 'overlay_alpha', 0.8))
        for th in thetas_overlay:
            ax_surf.plot([th, th], [0, T], color=ov_color, alpha=ov_alpha, linewidth=0.9)
    fig.colorbar(pcm, ax=ax_surf, orientation='vertical', pad=0.1)

    # Middle-left: instability polar for this side
    ax_inst = fig.add_subplot(gs[1, 0], projection='polar')
    # Build instability radial for this side
    inst_rad = np.zeros(T)
    for ti in range(T):
        inst_rad[ti] = float(instab_scores.get(side, {}).get(ti, 0.0))
    Zr = np.interp(r_up, np.arange(T), inst_rad)
    Z2 = np.repeat(Zr[:, None], theta.shape[0], axis=1)
    pcm2 = ax_inst.pcolormesh(TH, R, Z2, cmap='magma', vmin=0.0, vmax=1.0, shading='auto')
    ax_inst.set_ylim(0, T)
    ax_inst.set_yticks([0, T // 4, T // 2, 3 * T // 4, T - 1])
    ax_inst.set_yticklabels(["0", str(T // 4), str(T // 2), str(3 * T // 4), str(T - 1)])
    ax_inst.set_title(f"Side {side} – Instability")

    # Middle-right: angular heatmap (track x angle)
    ax_hm = fig.add_subplot(gs[1, 1])
    if heatmap is not None:
        im = ax_hm.imshow(heatmap, aspect='auto', origin='lower', interpolation='nearest', cmap='viridis')
        ax_hm.set_xlabel('Angle bin')
        ax_hm.set_ylabel('Track')
        ax_hm.set_title('Angular Activity (track × angle)')
        plt.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04)
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
    outfile = Path(str(out_prefix) + f"_side{side}_report.png")
    plt.savefig(str(outfile), dpi=180)
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

    


def render_instability_map(instab_scores: dict, T: int, out_prefix: Path) -> None:
    """Render an instability polar map for both sides using instab_scores.{side}{track}=score."""
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
        Z = np.repeat(Zr[:, None], theta.shape[0], axis=1)
        pcm = ax.pcolormesh(TH, R, Z, cmap='magma', vmin=vmin, vmax=vmax, shading='auto')
        ax.set_ylim(0, T)
        ax.set_yticks([0, T // 4, T // 2, 3 * T // 4, T - 1])
        ax.set_yticklabels(["0", str(T // 4), str(T // 2), str(3 * T // 4), str(T - 1)])
        ax.set_title(f"Side {side}")
    if pcm is not None:
        cbar = plt.colorbar(pcm, cax=cax, orientation='vertical')
        cbar.set_label('Instability Score (0-1)')
    plt.tight_layout()
    outfile = Path(str(out_prefix) + "_instability_map.png")
    plt.savefig(str(outfile), dpi=220)
    plt.close()
