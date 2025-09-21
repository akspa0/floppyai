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
