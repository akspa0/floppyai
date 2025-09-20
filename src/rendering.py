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
        if side_int in track_obj:
            return track_obj.get(side_int, [])
        return track_obj.get(str(side_int), [])

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
    up_factor = 4
    Tu = max(T * up_factor, T)
    r_up = np.linspace(0, T - 1, Tu)

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
        if masks[side].any():
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

    # Individual high-res side images
    for side in [0, 1]:
        fig = plt.figure(figsize=(7, 6))
        gs = GridSpec(1, 2, width_ratios=[1, 0.05], figure=fig)
        ax = fig.add_subplot(gs[0, 0], projection='polar')
        cax = fig.add_subplot(gs[0, 1])
        if masks[side].any():
            Z = np.repeat(radial_up[side][:, None], theta.shape[0], axis=1)
            pcm = ax.pcolormesh(TH, R, Z, cmap='viridis', vmin=vmin, vmax=vmax, shading='auto')
            ax.set_ylim(0, T)
            ax.set_yticks([0, T // 4, T // 2, 3 * T // 4, T - 1])
            ax.set_yticklabels(["0", str(T // 4), str(T // 2), str(3 * T // 4), str(T - 1)])
            ax.set_title(f"Side {side}")
            cbar = fig.colorbar(pcm, cax=cax, orientation='vertical')
            cbar.set_label('Bits per Revolution')
        else:
            ax.set_title(f"Side {side} (no data)")
            ax.set_ylim(0, T)
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center')
        plt.tight_layout()
        out_single = Path(str(out_prefix) + f"_side{side}.png")
        plt.savefig(str(out_single), dpi=240)
        plt.close()

    # Overlays
    try:
        ov_enabled = bool(getattr(args, 'format_overlay', False))
        ov_alpha = float(getattr(args, 'overlay_alpha', 0.8))
        ov_color = getattr(args, 'overlay_color', '#ff3333')
        overlay_info = sm.get('global', {}).get('insights', {}).get('overlay', {}) if isinstance(sm.get('global'), dict) else {}
        bside = overlay_info.get('by_side', {}) if isinstance(overlay_info, dict) else {}
        thetas_by_side = {}
        for s in [0, 1]:
            bdeg = bside.get(str(s), {}).get('boundaries_deg', [])
            if not bdeg:
                continue
            thetas_by_side[s] = [np.deg2rad(x) for x in bdeg]
        if ov_enabled and thetas_by_side:
            # Combined overlay
            fig = plt.figure(figsize=(13, 5.5))
            gs = GridSpec(1, 3, width_ratios=[1, 1, 0.05], figure=fig)
            ax0 = fig.add_subplot(gs[0, 0], projection='polar')
            ax1 = fig.add_subplot(gs[0, 1], projection='polar')
            cax = fig.add_subplot(gs[0, 2])
            pcm = None
            for ax, side in [(ax0, 0), (ax1, 1)]:
                if masks[side].any():
                    Z = np.repeat(radial_up[side][:, None], theta.shape[0], axis=1)
                    pcm = ax.pcolormesh(TH, R, Z, cmap='viridis', vmin=vmin, vmax=vmax, shading='auto')
                    ax.set_ylim(0, T)
                    ax.set_yticks([0, T // 4, T // 2, 3 * T // 4, T - 1])
                    ax.set_yticklabels(["0", str(T // 4), str(T // 2), str(3 * T // 4), str(T - 1)])
                    if side in thetas_by_side:
                        for th in thetas_by_side[side]:
                            ax.plot([th, th], [0, T], color=ov_color, alpha=ov_alpha, linewidth=0.9)
            if pcm is not None:
                cbar = fig.colorbar(pcm, cax=cax, orientation='vertical')
                cbar.set_label('Bits per Revolution')
            plt.tight_layout()
            out_overlay = Path(str(out_prefix) + "_disk_surface_overlay.png")
            plt.savefig(str(out_overlay), dpi=220)
            plt.close()

            # Per-side overlays
            for side in [0, 1]:
                fig = plt.figure(figsize=(7, 6))
                gs = GridSpec(1, 2, width_ratios=[1, 0.05], figure=fig)
                ax = fig.add_subplot(gs[0, 0], projection='polar')
                cax = fig.add_subplot(gs[0, 1])
                if masks[side].any():
                    Z = np.repeat(radial_up[side][:, None], theta.shape[0], axis=1)
                    pcm = ax.pcolormesh(TH, R, Z, cmap='viridis', vmin=vmin, vmax=vmax, shading='auto')
                    ax.set_ylim(0, T)
                    ax.set_yticks([0, T // 4, T // 2, 3 * T // 4, T - 1])
                    ax.set_yticklabels(["0", str(T // 4), str(T // 2), str(3 * T // 4), str(T - 1)])
                    if side in thetas_by_side:
                        for th in thetas_by_side[side]:
                            ax.plot([th, th], [0, T], color=ov_color, alpha=ov_alpha, linewidth=0.9)
                    cbar = fig.colorbar(pcm, cax=cax, orientation='vertical')
                    cbar.set_label('Bits per Revolution')
                else:
                    ax.set_ylim(0, T)
                    ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center')
                plt.tight_layout()
                out_single_overlay = Path(str(out_prefix) + f"_side{side}_overlay.png")
                plt.savefig(str(out_single_overlay), dpi=240)
                plt.close()
    except Exception as e:
        print(f"Overlay rendering skipped due to error: {e}")


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
        from matplotlib import pyplot as plt
        cbar = plt.colorbar(pcm, cax=cax, orientation='vertical')
        cbar.set_label('Instability Score (0-1)')
    plt.tight_layout()
    outfile = Path(str(out_prefix) + "_instability_map.png")
    plt.savefig(str(outfile), dpi=220)
    plt.close()
