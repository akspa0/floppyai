from __future__ import annotations
"""
Silkscreen: map an arbitrary image (any resolution) onto a disk side (tracks × angle)
by generating per‑track flux streams whose local transition density matches pixel
intensity in a polar grid.

Strategy
- Resample input image to a target grid [radial_bins × angular_bins]
  where radial_bins = number of tracks to write and angular_bins is the phase resolution.
- For each track (row), convert brightness to a per‑bin transition count between
  [off_count_min, on_count_max] with optional dithering along theta.
- Convert transition counts into intervals so that total per‑rev time equals Tr = 60e9/RPM ns,
  respecting [min_interval_ns, max_interval_ns]. Small residual error is distributed across bins.
- Write one or more revolutions per track using a KryoFlux‑like stream or internal format.

Notes
- This MVP focuses on analysis and visual round‑trip. Fine‑tuning for hardware timing nuances
  will be iterative and guided by Linux dtc captures.
"""

from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from PIL import Image

from stream_export import write_internal_raw, write_kryoflux_stream


def _resample_to_polar_grid(image_path: str, radial_bins: int, angular_bins: int) -> np.ndarray:
    """Load image and resample to [radial_bins × angular_bins] grayscale in 0..1.
    The radial axis corresponds to track index order (0..N-1), angular to phase.
    """
    img = Image.open(image_path).convert("L")
    # Resize to (angular_bins wide, radial_bins tall)
    img_resized = img.resize((int(angular_bins), int(radial_bins)), Image.LANCZOS)
    arr = np.asarray(img_resized, dtype=np.float32) / 255.0  # shape: (radial, angular)
    return arr


essential_dithers = {"none", "ordered", "floyd"}

def _dither_row(vec: np.ndarray, method: str, threshold: float) -> np.ndarray:
    """Apply simple 1D dithering/threshold to a grayscale row in 0..1.
    Returns 0/1 vector where 1 denotes darker (more transitions).
    """
    v = np.clip(vec.astype(np.float32), 0.0, 1.0)
    if method == "ordered":
        # 8‑element ordered pattern across theta
        pat = np.array([0.1,0.3,0.5,0.7,0.9,0.7,0.5,0.3], dtype=np.float32)
        tiled = np.resize(pat, v.shape)
        return (v < (threshold + 0.2*(tiled - 0.5))).astype(np.uint8)
    if method == "floyd":
        # 1D Floyd–Steinberg‑like error diffusion
        vv = v.copy()
        out = np.zeros_like(vv, dtype=np.uint8)
        for i in range(vv.shape[0]):
            old = vv[i]
            new = 1.0 if old < threshold else 0.0
            out[i] = 1 if new >= 0.5 else 0
            err = old - new
            if i + 1 < vv.shape[0]:
                vv[i + 1] += err * 7/16
            if i + 2 < vv.shape[0]:
                vv[i + 2] += err * 1/16
        return out
    # default: hard threshold (dark → 1)
    return (v < threshold).astype(np.uint8)


def _scale_counts_to_target(weights: np.ndarray, target_total: int, min_per_bin: int = 0) -> np.ndarray:
    """Distribute integer counts across bins to sum exactly to target_total.
    Uses floor on proportional allocation then assigns remainder to largest residuals.
    """
    w = np.clip(weights.astype(np.float64), 0.0, None)
    if np.all(w <= 0):
        # fallback: uniform distribution
        base = target_total // max(1, w.shape[0])
        rem = target_total - base * w.shape[0]
        counts = np.full(w.shape[0], base, dtype=int)
        counts[:rem] += 1
        return counts
    w_sum = float(np.sum(w))
    raw = (w / w_sum) * float(max(0, target_total - min_per_bin * w.shape[0]))
    base = np.floor(raw).astype(int)
    residual = raw - base
    counts = base + min_per_bin
    cur = int(np.sum(counts))
    need = int(target_total - cur)
    if need > 0:
        idx = np.argsort(-residual)
        counts[idx[:need]] += 1
    elif need < 0:
        idx = np.argsort(residual)  # remove from smallest residuals first
        for j in idx:
            if need == 0:
                break
            if counts[j] > min_per_bin:
                counts[j] -= 1
                need += 1
    return counts.astype(int)


def _build_one_rev_intervals(counts: np.ndarray, rpm: float, min_ns: int, max_ns: int) -> np.ndarray:
    """Build one revolution worth of intervals matching total time Tr = 60e9/RPM ns.
    Uses near‑uniform intervals, then distributes residual error.
    """
    total_counts = int(np.sum(counts))
    # Guard: ensure at least 1 transition
    total_counts = max(1, total_counts)
    Tr = int(round(60_000_000_000.0 / float(rpm)))  # ns per revolution
    base = Tr // total_counts
    base = max(min_ns, min(base, max_ns))
    intervals = np.full(total_counts, base, dtype=np.int64)
    # adjust to match Tr exactly by spreading remainder
    diff = Tr - int(np.sum(intervals))
    if diff != 0:
        step = 1 if diff > 0 else -1
        n = abs(diff)
        idx = (np.arange(n, dtype=np.int64) * 9973) % total_counts
        if step > 0:
            can = intervals[idx] < max_ns
        else:
            can = intervals[idx] > min_ns
        if np.any(can):
            intervals[idx[can]] = np.clip(intervals[idx[can]] + step, min_ns, max_ns)
        # Recompute residual and, if small, finalize with a short loop
        diff2 = Tr - int(np.sum(intervals))
        if diff2 != 0:
            s2 = 1 if diff2 > 0 else -1
            for i in range(min(abs(diff2), 4096)):
                j = (i * 7919) % total_counts
                nv = intervals[j] + s2
                if min_ns <= nv <= max_ns:
                    intervals[j] = nv
            # final residual ignored if any; stays within a few µs budget
    return intervals


def generate_silkscreen(
    image_path: str | None,
    tracks: Iterable[int],
    side: int,
    output_dir: str,
    angular_bins: int = 720,
    rpm: float = 300.0,
    avg_interval_ns: int = 2200,
    min_interval_ns: int = 2000,
    max_interval_ns: int = 8000,
    on_count_max: int = 6,
    off_count_min: int = 1,
    dither: str = "floyd",
    threshold: float = 0.5,
    revolutions: int = 1,
    output_format: str = "kryoflux",
    polar_override: np.ndarray | None = None,
    disk_name: str = "disk",
) -> dict:
    """Silkscreen an image onto specified tracks of a side by generating per‑track .raw streams.

    Returns a manifest dict with file list and parameters.
    """
    tlist = sorted({int(t) for t in tracks})
    radial_bins = len(tlist)
    if polar_override is not None:
        polar = np.asarray(polar_override, dtype=np.float32)
        # Ensure shape matches (radial_bins, angular_bins)
        if polar.shape != (radial_bins, int(angular_bins)):
            # Resize using PIL for simplicity (theta width, radial height)
            arr8 = np.clip((polar * 255.0).round(), 0, 255).astype(np.uint8)
            img = Image.fromarray(arr8, mode='L') if arr8.ndim == 2 else Image.fromarray(arr8.squeeze(), mode='L')
            img = img.resize((int(angular_bins), int(radial_bins)), Image.LANCZOS)
            polar = np.asarray(img, dtype=np.float32) / 255.0
    else:
        if not image_path:
            raise ValueError("generate_silkscreen requires either image_path or polar_override")
        polar = _resample_to_polar_grid(image_path, radial_bins=radial_bins, angular_bins=int(angular_bins))

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Group all outputs for this set under a disk-named subfolder
    try:
        safe_disk = str(disk_name).strip().replace(" ", "_") or "disk"
    except Exception:
        safe_disk = "disk"
    disk_dir = out_dir / safe_disk
    disk_dir.mkdir(parents=True, exist_ok=True)
    # Save ground-truth polar target for later comparison
    gt_png = disk_dir / 'polar_target.png'
    gt_npy = disk_dir / 'polar_target.npy'
    try:
        arr8 = np.clip((polar * 255.0).round(), 0, 255).astype(np.uint8)
        # Save with theta as width and tracks as height
        img = Image.fromarray(arr8, mode='L')
        img.save(str(gt_png))
        np.save(str(gt_npy), polar)
    except Exception:
        pass
    files: List[dict] = []
    pattern_side = int(side)
    blank_side = 1 - pattern_side
    for idx, track in enumerate(tlist):
        row_gray = polar[idx, :]  # 0..1
        # Optional dithering along theta
        if dither not in essential_dithers:
            dither_mode = "none"
        else:
            dither_mode = dither
        row_bin = _dither_row(row_gray, dither_mode, float(threshold))  # 0/1 (dark=1)
        # Build weights from brightness: darker → higher weight
        inv = 1.0 - np.clip(row_gray, 0.0, 1.0)
        weights = float(off_count_min) + inv * float(max(off_count_min, on_count_max) - off_count_min)
        # Target transitions per revolution derived from RPM and desired average interval
        Tr = int(round(60_000_000_000.0 / float(rpm)))
        target_transitions = max(angular_bins, int(round(Tr / max(1, int(avg_interval_ns)))))
        # Scale integer counts to match target total exactly
        counts = _scale_counts_to_target(weights, target_transitions, min_per_bin=0)
        # Build one revolution intervals matching Tr
        one_rev = _build_one_rev_intervals(counts, rpm=float(rpm), min_ns=int(min_interval_ns), max_ns=int(max_interval_ns))
        # Repeat for requested revolutions (pattern side)
        intervals = np.tile(one_rev, int(max(1, revolutions)))

        # Write pattern side file
        fname_pat = f"track{track:02d}.{pattern_side}.raw"
        out_path_pat = str(disk_dir / fname_pat)
        if output_format == "internal":
            write_internal_raw(intervals, track, pattern_side, out_path_pat, num_revs=revolutions)
        else:
            write_kryoflux_stream(intervals, track, pattern_side, out_path_pat, num_revs=revolutions, rpm=float(rpm))
        files.append({"track": track, "side": pattern_side, "file": out_path_pat, "intervals": int(intervals.size)})

        # Build a uniform/blank flux stream for the opposite side so downstream tools have files
        weights_blank = np.ones_like(weights, dtype=np.float64)
        counts_blank = _scale_counts_to_target(weights_blank, target_transitions, min_per_bin=0)
        one_rev_blank = _build_one_rev_intervals(counts_blank, rpm=float(rpm), min_ns=int(min_interval_ns), max_ns=int(max_interval_ns))
        intervals_blank = np.tile(one_rev_blank, int(max(1, revolutions)))
        fname_blk = f"track{track:02d}.{blank_side}.raw"
        out_path_blk = str(disk_dir / fname_blk)
        if output_format == "internal":
            write_internal_raw(intervals_blank, track, blank_side, out_path_blk, num_revs=revolutions)
        else:
            write_kryoflux_stream(intervals_blank, track, blank_side, out_path_blk, num_revs=revolutions, rpm=float(rpm))
        files.append({"track": track, "side": blank_side, "file": out_path_blk, "intervals": int(intervals_blank.size)})

    manifest = {
        "image": (str(image_path) if image_path else None),
        "angular_bins": int(angular_bins),
        "rpm": float(rpm),
        "avg_interval_ns": int(avg_interval_ns),
        "min_interval_ns": int(min_interval_ns),
        "max_interval_ns": int(max_interval_ns),
        "on_count_max": int(on_count_max),
        "off_count_min": int(off_count_min),
        "dither": dither,
        "threshold": float(threshold),
        "revolutions": int(revolutions),
        "output_format": output_format,
        "side": int(side),
        "tracks": tlist,
        "output_dir": str(disk_dir),
        "disk_name": safe_disk,
        "files": files,
        "ground_truth_png": str(gt_png),
        "ground_truth_npy": str(gt_npy),
    }
    return manifest
