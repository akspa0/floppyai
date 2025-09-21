from __future__ import annotations
"""
structure_finder: reconstruct a polar (track × angle) image from surface_map.json
and optionally compare it to an expected polar target (from silkscreen) to
report correlation and phase alignment metrics.
"""
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

import numpy as np
from PIL import Image

from utils.json_io import dump_json


def _load_surface_map(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)


def _side_entries(track_obj, side_int: int):
    if not isinstance(track_obj, dict):
        return []
    v = track_obj.get(side_int)
    if v is None:
        v = track_obj.get(str(side_int))
    if isinstance(v, dict):
        return [v]
    return v if isinstance(v, list) else []


def _resample_hist(hist: np.ndarray, bins: int, target_bins: int) -> np.ndarray:
    """Resample an angular histogram of length `bins` to `target_bins` by
    interpolating on the angle centers with wraparound.
    """
    if bins == target_bins:
        return hist.copy()
    th_src = (np.arange(bins) + 0.5) * (2 * np.pi / float(bins))
    th_dst = (np.arange(target_bins) + 0.5) * (2 * np.pi / float(target_bins))
    # Extend by wrap for robust interpolation
    x = np.concatenate([th_src - 2*np.pi, th_src, th_src + 2*np.pi])
    y = np.concatenate([hist, hist, hist])
    out = np.interp(th_dst % (2*np.pi), x, y)
    return out


def recover_image(
    surface_map_path: str,
    output_dir: str,
    side: int = 0,
    tracks: Optional[List[int]] = None,
    angular_bins: int = 720,
    expected_image: Optional[str] = None,
) -> Dict:
    """Reconstruct a polar image (track × angle) from surface_map.json for a side.

    Returns a manifest with paths and metrics.
    """
    sm = _load_surface_map(surface_map_path)
    # Determine track set
    tr_all = []
    for tk in sm:
        if tk == 'global':
            continue
        try:
            tr_all.append(int(tk))
        except Exception:
            pass
    tr_all = sorted(set(tr_all))
    if tracks is None or len(tracks) == 0:
        tracks = tr_all
    else:
        tracks = [t for t in tracks if t in tr_all]
    if not tracks:
        raise ValueError("No matching tracks found in surface_map.json for requested set")

    # Build matrix (radial × angular_bins)
    radial = len(tracks)
    ang_bins = int(angular_bins)
    mat = np.zeros((radial, ang_bins), dtype=np.float32)
    present = np.zeros(radial, dtype=bool)

    # First pass: find a typical normalization scale per track (max of its histogram)
    for i, t in enumerate(tracks):
        entry_list = _side_entries(sm.get(str(t), {}), side)
        if not entry_list:
            continue
        ent = entry_list[0]
        ah = ent.get('analysis', {}).get('angular_hist') if isinstance(ent, dict) else None
        bins = ent.get('analysis', {}).get('angular_bins') if isinstance(ent, dict) else None
        if isinstance(ah, list) and isinstance(bins, int) and bins > 0:
            h = np.array(ah[:bins], dtype=np.float32)
            # Normalize per track to 0..1 by its max (avoid division by zero)
            mx = float(np.max(h))
            if mx > 0:
                h = h / mx
            h_res = _resample_hist(h, bins, ang_bins)
            mat[i, :] = h_res
            present[i] = True

    if not np.any(present):
        raise ValueError("No angular histograms found for requested side/tracks")

    # Save recovered image (theta across width)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_path = out_dir / 'recovered_polar.png'
    npy_path = out_dir / 'recovered_polar.npy'
    try:
        arr = np.clip(mat, 0.0, 1.0)
        img8 = (arr * 255.0 + 0.5).astype(np.uint8)
        Image.fromarray(img8, mode='L').save(str(img_path))
        np.save(str(npy_path), arr)
    except Exception:
        pass

    metrics: Dict[str, float] = {}
    expected_path_out: Optional[str] = None
    # If expected provided, resample to same [radial × ang_bins] and compute correlation
    if expected_image and Path(expected_image).exists():
        expected_path_out = str(expected_image)
        try:
            eimg = Image.open(expected_image).convert('L')
            eimg = eimg.resize((ang_bins, radial), Image.LANCZOS)
            E = np.asarray(eimg, dtype=np.float32) / 255.0
            # Compute simple normalized cross-correlation across all pixels
            A = mat.astype(np.float32)
            a = A - np.mean(A)
            b = E - np.mean(E)
            denom = float(np.linalg.norm(a) * np.linalg.norm(b))
            corr = float((a*b).sum() / denom) if denom > 0 else 0.0
            metrics['corr_coeff'] = corr
            # Estimate best circular phase (θ shift) by 1D FFT peak on angular average
            avg_row = np.nanmean(A, axis=0)
            avg_exp = np.nanmean(E, axis=0)
            # Cross-correlation via FFT
            F1 = np.fft.rfft(avg_row)
            F2 = np.conj(np.fft.rfft(avg_exp))
            cc = np.fft.irfft(F1 * F2)
            shift = int(np.argmax(cc))
            metrics['best_phase_bins'] = shift
        except Exception:
            pass

    manifest = {
        'input_surface_map': str(surface_map_path),
        'side': int(side),
        'tracks': tracks,
        'angular_bins': ang_bins,
        'recovered_png': str(img_path),
        'recovered_npy': str(npy_path),
        'expected': expected_path_out,
        'metrics': metrics,
    }
    out_json = Path(output_dir) / 'recovered_manifest.json'
    dump_json(out_json, manifest)
    return manifest
