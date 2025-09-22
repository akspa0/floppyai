from __future__ import annotations
"""
image2flux: Generate a flux stream from an input image for round‑trip experiments.

MVP approach (angular‑only pattern):
- Convert image to grayscale/binary vector of length `angular_bins`
- For each revolution, emit a number of transitions per angular bin:
  - ON (dark) bins emit `on_count` transitions
  - OFF (light) bins emit `off_count` transitions
- Each transition interval is assigned a fixed nanosecond value (e.g., 2000 ns)
- Write either internal FLUX format or a KryoFlux‑like stream with per‑rev index markers

Notes:
- This MVP focuses on analysis within FloppyAI. It does not attempt strict
  hardware‑accurate timing per RPM yet (that will come in a later iteration).
- The analyzer normalizes per‑rev timing when building angular histograms,
  so this representation is sufficient to visualize intended angular patterns
  and test structure‑finding.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image

from stream_export import write_internal_raw, write_kryoflux_stream


def _load_image_row(image_path: str, angular_bins: int, dither: str = "none", threshold: float = 0.5) -> np.ndarray:
    """Load image, convert to a 1D angular stripe (length = angular_bins) of 0/1.
    - If the image is 2D, we resize to (angular_bins, 1) and then threshold/dither.
    - If dither == 'ordered' or 'floyd', apply simple mapping; otherwise plain threshold.
    Returns a 1D numpy array of shape (angular_bins,) with values {0,1}.
    """
    img = Image.open(image_path).convert("L")  # grayscale
    # Resize preserving aspect into a single row
    img_small = img.resize((int(angular_bins), 1), Image.LANCZOS)
    arr = np.asarray(img_small, dtype=np.float32) / 255.0  # shape (1, angular_bins)
    vec = arr[0]

    if dither == "ordered":
        # Simple 2x2 Bayer thresholding repeated across the row
        bayer = np.array([0.25, 0.75], dtype=np.float32)
        pat = np.resize(bayer, vec.shape)
        out = (vec < (threshold + 0.15 * (pat - 0.5))).astype(np.uint8)
    elif dither == "floyd":
        # Very simple error diffusion along the row (1D Floyd–Steinberg‑like)
        v = vec.copy()
        out = np.zeros_like(v, dtype=np.uint8)
        for i in range(v.shape[0]):
            old = v[i]
            new = 1.0 if old < threshold else 0.0  # dark=1
            out[i] = 1 if new >= 0.5 else 0
            err = old - new
            if i + 1 < v.shape[0]:
                v[i + 1] += err * 7/16
            if i + 2 < v.shape[0]:
                v[i + 2] += err * 1/16
    else:
        out = (vec < threshold).astype(np.uint8)  # darker → 1

    return out.astype(np.uint8)


def _build_intervals_from_vector(vec01: np.ndarray, revolutions: int, on_count: int, off_count: int, interval_ns: int) -> np.ndarray:
    """Convert a 0/1 angular vector into a list of flux intervals for N revolutions.
    For each angular bin, emit 'on_count' or 'off_count' transitions using a fixed
    interval length. Concatenate across bins and revolutions.
    """
    bins = int(vec01.shape[0])
    per_rev_counts = np.where(vec01 > 0, int(on_count), int(off_count))
    per_rev_total = int(np.sum(per_rev_counts))
    if per_rev_total <= 0:
        per_rev_total = bins  # fallback: 1 per bin
        per_rev_counts = np.ones(bins, dtype=int)

    # Build one rev
    one_rev = np.full(per_rev_total, int(interval_ns), dtype=np.int64)
    # For clarity, we don't insert separators; analyzer uses cumulative sums per rev.

    # Repeat for N revs
    intervals = np.tile(one_rev, int(max(1, revolutions)))
    return intervals


def generate_from_image(
    image_path: str,
    track: int,
    side: int,
    output_path: str,
    revolutions: int = 1,
    angular_bins: int = 720,
    on_count: int = 4,
    off_count: int = 1,
    interval_ns: int = 2000,
    output_format: str = "kryoflux",
    rpm: float = 300.0,
    dither: str = "none",
    threshold: float = 0.5,
) -> Tuple[str, int]:
    """High‑level helper to generate a flux stream from an image.

    Returns (output_path, total_intervals).
    """
    vec01 = _load_image_row(image_path, angular_bins=angular_bins, dither=dither, threshold=float(threshold))
    intervals = _build_intervals_from_vector(
        vec01,
        revolutions=int(revolutions),
        on_count=int(on_count),
        off_count=int(off_count),
        interval_ns=int(interval_ns),
    )

    if output_format == "internal":
        write_internal_raw(intervals.tolist(), track, side, output_path, num_revs=revolutions)
    else:
        # Compute exact per-revolution lengths so OOB index blocks align
        per_rev_len = int(intervals.size // max(1, int(revolutions)))
        rev_lengths = [per_rev_len] * int(revolutions)
        write_kryoflux_stream(
            intervals.tolist(),
            track,
            side,
            output_path,
            num_revs=revolutions,
            rpm=float(rpm),
            rev_lengths=rev_lengths,
        )

    return output_path, int(intervals.size)
