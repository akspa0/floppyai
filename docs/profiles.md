# Profiles (JSON)

FloppyAI uses JSON profiles to provide friendly, format-aware defaults for generation, analysis, and overlays. Profiles live under:

- `FloppyAI/profiles/`

Each profile file (e.g., `35HD.json`) declares media characteristics (RPM, safe track range), analyzer heuristics (bitcell-based thresholds), and overlay hints (MFM/GCR candidates).

## Built-in Profiles

- 35HD (3.5" 1.44MB, MFM)
- 35DD (3.5" 720KB, MFM)
- 35HDGCR (3.5" 800K-like, GCR)
- 35DDGCR (3.5" 400K-like, GCR)
- 525HD (5.25" 1.2MB, MFM)
- 525DD (5.25" 360KB, MFM)
- 525DDGCR (5.25" Apple II family, GCR)
- auto (generic fallback; conservative defaults)
- safe-usb (generator patterns only): conservative bitcells and enabled flux sanitizer for hardware-friendly streams

## What Profiles Control

- RPM defaults (used by generation timing and analysis normalization)
- Safe track limits (`safe_max_tracks`) used by CLI defaults
- Analyzer parameters (via `base_cell_ns`, short/long thresholds, weak-bit window ranges, interval histogram range)
- Overlay defaults (mode and candidate sector counts)

CLI flags still override profile values. For example, `--rpm` overrides the RPM from a profile.

If neither `--profile` nor `--rpm` is provided, FloppyAI defaults to `RPM=300.0` and prints a one‑line warning to help you choose a profile or RPM explicitly.

## JSON Schema (Example)

```json
{
  "name": "35HD",
  "media": "35",
  "encoding": "MFM",
  "rpm": 300.0,
  "safe_max_tracks": 81,
  "base_cell_ns": 4000.0,
  "analyzer": {
    "short_cell_multiple": 0.5,
    "long_interval_sigma": 3.0,
    "weak_window_multiples": [1.5, 2.5],
    "interval_hist_min_ns": 150.0,
    "interval_hist_max_ns": 60000.0
  },
  "overlays": {
    "mode": "mfm",
    "mfm_candidates": [8, 9, 15, 18]
  }
}
```

## How Analysis Uses Profiles

`FluxAnalyzer.analyze()` accepts an optional `profile_name`. When provided, the analyzer:

- Derives a nominal bitcell (`base_cell_ns`) and uses it to set thresholds
  - Short cells: `interval < short_cell_multiple × base_cell_ns`
  - Long intervals: `interval > mean + long_interval_sigma × std`
  - Weak-bit window: `[low, high] × base_cell_ns`, filtered by cross‑rev inconsistency
- Uses profile-specified interval histogram min/max where applicable

If no profile is provided, data-driven heuristics are used (current defaults remain for backward compatibility).

## Sanitizer and Safe-USB

The `safe-usb` generator profile enables the flux sanitizer to clamp very short cells and split very long gaps (keepalive), helping stability on modern USB controllers. See `docs/sanitizing_flux.md` for details and advanced flags.

## Overlays and Profiles

When `--format-overlay` is enabled, profiles bias the overlay mode and candidates:

- MFM profiles: `overlay_mode = mfm`; typical candidates include 8, 9, 15, 18
- GCR profiles: `overlay_mode = gcr`; typical candidates include 12, 11, 10, 9, 8
- `auto` profile: conservative defaults; you can still force `--overlay-mode`

## Custom Profiles

Add a new JSON to `FloppyAI/profiles/` with a unique `name`. The CLI will recognize it when passed via `--profile <name>`. Keep values reasonable (e.g., RPM within 200–400, `safe_max_tracks` within hardware limits).

