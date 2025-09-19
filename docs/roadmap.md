# FloppyAI Roadmap

This document outlines the near-term plan for expanding FloppyAI beyond analysis-only to a surface-aware, placement-guided tool. It captures what is implemented now, what’s next, and future phases.

## Scope
- 3.5" and 5.25" floppy support via drive profiles and free-form RPM.
- Flux-first quality metrics to distinguish formatting-like structure vs chaotic flux.
- High-fidelity visuals to inspect media condition and plan data placement.

## Status Summary

- Implemented
  - Profiles and RPM:
    - CLI flags `--profile {35HD,35DD,525HD,525DD}` and float `--rpm` accepted in `analyze_disk` and `analyze_corpus`.
    - Effective RPM: `--rpm` overrides; else profile; else 360.0.
    - `analyze_corpus --generate-missing` propagates the same RPM to each `analyze_disk` run.
  - Quality metrics (track/side):
    - `structure_score` ∈ [0,1] from normalized relative variance (lower variance ⇒ more structure).
    - `instability_score` ∈ [0,1] combining relative variance and outlier fraction.
    - Persisted in `surface_map.json` under the side summary entry.
  - Visuals:
    - High-fidelity polar density map with shared scale across sides.
    - High-fidelity polar instability map with shared scale across sides.
    - Composite now includes the instability map panel when available (fallback to rev heatmap otherwise).
  - Exports:
    - CSV: `<label>_instability_summary.csv` with `track,side,structure_score,instability_score`.

- In progress / Upcoming (near-term)
  - Documentation additions:
    - `docs/specs/surface-instability.md` with formulas and JSON field details.
    - `docs/specs/rpm-profiles.md` with mapping and guidance.
  - README usage updates describing profiles, RPM guidance, and new outputs (instability map, CSV).
  - Basic tests for numeric ranges and file outputs on sample data.

- Future (Phase 2+)
  - Angular binning (optional):
    - Flags (to be introduced later): `--angular-bins N` (default 0 – off), `--instability-thresh X`.
    - Bin each revolution into N angular slices; compute instability per bin; group into `bad_windows` (theta ranges) per track/side.
    - Visuals: R×Θ instability map; optional arc overlays on the density panel.
  - ML demodulation and "structured chaos" encoding (exploratory):
    - Data-set builder for interval sequences with labels.
    - Neural demodulator (Conv/GRU) converting normalized interval sequences to bits.
    - Placement-aware encoding with ECC tuned to measured surface/instability; planner avoids bad windows.

## CLI Reference (current)

- 5.25" HD (360 RPM):
```
python main.py analyze_corpus ..\stream_dumps --generate-missing --profile 525HD
```
- 3.5" HD or DD (300 RPM; wiring ready):
```
python main.py analyze_corpus ..\35_streams --generate-missing --profile 35HD
# or explicitly
python main.py analyze_corpus ..\35_streams --generate-missing --rpm 300
```
- `--rpm` accepts any float; if provided, it overrides `--profile`.

## Outputs (per disk)
- `<label>_surface_disk_surface.png` — polar density map, both sides.
- `<label>_instability_map.png` — polar instability map, both sides.
- `<label>_composite_report.png` — composite including instability (or heatmap fallback).
- `<label>_instability_summary.csv` — per-track/side scores.
- `surface_map.json` — now includes `structure_score` and `instability_score` under side summaries.

## Testing Checklist
- Run `analyze_disk` on a 5.25" test set; confirm new images and CSV appear.
- Run `analyze_corpus --generate-missing --profile 525HD`; confirm each disk inherits 360 RPM, and composites include instability.
- When 3.5" streams are available, re-run with `--profile 35HD` or `--rpm 300` and verify RPM validation logs and outputs.

## Notes
- Angular binning and overlays are deferred to a future phase and intentionally absent from the CLI for now to keep current runs fast.
- All new features are optional; if data is missing (e.g., instability map), composite falls back to existing panels.
