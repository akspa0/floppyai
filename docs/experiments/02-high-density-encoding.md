# Experiment 02 — High-Density Encoding Sweep

Characterize how increasing nominal bit density impacts stability, jitter, and readability on floppy media. Sweep densities while keeping pattern and other parameters fixed.

## Objectives & Hypotheses

- Determine practical upper limits of density for a given media/profile.
- Identify density ranges where instability/jitter grows nonlinearly.
- Compare densities across tracks (outer vs inner) and sides for asymmetries.

## Safety & Setup

- Use sacrificial media; start on outer tracks (79–83).
- Limit revs per write; add cooldown between runs.
- Enforce bounds (e.g., `--density ≤ 2.5`).
- Validate directory structure with `--simulate` before hardware runs.

## Experiment Matrix

- Pattern: `prbs7` (deterministic, good for correlation)
- Densities: 0.8, 1.0, 1.2, 1.5, 1.8, 2.0
- Tracks: 79–83; Sides: 0
- Revs: 1–2; Repetitions: 2 per density
- Media/profile: select consistent `--media-type` (e.g., `35HD`) and `--rpm` or `--profile`

## CLI

- Dry-run matrix:
  ```bash
  python -m FloppyAI.src.main experiment \
    --patterns prbs7 \
    --densities 0.8,1.0,1.2,1.5,1.8,2.0 \
    --tracks 79-83 --sides 0 \
    --revs 1 --reps 2 \
    --simulate --output-dir .\test_outputs\experiments\density_sweep_dryrun
  ```

- Minimal hardware run (single cell):
  ```bash
  python -m FloppyAI.src.main experiment \
    --patterns prbs7 --densities 1.0 --tracks 80 --sides 0 \
    --revs 1 --reps 1 \
    --output-dir .\test_outputs\experiments\density_sweep_sample
  ```

- Direct generation (if running pieces manually):
  ```bash
  python -m FloppyAI.src.main generate 80 0 \
    --revs 1 --density 1.5 --pattern prbs7 --seed 42 \
    --analyze --output-dir .\test_outputs\experiments\gen_density_1p5
  ```

## Metrics & Reports

- Edge jitter vs density; instability score vs density
- Realized vs intended density
- Spectral features; correlation with intended sequence
- Per-density summary table and trend plots

## Protocol (per density)

1. Generate PRBS7 at target density.
2. Write to track 80 side 0 (or matrix-defined track).
3. Read back N revs.
4. Analyze and compute metrics.
5. Append to manifest and report; cooldown.

## Interpretation

- Expect a threshold where jitter grows sharply; note media- and track-dependent behavior.
- Compare outer tracks to inner tracks if expanded beyond 79–83.

## Artifacts & Directory Layout

- `experiment_manifest.json` documenting matrix and per-run metadata
- Per-run folders under the selected `--output-dir`
- Summary `experiment_report.md` with tables and links to plots/PNGs

## Reproducibility

- Fix `--seed` for generation
- Log all parameters and environment details

## Appendix: Parameters

- `--density` scale factor (>1 increases nominal bit rate)
- `--pattern prbs7` deterministic sequence for correlation

## Changelog

- v0.1 — initial draft
