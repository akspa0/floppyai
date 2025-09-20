# Experiment 01 — Extreme Streams

Push the recording limits of floppy media using controlled flux patterns and density scaling. Write patterns, re-read, and analyze deviations between intended and observed behavior.

## Objectives & Hypotheses

- Determine how far we can scale density before instability/jitter explodes.
- Observe effects of pattern structure (random vs PRBS vs alternating vs long runs) on readability and stability.
- Identify spectral/temporal signatures of failure modes (e.g., DC bias, saturation, jitter growth).

## Safety & Setup

- Use sacrificial media only. Label clearly.
- Start on outer tracks (e.g., 79–83) and limit revs.
- Cooldown between writes (e.g., 10–20s).
- Enforce bounds: `--density ≤ 2.5`, reasonable run-lengths, bounded experiment duration.
- Dry-run first with `--simulate`.

## Experiment Matrix

- Patterns (`--pattern`):
  - `random` — pseudo-random bits (controlled by `--seed`)
  - `prbs7`, `prbs15` — pseudo-random binary sequences
  - `alt` — alternating with configurable run-lengths (e.g., 01, 0011, 000111, ... via `--runlen`)
  - `runlen` — extreme long runs of identical bits
  - `chirp` — sweep nominal cell length within revs (`--chirp-start-ns`, `--chirp-end-ns`)
  - `dc_bias` — modulate duty cycle / transition spacing asymmetrically (`--dc-bias`)
  - `burst` — interleave clean segments with noise segments (`--burst-period`, `--burst-duty`, `--burst-noise`)
- Densities: e.g., 0.8, 1.0, 1.2, 1.5, 2.0
- Tracks: outer band (79–83)
- Sides: 0, 1
- Revs: 1–2 per write
- Repetitions: 2 per cell in the matrix (optional)

## CLI (Proposed)

The experiment harness will orchestrate generate → write → read → analyze → report.

- Dry-run (no hardware):
  ```bash
  python FloppyAI/src/main.py experiment \
    --patterns prbs7,alt,random \
    --densities 0.8,1.0,1.2 \
    --tracks 80-83 --sides 0 \
    --revs 1 --reps 1 \
    --simulate --output-dir .\test_outputs\experiments\dryrun
  ```

- Minimal hardware run (safe defaults, single cell):
  ```bash
  python FloppyAI/src/main.py experiment \
    --patterns prbs7 \
    --densities 1.0 \
    --tracks 80 --sides 0 \
    --revs 1 --reps 1 \
    --output-dir .\test_outputs\experiments\prbs7_t80_s0
  ```

- Direct pattern generation (without harness), using extended `generate` flags:
  ```bash
  python FloppyAI/src/main.py generate 80 0 \
    --revs 1 --density 1.2 --pattern prbs7 --seed 42 \
    --analyze --output-dir .\test_outputs\experiments\gen_only
  ```

## Metrics & Reports

- Edge jitter: stddev of interval error vs intended timing
- Instability score (existing)
- Realized density vs intended density
- Spectral features (FFT, spectral flatness, dominant peaks)
- Correlation vs intended sequence (when deterministic)
- Bitcell histogram skew (DC bias proxy)

Outputs per run:
- `surface_map.json`, `overlay_debug.json` (if overlays enabled)
- Composite and surface PNGs, interval histograms, optional spectrum plots
- `experiment_manifest.json` (matrix configuration, per-run metadata)
- `experiment_report.md` (tables + links to artifacts)

## Protocol (per cell)

1. Generate pattern to `.raw` with the given density and revs.
2. Write the `.raw` to track/side.
3. Read N revs back to a new `.raw` capture.
4. Analyze capture and compute metrics.
5. Append metrics to report; cool down before next cell.

## Interpretation

- Look for non-linear growth in jitter/instability as density increases.
- Random vs PRBS: similar spectral flatness but PRBS is deterministic → better correlation checks.
- Alternating patterns: highlight channel bandwidth; watch for harmonic peaks.
- Run-length extremes and dc_bias: expose baseline wander / AGC limits.
- Chirp: identify ranges where timing recovery loses lock.

## Notes

- All commands should be run from the repository root with `python FloppyAI/src/main.py` (module syntax also works).
- Start with `--simulate` to validate directory structure and manifests.
- Use `--media-type` (e.g., `35HD`) and consistent `--rpm` or `--profile` for comparability.
