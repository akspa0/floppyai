# Experiment 01 — Extreme Streams

Push the recording limits of floppy media using controlled flux patterns and density scaling. Write patterns, re-read, and analyze deviations between intended and observed behavior.

## Objectives & Hypotheses

- Determine how far we can scale density before instability/jitter explodes.
- Observe effects of pattern structure (random vs PRBS vs alternating vs long runs) on readability and stability.
- Identify spectral/temporal signatures of failure modes (e.g., DC bias, saturation, jitter growth).
- Demonstrate an end‑to‑end pipeline that takes an input image → generates a flux stream → writes to disk on Linux (dtc) → reads back → analyzes and reconstructs the intended structure.

## Safety & Setup

- Use sacrificial media only. Label clearly.
- Start on outer tracks (e.g., 79–83) and limit revs.
- Cooldown between writes (e.g., 10–20s).
- Enforce bounds: `--density ≤ 2.5`, reasonable run-lengths, bounded experiment duration.
- Dry-run first with `--simulate`.
- Hardware on Linux DTC host (sudo): Do not orchestrate DTC from Windows. Generate streams on Windows, then on the Linux host run bash scripts to `dtc write` and `dtc read`. Transfer captures back to Windows for analysis. See `docs/usage.md` → Cross‑machine workflow.

Linux host write/read script (concept):
```
#!/usr/bin/env bash
set -euo pipefail

# Example usage:
# ./write_read_roundtrip.sh --write /path/to/image2flux_foo_t80_s0.raw --track 80 --side 0 --revs 16 --drive 0 --cooldown 10 --out /captures/foo

WRITE_RAW=""
TRACK=80
SIDE=0
REVS=16
DRIVE=0
COOLDOWN=10
OUTDIR="./captures"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --write) WRITE_RAW="$2"; shift; shift;;
    --track) TRACK="$2"; shift; shift;;
    --side) SIDE="$2"; shift; shift;;
    --revs) REVS="$2"; shift; shift;;
    --drive) DRIVE="$2"; shift; shift;;
    --cooldown) COOLDOWN="$2"; shift; shift;;
    --out) OUTDIR="$2"; shift; shift;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

TS=$(date +%Y%m%d_%H%M%S)
RUN="$OUTDIR/$TS"
mkdir -p "$RUN"
LOG="$RUN/dtc_roundtrip.log"

{
  echo "DTC roundtrip @ $TS"
  echo "WRITE_RAW=$WRITE_RAW TRACK=$TRACK SIDE=$SIDE REVS=$REVS DRIVE=$DRIVE"
  echo "Cooling $COOLDOWN s pre-write"; sleep "$COOLDOWN"
  sudo dtc -d "$DRIVE" -i 0 -t "$TRACK" -s "$SIDE" -f "$WRITE_RAW" write
  echo "Cooling $COOLDOWN s pre-read"; sleep "$COOLDOWN"
  OUTRAW="$RUN/${TRACK}.${SIDE}.raw"
  sudo dtc -d "$DRIVE" -i 0 -t "$TRACK" -s "$SIDE" -r "$REVS" -f "$OUTRAW" read
  echo "DONE: $OUTRAW"
} |& tee "$LOG"
```
Adjust to your environment; sudo is typical. Keep cooldowns and high revs for forensic‑rich captures.

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

- Image→Flux generation (Windows):
  ```bash
  python FloppyAI/src/main.py image2flux .\assets\checker_64.png 80 0 \
    --revs 1 --angular-bins 720 --on-count 4 --off-count 1 --interval-ns 2000 \
    --output-format kryoflux --output-dir .\test_outputs\experiments\images
  ```
  Transfer the resulting `.raw` to the Linux host and use the script above to write+read.

## Metrics & Reports

- Edge jitter: stddev of interval error vs intended timing
- Instability score (existing)
- Realized density vs intended density
- Spectral features (FFT, spectral flatness, dominant peaks)
- Correlation vs intended sequence (when deterministic)
- Bitcell histogram skew (DC bias proxy)

Round‑trip acceptance for image patterns:
- Angular‑only mapping: recovered wedge correlation ≥ 0.8; phase alignment within ±2 angular bins
- Report includes recovered thumbnail and correlation score (future: structure_finder integration)

Outputs per run:
- `surface_map.json`, `overlay_debug.json` (if overlays enabled)
- Composite and surface PNGs, interval histograms, optional spectrum plots
- `experiment_manifest.json` (matrix configuration, per-run metadata)
- `experiment_report.md` (tables + links to artifacts)

## Protocol (per cell)

1. Generate pattern to `.raw` with the given density and revs.
2. If using Linux DTC host: transfer `.raw` to Linux and run bash scripts (sudo) to write and read back captures.
3. Otherwise (local DTC): write the `.raw` to track/side and read N revs back to a new `.raw` capture.
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
