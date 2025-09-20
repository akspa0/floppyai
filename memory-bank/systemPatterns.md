# System Patterns — FloppyAI

This document captures architecture, design decisions, and recurring patterns used across the project.

## Architecture Overview

- Entry point: `src/main.py` (CLI)
  - Uses `argparse` with subcommands that delegate to command modules.
  - Must be run from the repository root using module syntax:
    ```bash
    python -m FloppyAI.src.main --help
    ```
- Command modules:
  - `src/cmd_stream_ops.py` — analyze single stream, read, write, generate, encode
  - `src/cmd_corpus.py` — corpus aggregation; can call `analyze_disk` on missing inputs
  - `src/cmd_diff.py` — compare multiple reads of the same disk
  - `src/cmd_experiments.py` — (planned) orchestrate experiment matrices
- Analysis modules:
  - `src/analysis/analyze_disk.py` — shim `run(args)` that delegates to `main.analyze_disk` (migration target for Phase 3)
  - `src/analysis/metrics.py` — (planned) edge jitter, spectrum, correlation, summaries
- Utilities:
  - `src/utils/json_io.py` — centralized JSON dumping (`dump_json`) with a custom encoder
  - `src/utils/io_paths.py` — output directory handling (`get_output_dir`), safe labeling helpers

## Data Model & Outputs

- Per-run outputs (by default under `test_outputs/<timestamp>/`, or under `--output-dir`):
  - `surface_map.json` — main per-track/side analysis payload
  - `overlay_debug.json` — overlay-only snapshot
  - Images: composite, polar surfaces, per-side surfaces
  - Per-analysis CSVs (e.g., instability summaries)
- Corpus aggregation outputs:
  - `corpus_summary.json`, plots, and montage grids
  - Manifests: `corpus_inputs.txt`, `corpus_missing_inputs.txt`
- Diff outputs:
  - `diff/diff_summary.json`, `diff/diff_densities.csv`

## Conventions & Patterns

- Invocation pattern: run from repo root using `python -m FloppyAI.src.main` to ensure package resolution in command and subprocess flows.
- JSON I/O must use `utils/json_io.dump_json` to normalize numpy, Path, and special values.
- Output directory policy: all commands accept `--output-dir`; if absent, a timestamped folder under `test_outputs/` is created.
- Optional LLM summary:
  - Controlled via `--summarize`, `--lm-host`, `--lm-model`, `--lm-temperature`.
  - LLM JSON is validated/coerced; a text narrative is rendered to `.txt`.
- Overlays:
  - Controlled via `--format-overlay`, `--overlay-mode {mfm,gcr,auto}`, `--angular-bins`, `--overlay-sectors-hint`, `--gcr-candidates`, `--overlay-color`, `--overlay-alpha`.
  - Overlay metadata lives under `global.insights.overlay` and per-track overlay blocks.

## CLI & Subprocess Rules

- The CLI sometimes shells out to `python -m FloppyAI.src.main ...` (e.g., corpus generating missing maps):
  - Set subprocess CWD to the repo root for reliable module resolution.
  - Prefer internal calls where possible to avoid environment skew; use subprocess as a fallback for isolation.

## Pending/Planned Patterns

- Move `analyze_disk()` out of `main.py` into `analysis/analyze_disk.py` and import it from CLI (Phase 3).
- Create `cmd_experiments.py` and `analysis/metrics.py` to support experiment orchestration and reporting.
- Metrics standardization (edge jitter, spectral features, correlation) with reusable plotting helpers.
