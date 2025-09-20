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
  - Experiment patterns:
  - `random`: Random flux intervals within density bounds
  - `prbs7`: Pseudo-random binary sequence using 7-bit LFSR
  - `alt`: Alternating long/short cell pattern
  - `zeros`: All-zeros pattern (long cells)
  - `ones`: All-ones pattern (short cells)
  - `sweep`: Frequency sweep pattern across cell length range
- Pattern generation includes reproducible seeding via `--seed` parameter.
- Density scaling via `--density` multiplier affects base cell length.

- The CLI sometimes shells out to `python -m FloppyAI.src.main ...` (e.g., corpus generating missing maps):
  - Set subprocess CWD to the repo root for reliable module resolution.
  - Prefer internal calls where possible to avoid environment skew; use subprocess as a fallback for isolation.
- New experiment subcommand uses subprocess orchestration: `python -m FloppyAI.src.main experiment matrix` runs generate→write→read→analyze cycles.
- Safety defaults: experiments prefer `--simulate` mode and outer tracks (0-9) to minimize hardware risk.

## Pending/Planned Patterns

- Move `analyze_disk()` out of `main.py` into `analysis/analyze_disk.py` and import it from CLI (Phase 3).
- Add safety confirmation flows for hardware experiments (cooldown periods, sacrificial media warnings).
- Expand pattern generation with more sophisticated test patterns (frequency sweeps, error injection, etc.).
- Implement real-time experiment monitoring and early termination for unstable runs.
