# Active Context — FloppyAI

Last updated: 2025-09-20 04:39 (local)

## Current Focus
- Refactor Phase 2: migrate JSON and path helpers (`utils/json_io.py`, `utils/io_paths.py`), continue replacing direct `json.dump` calls.
- Refactor Phase 3: move `analyze_disk()` from `main.py` to `analysis/analyze_disk.py` and let CLI delegate.
- Experiments initiative:
  - Docs created: `docs/experiments/index.md`, `docs/experiments/01-extreme-streams.md`, `docs/experiments/02-high-density-encoding.md`, `_template.md`.
  - Plan to add `cmd_experiments.py` and `analysis/metrics.py`.
  - Hardware orchestration: Do NOT build or rely on a cross-machine DTC wrapper. Hardware steps will be executed on the Linux DTC host via bash scripts; Windows runs default to `--simulate`.

## Recent Changes
- CLI modularization complete for Phase 1; subcommands live in `cmd_stream_ops.py`, `cmd_corpus.py`, `cmd_diff.py`.
- `main.py` updated to import subcommand handlers and use `utils.json_io.dump_json` for writes.
- README updated to emphasize `python -m FloppyAI.src.main` and expanded examples.
- New docs: usage, cheatsheet, docs index; experiments folder introduced.

## Next Steps
- Finish Phase 2 replacements in `main.py` and any command modules still using `json.dump` directly.
- Implement Phase 3 by moving `analyze_disk` logic into `analysis/analyze_disk.py`.
- Add `cmd_experiments.py` (orchestrator) and `analysis/metrics.py`.
- Extend `generate` with advanced patterns (`--pattern`, `--seed`, pattern-specific flags).
- Safety gates for hardware runs; default to `--simulate` and outer tracks.
- Document and use Linux-side bash scripts for hardware I/O on the DTC host (sudo required). Cross-machine flows are manual: copy generated `.raw` to Linux, run scripts, copy captures back, then analyze on Windows.

## Open Decisions
- Orchestration style: internal calls vs subprocess for `experiment` (prefer internal; retain subprocess fallback for isolation/packaging).
- Storage of intended sequences for correlation (store params + checksum vs full intended intervals).
- Cross-machine automation: SSH-based orchestration is explicitly out of scope. Keep hardware steps manual on the Linux host for safety and simplicity.

## Links
- README: `../README.md`
- Docs hub: `../docs/index.md`
- Experiments hub: `../docs/experiments/index.md`
