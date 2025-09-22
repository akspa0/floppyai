# Project Brief — FloppyAI

FloppyAI analyzes KryoFlux raw flux streams to understand and visualize the magnetic surface of floppy disks. It provides per‑track metrics, polarization maps, optional format‑aware overlays (MFM/GCR), corpus aggregation, and comparison tools. A new initiative adds repeatable experiments to probe the physical limits of the recording medium.

## Problem Statement
- Existing tools focus on sector decoding. We need a surface‑first approach: operate directly on flux to characterize media quality, stability, and patterns.
- The monolithic CLI needed refactoring to promote modularity, reuse, and maintainability.

## Goals
- Modular CLI with subcommands under `src/` wired via `main.py`.
- High‑quality visuals and metrics from flux data (no filesystem assumptions).
- Reproducible CLI and outputs that default to timestamped directories but accept `--output-dir` overrides.
- Experiments framework to generate/write/capture/analyze extreme streams.

## Success Criteria
- Run all commands from the `FloppyAI/` directory: `python -m src.main`.
- `main.py` delegates to modules: `cmd_stream_ops.py`, `cmd_corpus.py`, `cmd_diff.py`, and future `cmd_experiments.py`.
- JSON I/O goes through `utils/json_io.py` (`dump_json`) with a consistent encoder.
- Docs: README, usage guide, cheatsheet, and experiments docs are up to date and cross‑linked.
- Experiments have safety gates, manifests, and metrics; simulate‑first flows work.

## Scope (Current)
- CLI commands: analyze, read, write, generate, encode, analyze_disk, analyze_corpus, compare_reads, classify_surface, plan_pool.
- MFM/GCR overlay heuristics with tunable parameters.
- Experiments documentation and soon, an orchestration command.

## Out of Scope (Now)
- Full sector decoding and filesystem recovery.
- Advanced ML training; current LLM usage is for summaries only.

## References
- README: `../README.md`
- Docs hub: `../docs/index.md`
- Experiments hub: `../docs/experiments/index.md`
