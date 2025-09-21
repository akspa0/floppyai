# Product Context — FloppyAI

FloppyAI provides a surface-first analysis of floppy media using KryoFlux raw flux streams. It focuses on visualizing and quantifying physical recording characteristics rather than decoding sectors.

## Why this exists
- To map and understand magnetic surface quality for preservation and experimental encoding.
- To compare multiple reads, build corpora, and identify safe regions for dense writes.
- To run controlled experiments that push recording limits and document outcomes.

## Problems it solves
- Lack of tooling that treats flux as a first-class signal for visualization and metrics.
- Difficulty comparing multiple reads or many disks consistently.
- Fragmented workflows for generating, writing, capturing, and analyzing experimental streams.

## How it should work
- Run from repo root using module syntax: `python -m FloppyAI.src.main`.
- Each CLI subcommand has a clear purpose and consistent output layout under `test_outputs/<timestamp>/` or a user-provided `--output-dir`.
- JSON artifacts are consistent and machine-readable (centralized encoder).
- Experiments are reproducible and documented.

## User experience goals
- Simple CLI that favors sensible defaults and safe behavior around hardware.
- Rich visuals (polar surfaces, composites) and clear metrics.
- Documentation-first: README, usage, cheatsheet, and experiments index.

## Key workflows
- `analyze_disk` → per-disk surface maps and visuals.
- `analyze_corpus` → aggregate across disks with optional LLM summary.
- `compare_reads` → diff metrics across multiple reads.
- `generate`, `write`, `read` → building blocks for experiments.
- Experiments (planned): orchestrated generate→write→read→analyze→report.

## References
- README: `../README.md`
- Docs hub: `../docs/index.md`
- Experiments hub: `../docs/experiments/index.md`
