# Experiment NN — Title

Short description of the experiment and what it validates.

## Objectives & Hypotheses

- What are we trying to learn?
- What outcomes would support or refute the hypothesis?

## Safety & Setup

- Sacrificial media, track ranges, rev limits, cooldowns
- Environment assumptions (hardware/software)
- Dry-run flow (`--simulate`) and confirmation gates

## Experiment Matrix

- Variables (patterns, densities, tracks, sides, revs, repetitions)
- Ranges and defaults
- Any constraints/bounds

## CLI

- Recommended invocation from repository root:
  ```bash
  python -m FloppyAI.src.main --help
  ```
- Commands used (copy/paste examples):
  - Generation / Write / Read / Analyze
  - Orchestration (if using an `experiment` subcommand)

## Metrics & Reports

- Quantitative metrics (definitions)
- Visuals/plots generated
- Paths to outputs and how to read them

## Protocol (per matrix cell)

1. Generate data (include parameters)
2. Write to track/side
3. Read back N revs
4. Analyze and compute metrics
5. Record results (append to manifest/report)

## Interpretation

- How to interpret elevated/low values in each metric
- Known artifacts/edge cases
- Example interpretations

## Artifacts & Directory Layout

- `experiment_manifest.json` schema (high-level)
- Per-run folders and naming conventions
- Where to find summary reports and key artifacts

## Reproducibility

- Seeds, parameter logs, and environment capture
- How to re-run a subset or single matrix cell

## Appendix: Parameters

- Table or list of parameters and meanings
- Defaults vs experiment-specific overrides

## Changelog

- v0.1 — initial draft
