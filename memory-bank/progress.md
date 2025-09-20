# Progress — FloppyAI

Last updated: 2025-09-20 02:28 (local)

## What Works
- Modular CLI wiring: `cmd_stream_ops.py`, `cmd_corpus.py`, `cmd_diff.py` integrated into `main.py`.
- `main.py` now delegates to modules and uses `utils.json_io.dump_json` for JSON writes (major sites replaced).
- Documentation refreshed: README, usage, cheatsheet, docs index, and experiments docs.
- Experiments docs created: `01-extreme-streams.md`, `02-high-density-encoding.md`, `_template.md`.
- **Experiments scaffolding completed**: `cmd_experiments.py` with matrix orchestration, `analysis/metrics.py` with comprehensive metrics and plotting.
- **Enhanced generate command**: Added `--pattern`, `--seed`, `--density` flags with support for `random`, `prbs7`, `alt`, `zeros`, `ones`, `sweep` patterns.
- **Experiment CLI integrated**: `python -m FloppyAI.src.main experiment matrix` command available.

## What's Left
- Phase 2: finish migrating leftover `json.dump` sites to `utils.json_io.dump_json` (verify all in `main.py` and commands).
- Phase 3: move `analyze_disk()` implementation into `analysis/analyze_disk.py` and update CLI to prefer shim.
- Add safety gates for hardware runs with `--simulate` default and outer track preferences.

## Current Status
- Phase 1: completed.
- Phase 2: in progress.
- Phase 3: pending.
- Experiments: **scaffolding complete**; ready for testing and refinement.

## Known Issues / Risks
- Subprocess `-m` invocations require repo root as CWD; doc emphasizes this.
- Hardware runs require safety gates; defaults will err on caution.
- LLM summaries require an environment-provided endpoint; not always available.
- **New**: Experiment matrix runs require KryoFlux hardware for full testing; simulation mode available.

## Next Steps
- Test experiment matrix functionality with `--simulate` mode.
- Migrate `analyze_disk()` to `analysis/` and update imports; keep CLI backward compatible.
- Add safety confirmation flows and track range preferences for hardware experiments.
- Refine pattern generation algorithms and add more pattern types.

## References
- README: `../README.md`
- Docs hub: `../docs/index.md`
- Experiments hub: `../docs/experiments/index.md`
