# Progress — FloppyAI

Last updated: 2025-09-20 02:06 (local)

## What Works
- Modular CLI wiring: `cmd_stream_ops.py`, `cmd_corpus.py`, `cmd_diff.py` integrated into `main.py`.
- `main.py` now delegates to modules and uses `utils.json_io.dump_json` for JSON writes (major sites replaced).
- Documentation refreshed: README, usage, cheatsheet, docs index, and experiments docs.
- Experiments docs created: `01-extreme-streams.md`, `02-high-density-encoding.md`, `_template.md`.

## What's Left
- Phase 2: finish migrating leftover `json.dump` sites to `utils.json_io.dump_json` (verify all in `main.py` and commands).
- Phase 3: move `analyze_disk()` implementation into `analysis/analyze_disk.py` and update CLI to prefer shim.
- Implement experiment harness (`cmd_experiments.py`) and metrics (`analysis/metrics.py`).
- Extend `generate` patterns and add safety gates.

## Current Status
- Phase 1: completed.
- Phase 2: in progress.
- Phase 3: pending.
- Experiments: planned with docs; code scaffolding next.

## Known Issues / Risks
- Subprocess `-m` invocations require repo root as CWD; doc emphasizes this.
- Hardware runs require safety gates; defaults will err on caution.
- LLM summaries require an environment-provided endpoint; not always available.

## Next Steps
- Create `cmd_experiments.py` orchestrator with `--simulate` default; add matrix inputs.
- Add `analysis/metrics.py` with jitter, spectrum, correlation metrics, and plots.
- Extend `cmd_stream_ops.generate` with `--pattern` and `--seed` and the initial set: `random`, `prbs7`, `alt`.
- Migrate `analyze_disk()` to `analysis/` and update imports; keep CLI backward compatible.

## References
- README: `../README.md`
- Docs hub: `../docs/index.md`
- Experiments hub: `../docs/experiments/index.md`
