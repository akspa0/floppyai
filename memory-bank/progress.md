# Progress — FloppyAI

Last updated: 2025-09-22 16:40 (local)

## What Works
- Modular CLI wiring: `cmd_stream_ops.py`, `cmd_corpus.py`, `cmd_diff.py` integrated into `main.py`.
- `main.py` now delegates to modules and uses `utils.json_io.dump_json` for JSON writes (major sites replaced).
- Documentation refreshed: README, usage, cheatsheet, docs index, and experiments docs.
- Experiments docs created: `01-extreme-streams.md`, `02-high-density-encoding.md`, `_template.md`.
- **Experiments scaffolding completed**: `cmd_experiments.py` with matrix orchestration, `analysis/metrics.py` with comprehensive metrics and plotting.
- **Enhanced generate command**: Added `--pattern`, `--seed`, `--density` flags with support for `random`, `prbs7`, `alt`, `zeros`, `ones`, `sweep` patterns.
- **Experiment CLI integrated**: `python -m FloppyAI.src.main experiment matrix` command available.
- Angular‑resolved instability maps integrated and bright by default (magma_r with percentile contrast) in:
  - `render_instability_map()` (both sides)
  - `render_side_report()` middle‑left panel
  - `render_disk_dashboard()` row 2
- Sector overlays and per‑track angular templates render consistently across disk surface, side reports, and dashboard; intra‑wedge peak ticks included.
- CLI simplified with profile‑driven overlays and new profiles: `35DDGCR`, `35HDGCR`, `525DDGCR`.
  - `--overlay-mode` defaults to `auto`; analyzer picks from profile.
  - GCR candidates auto‑selected by profile; still overridable.
- Docs updated: README and `docs/usage.md` reflect simplified commands and brighter instability visuals.
- New concept page `docs/instability.md` created; cheatsheet updated with an instability quick reference.
- New CLI `image2flux` subcommand scaffolding added (angular‑only MVP) in `src/encoding/image2flux.py` and wired in `main.py`.
- STREAM generator defaults finalized: OOB‑first; first OOB at byte 0 is KFInfo `KryoFlux stream - version 3.50` (no NUL); StreamInfo initial+periodic with payload SP equal to actual ISB position; Index per rev; StreamEnd then EOF. HxC loads our generated streams by default.

## What's Left
- Phase 2: finish migrating leftover `json.dump` sites to `utils.json_io.dump_json` (verify all in `main.py` and commands).
- Phase 3: move `analyze_disk()` implementation into `analysis/analyze_disk.py` and update CLI to prefer shim.
- Add safety gates for hardware runs with `--simulate` default and outer track preferences.
- Document Linux-side hardware scripts for DTC host usage (sudo), and reference them from docs (`usage.md`, experiments).
- Establish manual cross-machine workflow for experiments: generate on Windows → transfer to Linux → run scripts → bring captures back → analyze on Windows.
- Optional: add `--overlay-fill` to softly shade sectors for visibility.
- Continue Phase 2 cleanup: ensure all JSON writes use `utils.json_io.dump_json`.
- Validate overlay heuristics across more datasets (MFM and GCR; zoned tracks) and tune defaults if needed.
- Implement `structure_finder` (pattern reconstruction) and integrate round‑trip metrics (correlation, phase), add end‑to‑end tutorial assets.
- Add Linux write/read script in `scripts/linux/` and document exact flags and safety notes.
- Validate DTC write‑from‑stream (`-i0 -w`) using correct `-f` prefix (`.../track`) and a small track set. Publish confirmed command lines.

## Current Status
- Phase 1: completed.
- Phase 2: in progress.
- Phase 3: pending.
- Experiments: **scaffolding complete**; ready for testing and refinement.
- Decision recorded: do not implement or rely on a cross-machine DTC wrapper; hardware steps will be executed manually on the Linux DTC host via bash scripts.
- Visualization and overlay pipeline stabilized; default outputs are forensic‑rich without per‑track PNG spam.
- Profiles reduce CLI complexity while preserving expert overrides.
- Cross‑machine workflow remains manual via Linux DTC scripts per project decision.
 - STREAM exporter defaults locked; HxC validation positive (loads by default with OOB‑first header). DTC write validation pending.

## Known Issues / Risks
- Subprocess `-m` invocations require repo root as CWD; doc emphasizes this.
- Hardware runs require safety gates; defaults will err on caution.
- LLM summaries require an environment-provided endpoint; not always available.
- **New**: Experiment matrix runs require KryoFlux hardware for full testing; simulation mode available.
- **Cross-machine**: No SSH orchestration from Windows. Manual artifact transfer between Windows and Linux is assumed; ensure consistent labeling and paths.
- Weak or highly zoned GCR may still lower side‑level confidence; per‑track inspection in `surface_map.json` recommended.
- LLM summary features depend on local endpoint availability.

## Next Steps
- Test experiment matrix functionality with `--simulate` mode.
- Migrate `analyze_disk()` to `analysis/` and update imports; keep CLI backward compatible.
- Add safety confirmation flows and track range preferences for hardware experiments.
- Refine pattern generation algorithms and add more pattern types.
- Add and document Linux-side DTC scripts; verify they produce captures compatible with `analyze`/`analyze_disk`.
- Gather feedback on the new defaults from recent runs; adjust contrast percentile if needed.
- Add quickstart examples for each profile in docs/cheatsheet.

## References
- README: `../README.md`
- Docs hub: `../docs/index.md`
- Experiments hub: `../docs/experiments/index.md`
