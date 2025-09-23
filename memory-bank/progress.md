# Progress — FloppyAI

Last updated: 2025-09-23 03:11 (local)

## What Works
- **Strict DTC writer accepted by dtc:** Our streams are now recognized and dtc starts writing.
- **CAPS/C2‑aligned exporter:** `src/stream_export_dtc.py` + `src/stream_export_dtc_strict.py` implement OOB‑first Info‑first, Index payload `(SP, last_flux_ticks, round(elapsed_sck_seconds*ick_hz))`, dummy flux after last index, StreamEnd then EOF.
- **Deterministic headers:** KFInfo sck/ick text snapped to reference; Info payloads padded to 47/141 bytes.
- **Generator profiles:** `tools/patterns_to_stream_set.py` now supports `--profile` presets and `--pattern image` (silkscreen) mapping rows→tracks; minimal CLI typing.
- **dtc Linux wrapper updated:** Uses `-f <prefix>` spacing and logs exact commands; generates both `trackNN.S.raw` and `NN.S.raw` for max compatibility.

## What's Left
- Implement `tools/structure_finder.py` to reconstruct per‑side composites from read‑back sets (no per‑track PNG spam) and compute correlation to intended pattern/image.
- Add `docs/silkscreen_pipeline.md` and profile quickstart in `docs/profiles.md`.
- Optional: externalize profiles to `profiles.yaml` for easier user customization.
- Phase 2/3 CLI cleanup tasks continue (JSON I/O centralization, `analysis/` migration).

## Current Status
- Phase 1: completed.
- Phase 2: in progress.
- Phase 3: pending.
- Experiments: **scaffolding complete**; ready for testing and refinement.
- Decision recorded: do not implement or rely on a cross-machine DTC wrapper; hardware steps will be executed manually on the Linux DTC host via bash scripts.
- Strict DTC writer: implemented and accepted by dtc (begins writing), confirming stream format correctness.
- Hardware write errors on current host traced to USB controller behavior; move to older PC/USB2‑native recommended.
- Profiles and image silkscreen supported; ready for end‑to‑end tests once hardware path is stable.
- Visualization and overlay pipeline stabilized; default outputs are forensic‑rich without per‑track PNG spam.
- Profiles reduce CLI complexity while preserving expert overrides.
- Cross‑machine workflow remains manual via Linux DTC scripts per project decision.
 - STREAM exporter defaults locked; HxC validation positive (loads by default with OOB‑first header). DTC write validation pending.

## Known Issues / Risks
- USB 3.x controllers often deprecate/poorly support bulk streaming required by KryoFlux; streaming device errors likely on modern hosts.
- Prefer older PCs/USB2 hubs/controllers; ensure stable 5V/12V for 5.25" drives; confirm write‑protect off.
- Subprocess `-m` invocations require `FloppyAI/` as CWD; docs emphasize this.
- Hardware runs require safety defaults; use sacrificial media and small track ranges first.

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
