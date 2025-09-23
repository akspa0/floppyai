# Active Context — FloppyAI

Last updated: 2025-09-23 03:11 (local)

## Current Focus
- Strict DTC writer (OOB‑first, CAPS/C2‑aligned) implemented; dtc now accepts our streams and begins writing.
- Provide easy pattern generation via profiles and add image → flux “silkscreen” mode with sensible defaults.
- Hardware triage for write errors: likely USB controller issues on modern hosts; prefer older PC/USB2‑native for reliable bulk transfers.

## Recent Changes
- Strict DTC writer (`src/stream_export_dtc.py` + `src/stream_export_dtc_strict.py`):
  - OOB‑first Info‑first header per CAPSImg detection; Index payload = (StreamPosition, last_flux_ticks, round(elapsed_sck_seconds*ick_hz)).
  - KFInfo clocks text snapped to reference; Info payloads padded to common sizes (47/141) for deterministic headers.
  - Final Index followed by a dummy flux cell, then StreamEnd and EOF 0x0D0D (4×0x0D).
  - Strict path forces version `3.00s` and emits both `trackNN.S.raw` and `NN.S.raw` names.
- Generator (`tools/patterns_to_stream_set.py`):
  - Added `--writer strict-dtc`; added `--profile` presets (constant‑4us, alt‑4us‑2us, sweep‑2to6us, prbs7‑default, image‑radial).
  - New `--pattern image` (silkscreen), with per‑track row mapping and minimal flags; lazy imports to avoid hard numpy dep.
- Linux DTC wrapper (`scripts/linux/dtc_write_read_set.sh`): use `-f <prefix>` (space) and log exact dtc commands.
- OOB diff tool added: `tools/stream_oob_diff.py` to compare generated vs real DTC OOB sequences.

## Next Steps
- Execute end‑to‑end pattern/image writes on a reliable host (older PC/USB2) to avoid bulk USB streaming issues.
- Implement `tools/structure_finder.py` to build per‑side composites from read‑back captures (no per‑track PNG spam).
- Document profiles and silkscreen pipeline in docs; add quickstart commands.

## Open Decisions
- Whether to expose additional user knobs for image dithering/mapping (keep defaults minimal vs. expert mode).
- Optional addition of per‑track overlay JSON for zoned GCR in future iterations.

## Links
- README: `../README.md`
- Docs hub: `../docs/index.md`
- Experiments hub: `../docs/experiments/index.md`
