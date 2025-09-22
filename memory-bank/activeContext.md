# Active Context — FloppyAI

Last updated: 2025-09-22 16:40 (local)

## Current Focus
- Finalize STREAM exporter defaults that are accepted by tools. HxC now loads our generated streams by default (OOB‑first, version KFInfo at byte 0).
- Next: validate DTC write‑from‑stream acceptance (`-i0 -w`) using proper `-f` prefix and a small track set.
- Keep the end‑user flow simple: no flags required to generate valid streams.

## Recent Changes
- STREAM exporter (`src/stream_export.py`):
  - Default OOB‑first header; first OOB at byte 0 is KFInfo: `KryoFlux stream - version 3.50` (no trailing NUL).
  - StreamInfo (0x01) emitted initially (SP=0, ms=0) and periodically, with payload SP equal to actual ISB bytes at insertion.
  - Index (0x02) per revolution; no initial index by default. Counters computed from `sck_hz=24,027,428.5714285` and `ick_hz=3,003,428.5714285625`.
  - StreamEnd then EOF sentinel.
  - KFInfo hardware/clock strings disabled by default (can be enabled if needed later).
- Generator (`tools/patterns_to_stream_set.py`): defaults to `--header-mode oob`, `--version-string 3.50`.
- Probe (`tools/kfx_probe.py`): recognizes EOF sentinel cleanly and stops at EOF.
- Docs updated: `docs/kryoflux_stream_notes.md` aligned to the finalized defaults.

## Next Steps
- Validate DTC write‑from‑stream (`-i0 -w`) on a small set: ensure `-f` prefix points to `.../track` and that files are `trackNN.S.raw`.
- If DTC needs additional KFInfo (e.g., clocks), add after initial StreamInfo and re‑validate without regressing HxC.
- Update Linux script docs with confirmed DTC command lines and add a safety note.

## Open Decisions
- Whether to expose a user knob for instability contrast percentile; currently hard‑coded to a sensible default.
- Optional addition of per‑track overlay JSON for zoned GCR in future iterations.

## Links
- README: `../README.md`
- Docs hub: `../docs/index.md`
- Experiments hub: `../docs/experiments/index.md`
