# Active Context — FloppyAI

Last updated: 2025-09-22 03:41 (local)

## Current Focus
- STREAM write investigation: DTC currently refuses to open generated STREAM files for writing (`Image name:` / `Can't open image file:`) despite strict CLI ordering and multiple `-f` base variants. HxC can visualize Side 0; Side 1 may be sparse depending on pattern.
- Short‑term objective: Gather known‑good reference STREAMs via DTC capture and align our exporter (OOB/header) to protocol rev 1.1 and empirical DTC expectations.
- Maintain forensic‑rich capture defaults and logging.

## Recent Changes
- STREAM exporter updated (`src/stream_export.py`) to match KryoFlux spec more closely:
  - ISB encoding now uses Flux1 (0x0E..0xFF), Flux2 (0x00..0x07 + 1 byte), Flux3 (0x0C + MSB then LSB), and OVL16 (0x0B) correctly. 0x0C byte order fixed to MSB-first.
  - OOB blocks now spec-compliant:
    - KFInfo (type 0x04): ASCII `sck=..., ick=...\0`.
    - Index (type 0x02): 12-byte LE payload {StreamPosition, SampleCounter, IndexCounter}.
    - StreamEnd (type 0x03): 8-byte LE payload {StreamPosition, ResultCode=0}.
    - EOF (type 0x0D): size 0x0D0D.
  - Added `ick_hz` parameter; we compute counters and track ISB byte positions while encoding.
  - Removed non-spec OOB type 0x08 usage; ASCII preamble remains optional but default header is OOB-first.
- Generator updated (`tools/patterns_to_stream_set.py`): defaults to `--header-mode oob` and threads `--ick-hz` through to the exporter.
- New probe tool added: `tools/kfx_probe.py` dumps OOB blocks, ISB opcode histogram, and basic integrity/EOF checks for STREAM files.
- Linux DTC scripts updated:
  - Write attempts use STREAM mode `-i0 -w` with `-f` prefix first and multiple base fallbacks (`track`, `track.raw`, absolute forms).
  - Read uses `-i0`.
  - Improved logs show exact commands attempted.
- Docs updated:
  - README adds “KryoFlux STREAM writing status”.
  - New docs: `docs/kryoflux_stream_notes.md`, `docs/stream_write_investigation.md`.

## Next Steps
- Capture reference STREAMs via DTC (`-i0`) for a small set of tracks/sides (e.g., 00.0, 00.1).
- Use `tools/kfx_probe.py` to compare reference vs generated (OOB layout, ISB histograms, counters/positions).
- Align any remaining header/order nuances observed in references; then re‑test a minimal DTC write (`-i0 -w`).
- Update docs to finalize the accepted header/OOB shape and remove temporary toggles.

## Open Decisions
- Whether to expose a user knob for instability contrast percentile; currently hard‑coded to a sensible default.
- Optional addition of per‑track overlay JSON for zoned GCR in future iterations.

## Links
- README: `../README.md`
- Docs hub: `../docs/index.md`
- Experiments hub: `../docs/experiments/index.md`
