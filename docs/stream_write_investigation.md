# STREAM Write Investigation Log

Date: 2025-09-22

Summary
- HxC opens generated STREAM files; Side 0 shows rough flux; Side 1 may be sparse/garbled depending on pattern.
- DTC refuses to open generated files for writing (`Image name:` / `Can't open image file:`) despite strict CLI ordering and multiple base variants.

Commands Attempted (representative)
- From directory containing `trackNN.S.raw`:
  - `sudo /usr/bin/dtc -ftrack -i0 -d0 -s0 -e0 -g0 -w`
  - Tried base variants: `-ftrack.raw`, `-f$PWD/track`, `-f$PWD/track.raw`.
- Scripted attempts (`scripts/linux/dtc_write_read.sh`) try these in sequence with logs.

Variants Generated & Tried
- Header mode: `ascii` vs `oob`.
- OOB sample clock (type 8): present vs absent.
- Initial index OOB at start: present vs absent.
- All rejected by DTC write on the test host.

Spec Alignment (rev 1.1)
- Implemented C2 encoding map and OOB types (2 index with u16 timer, 3 end, 4 info, 8 sample clock).
- Info text: `KryoFlux stream - version 3.50\x00`.
- Index payload currently timer=0 (placeholder) until we compute from intervals.

Next Actions (tomorrow)
- Capture known-good DTC reference tracks (00.0 and 00.1) via `-i0`.
- Implement `tools/kfx_probe.py` to dump OOB/type/len/offset and opcode histogram.
- Compare reference vs generated; adjust exporter to match accepted OOB/header shape.
- Re-test a minimal single-track write using the matched variant.

Notes
- See `docs/kryoflux_stream_notes.md` for protocol summary.
- README updated with a Status section and guidance to use HxC for writing until resolved.
