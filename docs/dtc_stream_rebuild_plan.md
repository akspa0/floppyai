# DTC Stream Rebuild Plan (CAPSImg-Grounded)

## Goals
- Produce KryoFlux STREAM files that DTC accepts without error.
- Eliminate guesswork by matching DTC’s own output byte-for-byte for OOB structure and payload formatting.
- Keep HxC-compatible path intact; add a strict DTC path for hardware workflows.

## References (local)
- CAPSImg sources: `lib/capsimg_source_windows/`
  - Detection logic: `CAPSImg/DiskImageFactory.cpp::IsKFStream` (OOB-first; Info type at offset 0; looks for "KryoFlux" in Info payload)
  - Stream classes: `CStreamImage`, `CStreamCueImage` (we do not require cue files)
- Real DTC captures: use your recent working set under `FloppyAI/stream_dumps/...`
- Our writers:
  - HxC/default path: `src/stream_export.py` (kept unchanged)
  - DTC-style path (baseline): `src/stream_export_dtc.py`

## Constraints and Decisions
- DTC path: OOB-first streams (no ASCII preamble), per CAPS detection logic.
- First OOB must be Info (0x0D, 0x04, sizeLE, payload) and early payload must contain the string "KryoFlux".
- We match DTC’s header content and ordering exactly:
  - KFInfo #1: `host_date=YYYY.MM.DD, host_time=HH:MM:SS, hc=0`
  - KFInfo #2: `name=KryoFlux DiskSystem, version=V, date=Mon DD YYYY, time=HH:MM:SS, hwid=1, hwrv=1, hs=1, sck=S, ick=I`
  - Initial `StreamInfo` (SP=0, ms=0), periodic `StreamInfo` blocks with SP equal to ISB bytes at insertion, and ms matching DTC cadence
  - Index (0x02) only at rev boundaries; `SampleCounter` equals last emitted flux ticks; `IndexCounter` derived from sck->seconds->ick using DTC’s rounding
  - At least one flux cell after the final Index; then `StreamEnd` (SP, RC=0); then EOF sentinel (0x0D with size 0x0D0D => 4 bytes 0x0D)

## Work Plan
- Step 1. Implement OOB diff tooling
  - File: `tools/stream_oob_diff.py`
  - Parse OOB blocks from two streams and print a structured diff: types, positions, sizes, and decoded payloads for Info/StreamInfo/Index/StreamEnd/EOF.
  - Use this to drive byte-accurate convergence against a real capture.

- Step 2. Create strict DTC writer module
  - File: `src/stream_export_dtc_strict.py`
  - Emit OOB-first blocks matching DTC format; port text formatting (spacing, date formats, float precision for sck/ick) based on real capture.
  - Maintain structural fixes: `Index.SampleCounter = last_flux_ticks`; add a small dummy flux after final Index; `StreamEnd` then EOF.

- Step 3. Wire generator with strict writer option
  - File: `tools/patterns_to_stream_set.py`
  - Add `--writer strict-dtc` that imports and invokes `write_kryoflux_stream_dtc_strict`.
  - Force header mode to OOB-first for this path.

- Step 4. Validate & Iterate
  - Use `tools/stream_oob_diff.py` to compare `strict-dtc` output with a real DTC capture (same track/side), adjust until OOB diff is clean.
  - Run `tools/kfx_probe.py --validate` to confirm structural integrity.
  - On Linux DTC host, run `scripts/linux/dtc_write_read_set.sh --image-dir <dir> --drive 0 --tracks <range> --sides <list>`; inspect the script’s log for the exact dtc command and output.

## Acceptance Criteria
- OOB diff shows identical OOB sequence (type, order, sizes, payload text/fields) between our output and a real DTC capture for the same reference case.
- `kfx_probe.py --validate` reports OK.
- `dtc_write_read_set.sh` successfully writes `strict-dtc` streams on the Linux host.

## Risks & Mitigations
- DTC build variations: mitigate by basing on your exact real capture and dtc version.
- Float precision in sck/ick: tune formatting to match reference text exactly.
- Periodic `StreamInfo` cadence: mirror reference SP and ms pattern.

## Next Actions
- Implement `tools/stream_oob_diff.py`.
- Implement `src/stream_export_dtc_strict.py`.
- Add `--writer strict-dtc` to `tools/patterns_to_stream_set.py`.
- Run OOB diff against your provided reference and iterate until identical.
