# Building a Custom KryoFlux USB Writer (C2/pyusb)

## Objective
- Replace `dtc` for write-from-stream by talking directly to the KryoFlux board over USB using the C2 protocol defined in `lib/capsimg_source_windows/Device/C2Comm.h`.
- Gain precise control over pacing/backpressure to mitigate modern USB 3.x bulk streaming issues.
- Keep runs forensic-rich and safe by providing dry-run (WG off), small-track-range defaults, and clear logging.

## Constraints and Preconditions
- Firmware: We will not redistribute KryoFlux firmware. Two safe options:
  - Option A (preferred): Use `dtc` once to load firmware; then our tools attach and verify with `c2InfoFirmware`.
  - Option B (user-supplied): Allow the user to point to their local firmware blob (from their dtc install) to upload at init if supported by the bootloader. We won’t ship the blob.
- Cross-machine policy: Hardware operations occur on the Linux host (per project decision). Windows is for generation/analysis and simulation.
- Recommended hardware: Older PC/USB2-native controller (or known-good USB2 hub) to avoid bulk transfer issues.
- Safety: Start with WG off (dry-run). Only enable WG on sacrificial media after transport is validated.

## Protocol Sources and References
- C2 protocol and stream structures: `lib/capsimg_source_windows/Device/C2Comm.h`
  - Info queries (`c2InfoFirmware`, `c2InfoHardware`)
  - OOB structures and stream semantics
  - Write-stream signatures (WSIB/WSSB) `C2_WSSIGN*` and opcodes (`c2wSetupEnd`, `c2wTime2`, `c2wTableidx`, `c2wIQ*`, `c2wEscape`)
  - Buffer wrap/guard semantics (WA alignment, WRAP, guarded zeros)
- Our CAPS/C2-aligned exporter: `src/stream_export_dtc.py`, `src/stream_export_dtc_strict.py`
  - We already map ns→SCK ticks, produce C2-encoded ISB, and emit OOB Index/End.
  - Deterministic KFInfo payload sizes and clocks matching real captures.

## High-Level Design
- USB layer: `FloppyAI/tools/kf_usb.py`
  - pyusb/libusb device discovery and open/claim.
  - Bulk/control transfers with robust error handling and timeouts.
- C2 helper layer: `FloppyAI/tools/kf_c2.py`
  - Convenience wrappers for GET/SET options, info queries, result polling, and constants from `C2Comm.h`.
- Writer CLI: `FloppyAI/tools/kf_write_stream.py`
  - Convert our ISB into a WSIB/WSSB ‘program’:
    - Use `c2wTime2` or a table (`c2wTableidx`) for cell durations.
    - Program IQ control: WG=off at start; SUSPEND processing; RESUME+WG=on after pre-buffer; END cleanly.
  - Stream payload in small chunks (configurable) with inter-chunk delays; respect WA alignment; insert WRAP if needed.
  - Status/result polling and graceful abort on error.
  - Modes: dry-run (WG off), safe-write (outer tracks, single rev), advanced (user-tuned chunk size/pacing).
- Reader path (later): optional custom C2 reader for symmetry; initially rely on `dtc` for read-back.

## Pacing/Backpressure Strategy
- Pre-buffer: keep WG off; push enough bytes to the ring buffer; then issue IQ to set WG on and RESUME. This reduces underflow risk.
- Chunking: configurable chunk sizes (e.g., 8–32 KiB) and small sleeps (e.g., 2–6 ms) between bursts for modern USB controllers.
- Monitoring: poll device result/state; on any anomaly, stop WG, SUSPEND, and abort cleanly.
- User-tunable: `--chunk-bytes`, `--inter-chunk-ms`, and optional `--prebuffer-ms`.

## Firmware Handling Strategy
- Detect firmware presence via `c2InfoFirmware`.
- If missing:
  - Prompt to run `dtc` once to load firmware; or
  - Accept a `--firmware /path/to/firmware.bin` pointing to the user’s locally installed blob and attempt upload (only if permissible; otherwise fall back to dtc-first).
- Log the chosen path and checks; refuse to proceed without firmware to avoid undefined behavior.

## CLI Design (minimal, dtc-like)
- Required:
  - `--file track00.0.raw` (or `--prefix track --track 0 --side 0`)
  - `--drive 0`, `--track 0`, `--side 0`
- Optional safety:
  - `--wg-off` (dry-run, default), `--force` (enable WG)
  - `--chunk-bytes 16384`, `--inter-chunk-ms 4`, `--prebuffer-ms 50`
  - `--firmware /path/to/firmware.bin`
  - `--rpm`, `--density auto`
  - `--verbose`, `--log /path/to/log.txt`
- Examples:
  - Dry-run: `python tools/kf_write_stream.py --file track00.0.raw --drive 0 --track 0 --side 0 --wg-off`
  - Safe write: `python tools/kf_write_stream.py --file track00.0.raw --drive 0 --track 0 --side 0 --force --chunk-bytes 16384 --inter-chunk-ms 4`

## Phases and Deliverables
- Phase 0: Scaffolding (Linux)
  - `kf_usb.py` + `kf_c2.py`: open device, basic GET/SET, info queries, motor/side/track select.
  - Tool: `kf_diag.py` to print firmware/hardware info and verify board state.
- Phase 1: Dry-run streaming (WG off)
  - Build WSIB/WSSB from a short ISB; stream with SUSPEND/WG off; verify ring buffer behavior and WRAP/WA padding.
  - Add chunking/pacing knobs; log full sequence.
- Phase 2: Real write on sacrificial media
  - Prebuffer, then IQ to set WG on and RESUME; write one track; END cleanly.
  - Read-back with `dtc` (or later our reader) and analyze with `tools/structure_finder.py` (per-side composite).
- Phase 3: Usability polish
  - Helpful errors when firmware missing.
  - Profile presets for pacing settings (e.g., “usb2-safe”, “conservative”).
  - Optional: external `profiles.yaml` for user tuning.
- Optional Phase 4:
  - Custom reader mirroring the same transport and pacing (if needed).
  - Integrate into experiments scripts (Linux) to replace `dtc -w`.

## Acceptance Criteria
- Phase 1: Dry-run completes without device errors; logs validate proper WSIB/WSSB and pacing.
- Phase 2: Single-track write succeeds on older USB2 host (sacrificial media). Read-back shows expected flux.
- Our per-side composite (structure_finder) visibly matches intended pattern/image for simple test cases (bars/checker or low-frequency image).
- Errors and recovery paths are clear and safe (WG off on abort).

## Risks and Mitigations
- USB 3.x bulk behavior: Mitigated by chunking, pre-buffer, and explicit pacing; still recommend older USB2 host for best reliability.
- Firmware load handshake: If the board requires `dtc`’s proprietary loader sequence, defer to “run `dtc` once” model.
- Safety: Hard default to WG off, small track ranges (e.g., track 0 only), loud warnings, and require `--force` to enable WG.

## Dependencies
- Linux host with `libusb-1.0` and `pyusb`
  - `apt install libusb-1.0-0-dev`
  - `pip install pyusb`
- Our existing generator:
  - `tools/patterns_to_stream_set.py --profile <preset>` produces `trackNN.S.raw` + `NN.S.raw`.

## Alignment with Memory Bank
- Forensic-rich captures, Linux-only hardware, Windows for generation/analysis.
- Avoid per-track PNG spam; use per-side composites (structure_finder).
- Profiles to simplify user CLI.
- CAPS/C2-aligned stream semantics already implemented and validated (dtc accepted).
