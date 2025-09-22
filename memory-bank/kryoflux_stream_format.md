# KryoFlux Stream Format — Internal Reference

This document records the specific STREAM format structure our generator now emits, aligned with the public protocol and empirical DTC behavior (to be validated with reference captures).

## ISB (In-Stream Buffer) Flux Encoding
- Flux1: Single byte 0x0E..0xFF encodes values 14..255 (ticks).
- Flux2: Two bytes, 0x00..0x07 followed by one byte; value = (Header<<8) + Value1 (0..0x7FF).
- Flux3: Three bytes, 0x0C followed by two bytes; value = (MSB<<8) + LSB (0x0800..0xFFFF). Note MSB-first order inside Flux3 payload.
- Ovl16: Single byte 0x0B. Each occurrence adds 0x10000 to the value of the next Flux block. Repeat as needed for very large values.
- NOPs: 0x08 (1B), 0x09 (2B), 0x0A (3B). We do not emit NOPs.

We track `ISB bytes` as the count of all bytes emitted by Flux1/Flux2/Flux3/OVL16/NOP blocks (OOB bytes are excluded).

## OOB (Out-Of-Stream Buffer) Blocks
All OOB blocks start with a 4‑byte header: 0x0D, Type, Size (LE16), followed by a payload of `Size` bytes.

Types we emit:
- 0x04 — KFInfo (ASCII):
  - Payload: a null-terminated string like `"sck=24027428.5714285, ick=3003428.5714285625\0"`.
  - Purpose: communicates sample clock (SCK) and index clock (ICK) used to interpret counters.
- 0x02 — Index (12 bytes):
  - Payload: three LE32 fields: `StreamPosition`, `SampleCounter`, `IndexCounter`.
  - `StreamPosition`: ISB byte position at the index moment (i.e., where the next flux byte would be written/read).
  - `SampleCounter`: number of SCK ticks elapsed at the index moment.
  - `IndexCounter`: number of ICK cycles elapsed at the index moment.
  - Placement: emitted at each revolution boundary; optional initial one at start (all zeroes).
- 0x03 — StreamEnd (8 bytes):
  - Payload: two LE32 fields: `StreamPosition`, `ResultCode` (0=success).
  - `StreamPosition`: total ISB byte count at transfer end.
- 0x01 — StreamInfo (8 bytes, optional):
  - Payload: two LE32 fields: `StreamPosition`, `TransferTimeMs`.
  - Not emitted by default.
- 0x0D — EOF:
  - Size is set to 0x0D0D; there is no payload beyond the header.

## File Layout (Default)
1) KFInfo (0x04)
2) Optional Index (0x02) at start with zero counters
3) ISB flux stream for N revolutions
   - After each revolution: Index (0x02)
4) StreamEnd (0x03)
5) EOF (0x0D)

## Counters and Clocks
- `sck_hz`: Sample clock rate; used to convert `ns` intervals to tick counts and for `SampleCounter` accumulation.
- `ick_hz`: Index clock rate; used to compute `IndexCounter = round((SampleCounter / sck_hz) * ick_hz)`.
- We default to `sck_hz ≈ 24 MHz` and `ick_hz ≈ 3.003 MHz`. Both are configurable via CLI.

## Validation Plan
- Use `tools/kfx_probe.py` to inspect OOB block sequence, sizes, parsed fields, and ISB opcode histogram.
- Capture DTC reference STREAMs (read `-i0`) and compare.
- Iterate header/order subtleties to match DTC expectations before re-testing write-from-stream (`-i0 -w`).

## Notes
- We do not emit the non-spec OOB type 0x08 (sample clock). KFInfo (0x04) conveys clocks as ASCII.
- ASCII preamble (legacy) is supported but off by default; we prefer OOB-first headers.
