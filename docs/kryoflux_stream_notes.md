# KryoFlux STREAM Protocol Notes (rev 1.1, aligned)

These notes summarize the public STREAM protocol to align our generator with what DTC expects. We will validate against real DTC captures and adjust byte-for-byte.

References
- Local PDF: g:/floppy-as-audio_tape/kryoflux_stream_protocol_rev1.1.pdf
- Softpres Stream page: http://www.softpres.org/kryoflux:stream
- LoC format description: https://www.loc.gov/preservation/digital/formats/fdd/fdd000610.shtml

Core concepts
- STREAM carries two logical data types:
  - Flux transition timing (samples)
  - Disk index timing (per revolution)
- Out-of-band (OOB) blocks annotate the ISB sample stream with control and metadata.
- The file is byte-aligned. OOB payload words are little-endian. ISB flux values use a custom encoding.

C2/ISB sample encoding (ticks → bytes)
- Flux1: 0x0E..0xFF (1 byte) encodes values 14..255
- Flux2: 0x00..0x07 + 1 byte encodes value = (Header<<8) + Value1 (0..0x7FF)
- Flux3: 0x0C + 2 bytes encodes value = (Value1<<8) + Value2 (MSB then LSB) for 0x0800..0xFFFF
- Ovl16: 0x0B adds 0x10000 to the value of the next flux block; can repeat for very large values

OOB structure
- Each OOB block starts with a 4-byte header: 0x0D, Type, Size(LE16), followed by payload bytes.
- Types used in our generator:
  - Type 0x04: KFInfo (ASCII). The first OOB at byte 0 is: "KryoFlux stream - version {version}" with NO trailing NUL (size equals ASCII length).
  - Type 0x02: Disk Index (12 bytes payload: three LE32 words):
    - StreamPosition: ISB byte position where the next flux value will be written/read.
    - SampleCounter: SCK ticks since the last flux reversal up to the index event.
    - IndexCounter: Free-running ICK cycle count at index detection.
  - Type 0x03: StreamEnd (8 bytes payload: two LE32 words):
    - StreamPosition: Total ISB byte count at transfer end.
    - ResultCode: Hardware status (0 = success).
  - Type 0x01: StreamInfo (optional, 8 bytes payload):
    - StreamPosition (actual ISB byte position at OOB insertion) and TransferTime (ms).
  - Type 0x0D: EOF marker (Size typically 0x0D0D).

Multiple revolutions and index
- A complete track capture spans multiple revolutions (often 3–5+). Each revolution boundary should be annotated by an OOB Index (type 0x02).
- Index timing can be recovered from cumulative SampleCounter and IndexCounter; RPM may be derived between indices.

Header modes and defaults
- OOB-first (default): Begin with KFInfo (type 0x04) at byte 0 ("KryoFlux stream - version {version}"). Then StreamInfo and ISB samples.
- ASCII preamble (legacy): A null-terminated ASCII line at file start is supported via `--header-mode ascii`, but not used by default.

Generator alignment choices (current)
- Emit KFInfo (type 0x04) version string as first OOB, no trailing NUL.
- Emit StreamInfo (type 0x01): initial (SP=0, ms=0) and periodic, with payload SP equal to the actual ISB byte position at insertion.
- No initial Index at start-of-stream by default.
- Emit Index (type 0x02) after each revolution boundary with {StreamPosition, SampleCounter, IndexCounter}.
- End with StreamEnd (type 0x03) and EOF (type 0x0D) sentinel.

Open questions to validate against real DTC captures
- Does your DTC build require an initial index OOB at the beginning?
- Are there specific constraints on the KFInfo text (spacing, decimals)?
- Are there ordering nuances before the first ISB block we should mirror?

Planned next steps
- Capture reference tracks via DTC (`-i0`) and compare OOB and ISB histograms with `tools/kfx_probe.py`.
- Match our emitter to the accepted OOB/header shape and finalize defaults.
- Re-test minimal write (`-i0 -w`) once aligned.
