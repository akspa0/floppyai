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
  - Type 0x04: KFInfo (ASCII), e.g., "sck=24027428.5714285, ick=3003428.5714285625\0"
  - Type 0x02: Disk Index (12 bytes payload: three LE32 words):
    - StreamPosition: ISB byte position where the next flux value will be written/read.
    - SampleCounter: Total SCK ticks elapsed at index detection.
    - IndexCounter: Total ICK cycles elapsed at index detection.
  - Type 0x03: StreamEnd (8 bytes payload: two LE32 words):
    - StreamPosition: Total ISB byte count at transfer end.
    - ResultCode: Hardware status (0 = success).
  - Type 0x01: StreamInfo (optional, 8 bytes payload):
    - StreamPosition and TransferTime (ms).
  - Type 0x0D: EOF marker (Size typically 0x0D0D).

Multiple revolutions and index
- A complete track capture spans multiple revolutions (often 3–5+). Each revolution boundary should be annotated by an OOB Index (type 0x02).
- Index timing can be recovered from cumulative SampleCounter and IndexCounter; RPM may be derived between indices.

Header modes we will test
- OOB-first: Begin with KFInfo (type 0x04), optional initial Index (type 0x02), then ISB samples.
- ASCII preamble: A null-terminated ASCII info line first (legacy), then OOBs and samples. Our defaults use OOB-first.

Generator alignment choices (current)
- Emit KFInfo (type 0x04) with `sck` and `ick` values.
- Optional initial Index (type 0x02) at start with zeroed counters.
- After each revolution’s samples, emit Index (type 0x02) carrying {StreamPosition, SampleCounter, IndexCounter}.
- End with StreamEnd (type 0x03) and EOF (type 0x0D).

Open questions to validate against real DTC captures
- Does your DTC build require an initial index OOB at the beginning?
- Are there specific constraints on the KFInfo text (spacing, decimals)?
- Are there ordering nuances before the first ISB block we should mirror?

Planned next steps
- Capture reference tracks via DTC (`-i0`) and compare OOB and ISB histograms with `tools/kfx_probe.py`.
- Match our emitter to the accepted OOB/header shape and finalize defaults.
- Re-test minimal write (`-i0 -w`) once aligned.
