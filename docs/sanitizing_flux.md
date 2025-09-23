# Sanitizing Flux for Hardware-Friendly Streaming

Modern USB controllers and device firmware can struggle with extreme flux patterns (very short cells, very long gaps). To improve write stability, FloppyAI can pre-sanitize generated flux before export.

## What the Sanitizer Does
- Clamp minimum cell length.
  - Cells shorter than a threshold are raised to the minimum (default 2000 ns ≈ 2 µs).
- Split excessively long gaps (keepalive).
  - Very long intervals are split into smaller chunks so no single chunk exceeds a keepalive threshold (default 8 ms), while still observing a hard maximum cap (default 65 ms).
- Preserve revolution timing.
  - After reshape, the last interval is adjusted so the total per‑revolution time remains unchanged.

This maps to common guidance:
- “Don’t go below ~2 µs.”
- “Insert dummy transitions for huge gaps so the device isn’t asked to hold silence too long.”

## How to Use

### One‑off flags
```powershell
# Windows (generation only)
python .\FloppyAI\tools\patterns_to_stream_set.py \
  --tracks 0-0 --sides 0 \
  --pattern alt --long-ns 4200 --short-ns 2200 \
  --sanitize --sanitize-min-ns 2000 --sanitize-keepalive-ns 8000000 --sanitize-max-ns 65000000 \
  --writer strict-dtc \
  --output-dir .\dtc_set_safe
```

### Profile: safe-usb (recommended)
Use the built‑in `safe-usb` profile to avoid long flag lists:
```powershell
python .\FloppyAI\tools\patterns_to_stream_set.py \
  --tracks 0-10 --sides 0 \
  --profile safe-usb \
  --output-dir .\dtc_set_safe
```
The `safe-usb` profile:
- Enables the sanitizer.
- Uses conservative cell sizes (≈4.2 µs / 2.2 µs).
- Targets `--writer strict-dtc` and generates both `trackNN.S.raw` and `NN.S.raw`.

## Linux Write (DTC)
From the Linux DTC host (recommended: older USB2‑native PC/hub):
```bash
./FloppyAI/scripts/linux/dtc_write_read_set.sh \
  --image-dir ~/FloppyAI/dtc_set_safe \
  --drive 0 --tracks 0-0 --sides 0 --no-sudo
```
For initial validation, prefer single‑file mode to isolate a track:
```bash
cd ~/FloppyAI/dtc_set_safe
sudo dtc -f track00.0 -i0 -d0 -g0 -w -v
```

## Notes
- Sanitization runs per revolution before export; OOB Index placement (rev boundaries) is unchanged.
- Defaults are conservative and can be tuned per media/drive.
- Sanitizer is applied in the generator. An optional writer‑side safety net can be added later behind a `--sanitize-writer` flag if needed.

## Troubleshooting
- If you still see “streaming device transfer error,” try:
  - Increasing `--sanitize-min-ns` slightly (e.g., 2400 ns).
  - Decreasing `--sanitize-keepalive-ns` (e.g., 4–6 ms) to cause more frequent dummy splits.
  - Reducing stream chunk sizes on the hardware writer (when available) or using an older USB2 host.
- Verify disk write‑protect status and drive power (especially +12V for many 5.25" drives).
