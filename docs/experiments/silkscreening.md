# Silkscreening (Image→Flux) — Experimental Tools

This guide documents the experimental image→flux "silkscreening" pipeline in FloppyAI: generate KryoFlux-compatible flux streams from images or built-in patterns, write them to physical media using DTC on a Linux host, read them back, analyze, and reconstruct the intended image for fidelity checks.

Important notes:
- Generation and analysis are designed for Windows (or any Python machine).
- All hardware operations (write/read) happen on the Linux DTC host (bash scripts / manual `dtc`).
- Streams are written as real KryoFlux C2/OOB data with DTC-friendly file naming `trackNN.s.raw`.

---

## Prerequisites

- Python 3.10+
- Install dependencies:
  ```bash
  pip install -r FloppyAI/requirements.txt
  ```
- Suggested: SSD for faster I/O.

---

## Quickstart: Built-in Pattern

Generate a small wedges test (3.5" HD, 300 RPM) for tracks 0–3, both sides present:

```powershell
python FloppyAI/src/main.py silkscreen_pattern wedges \
  --side 0 \
  --tracks 0-3 \
  --rpm 300 \
  --avg-interval-ns 2200 \
  --disk-name wedges_test
```

Outputs:
- Directory: `test_outputs/<timestamp>/silkscreen_pattern/wedges_test/`
- Files: `track00.0.raw`, `track00.1.raw`, `track01.0.raw`, ...
- Ground truth (for image comparison): `polar_target.png`, `polar_target.npy`

---

## Quickstart: External Image

Use any image; it will be resampled to the (tracks × angle) polar grid.

```powershell
python FloppyAI/src/main.py silkscreen .\my_image.png \
  --side 0 \
  --tracks 0-79 \
  --rpm 300 \
  --avg-interval-ns 2200 \
  --disk-name my_image_hd
```

Outputs go to: `test_outputs/<timestamp>/silkscreen/my_image_hd/` with the same `trackNN.s.raw` naming.

---

## Parameter Reference (generation)

- `--rpm`: Drive RPM used to time the revolution (e.g., 300 for 3.5" HD/DD; 360 for many 5.25" HD).
- `--avg-interval-ns`: Target mean interval (ns) per transition; controls transitions per revolution and file size.
  - At 300 RPM, one revolution ≈ 200ms = 200,000,000ns.
  - Target transitions ≈ Tr / avg_interval_ns.
  - Examples at 300 RPM:
    - 2000ns → ~100,000 transitions → ~400 KB per rev
    - 2200ns → ~91,000 transitions → ~360–380 KB per rev
- `--min-interval-ns`, `--max-interval-ns`: Safety clamps for interval generation.
- `--angular-bins`: Theta resolution for mapping brightness → transition density (default 720).
- `--disk-name`: Subfolder label under the selected output directory.
- Pattern-specific options (for `silkscreen_pattern`): `--k`, `--duty`, `--theta-period`, `--radial-period`.

Notes:
- Both sides are emitted for every track. The non-pattern side is a uniform ("blank") flux stream matching the timing constraints so downstream tooling always has files for both sides.

---

## Output Structure and Naming

- Folder: `<chosen-output-dir>/<disk_name>/`
- Files per track and side:
  - `trackNN.s.raw` (e.g., `track00.0.raw` = track 0 side 0)
- Ground truth per disk:
  - `polar_target.png` and `polar_target.npy`

These names are important. DTC and FloppyAI’s tools rely on the `NN.s` convention to infer track and side.

---

## Linux DTC: Write and Read (Forensic-rich)

Perform all hardware operations on the Linux host. Below are representative commands; adjust paths and drives.

Write generated streams to disk:

```bash
# Prefix-based expansion: DTC looks for track%02d.%d.raw
# -i21: raw stream input
# -f  : prefix to files (no extension, DTC adds %02d.%d.raw)
# -w  : write to hardware
# -t  : track range
# -s  : side

# Side 0
sudo dtc -i21 -f /path/to/wedges_test/track -w -t 0-79 -s 0

# Side 1 (if desired to rewrite blank or alternate content)
sudo dtc -i21 -f /path/to/wedges_test/track -w -t 0-79 -s 1
```

Forensic-rich read-back capture (example: 16 revolutions per track, side 0 and 1):

```bash
# -i0 : input from hardware
# -r  : revolutions
# -f  : prefix for output streams
# -t  : track range
# -s  : side

sudo dtc -i0 -f /path/to/captures/wedges_test/track -r 16 -t 0-79 -s 0
sudo dtc -i0 -f /path/to/captures/wedges_test/track -r 16 -t 0-79 -s 1
```

Recommendations:
- Use cooldowns between long runs.
- Keep detailed logs of dtc version and command lines.
- Prefer higher rev counts (e.g., `-r 16+`) for stable analysis.

---

## Analyze and Reconstruct (Windows)

Analyze the captured streams:

```powershell
python FloppyAI/src/main.py analyze_disk .\captures\wedges_test --rpm 300 --output-dir .\test_outputs\an_wedges
```

Reconstruct the image (“structure finder”) and compare against ground truth:

```powershell
# Typically you will find surface_map.json inside the per-disk analysis output
python FloppyAI/src/main.py recover_image .\test_outputs\an_wedges\disks\wedges_test\surface_map.json \
  --side 0 \
  --angular-bins 720 \
  --output-dir .\test_outputs\recovery_wedges
```

Outputs include the reconstructed polar PNG and correlation/phase alignment metrics.

---

## Troubleshooting

- "Could not determine track/side":
  - Ensure filenames follow `trackNN.s.raw` and use the `-f /path/to/dir/track` prefix with DTC.
- "Invalid stream":
  - Ensure files were generated with the current writer (KryoFlux C2/OOB). Regenerate if you were using earlier pseudo streams.
- File too small:
  - Decrease `--avg-interval-ns` (e.g., 2000ns) for more transitions per rev.
- Generation slow:
  - Ensure NumPy is installed (bulk I/O). Use local SSD.
- RPM mismatch or poor analysis alignment:
  - Verify `--rpm` used during generation and analysis matches the actual drive.

---

## Roadmap

- Additional dithering modes and quantization.
- Optional controlled noise/jitter for the blank side.
- Parallelized per-track generation for very large sets.
- Advanced recovery metrics and phase-aligned composites.
