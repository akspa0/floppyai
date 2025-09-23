# Forensic‑Rich Flux Capture Guide (KryoFlux + FloppyAI)

This guide documents a plug‑and‑play workflow for capturing high‑fidelity flux data from floppy disks using KryoFlux on Linux, then analyzing results with FloppyAI on Windows. It prioritizes forensic‑rich data collection for experimentation and media/head tolerance studies (sectors/decoding are optional).

- Hardware ops (capture) happen on Linux via KryoFlux DTC.
- Analysis/visualization runs on Windows (or anywhere with Python + Matplotlib).
- We capture STREAM flux (`-i0`), high revolutions, minimal/zero retries, with cooldowns.

## Why STREAM (`-i0`) and high revolutions

- **STREAM (`-i0`)** preserves raw flux timings. This is essential for:
  - Flux intervals histogram (log‑scale)
  - Angular distributions (track × angle heatmaps)
  - Instability/variance metrics
- **Revolutions (`-r N`)** capture repeated passes of the same magnetic data, revealing variability from media, head alignment, and electronics.
- **Retries** repeatedly re‑seek problematic areas and can overstress fragile media. Prefer **more revolutions** instead of retries during capture.

## Provided Linux capture scripts

Location: `scripts/linux/`

All scripts:
- Save outputs in a timestamped subfolder under `--out-dir` (default: `./captures`).
- Name files parseably, e.g., `label_tNN_sS_rRR_YYYYmmdd_HHMMSS.raw`.
- Write a `run.log` with DTC version, path, parameters, and every command executed.
- Use `sudo` by default (pass `--no-sudo` to skip).
- Use `-i 0` (STREAM), `-p` (create directories), `-r <revs>`.
- Avoid retry flags by design (rely on revolutions).

### 1) Full‑disk capture

`capture_forensic_full_disk.sh`

Forensic‑rich full‑disk STREAM capture across a track range and sides with cooldowns and spin‑up.

Usage:
```bash
./capture_forensic_full_disk.sh [options]

# Common options
--drive <N>            # DTC drive index (default: 0)
--dtc-path <path>      # Path to dtc (default: dtc)
--out-dir <dir>        # Output directory (default: ./captures)
--label <name>         # Label prefix for files (default: full-disk)
--no-sudo              # Do not prefix dtc with sudo (default: sudo on)
--dry-run              # Print commands only

# Capture profile
--profile <name>       # 35HD, 35DD, 525HD (0..79), 525DD (0..39)
--sides <both|0|1>     # Which sides (default: both)
--start-track <N>      # Default from profile or 0
--end-track <N>        # Default from profile or 79/39
--step <N>             # Track step (default: 1)
--revs <N>             # Revolutions per track (default: 16)
--cooldown <sec>       # Pause between tracks (default: 2)
--spinup <sec>         # Spin-up delay before each side (default: 2)
```

Examples:
```bash
# 3.5" HD full disk, 16 revs, gentle cadence
./capture_forensic_full_disk.sh \
  --profile 35HD --sides both --revs 16 \
  --start-track 0 --end-track 79 --step 1 \
  --cooldown 3 --spinup 2 \
  --out-dir ./captures --label win95_set --drive 0

# 5.25" DD (40 tracks), 20 revs (heavier), larger cooldowns
./capture_forensic_full_disk.sh \
  --profile 525DD --sides both --revs 20 \
  --cooldown 5 --spinup 3 \
  --out-dir ./captures --label dd40_forensic --drive 0
```

### 2) Repeat a single track (time series)

`capture_forensic_repeat_track.sh`

Repeatedly capture the same track/side to study drift and tolerance over time.

Usage:
```bash
./capture_forensic_repeat_track.sh [options]

# Required
--track <N>            # Track number
--side <0|1>          # Side

# Common
--drive <N> --dtc-path <path> --out-dir <dir> --label <name> --no-sudo --dry-run

# Capture profile
--repeats <N>         # Passes (default: 10)
--revs <N>            # Revolutions per pass (default: 16)
--cooldown <sec>      # Pause between passes (default: 5)
--spinup <sec>        # Spin-up before first pass (default: 2)
```

Example:
```bash
./capture_forensic_repeat_track.sh \
  --track 40 --side 0 --repeats 12 --revs 16 \
  --cooldown 5 --spinup 2 \
  --out-dir ./captures --label repeat40 --drive 0
```

### 3) Repeated sweeps across a range

`capture_forensic_sweep.sh`

Perform multiple forward/backward sweeps across a track range for temporal studies.

Usage:
```bash
./capture_forensic_sweep.sh [options]

# Common
--drive <N> --dtc-path <path> --out-dir <dir> --label <name> --no-sudo --dry-run

# Capture profile
--profile <name>      # 35HD/35DD/525HD (0..79) or 525DD (0..39)
--sides <both|0|1>    # default: both
--start-track <N>     # default from profile or 0
--end-track <N>       # default from profile or 79/39
--step <N>            # default: 1
--sweeps <N>          # number of sweeps (default: 3)
--revs <N>            # revolutions per capture (default: 16)
--cooldown <sec>      # pause between captures (default: 3)
--spinup <sec>        # spin-up delay per side (default: 2)
```

Example:
```bash
./capture_forensic_sweep.sh \
  --profile 35HD --sides both \
  --start-track 0 --end-track 79 --step 1 \
  --sweeps 3 --revs 16 --cooldown 3 --spinup 2 \
  --out-dir ./captures --label sweepA --drive 0
```

### 4) Long‑revolution capture on a single track

`capture_forensic_long_rev.sh`

Capture a single track/side with a large number of revolutions (e.g., 32–48) to maximize statistical power.

Usage:
```bash
./capture_forensic_long_rev.sh [options]

# Required
--track <N>            # Track number
--side <0|1>          # Side

# Common
--drive <N> --dtc-path <path> --out-dir <dir> --label <name> --no-sudo --dry-run

# Capture profile
--revs <N>            # Revolutions (default: 32)
--cooldown <sec>      # Pause after capture (default: 5)
--spinup <sec>        # Spin-up before capture (default: 2)
```

Example:
```bash
./capture_forensic_long_rev.sh \
  --track 0 --side 0 --revs 48 --cooldown 5 --spinup 2 \
  --out-dir ./captures --label longrev0 --drive 0
```

## After capture: analyze with FloppyAI (Windows)

Copy the capture directory to Windows and run:

PowerShell:
```powershell
python src/main.py analyze_disk .\captures\win95_set_2025... ^
  --profile 35HD --angular-bins 720 ^
  --output-dir .\test_outputs\win95_mfm_full
```

Linux/macOS:
```bash
python src/main.py analyze_disk ./captures/win95_set_2025... \
  --profile 35HD --angular-bins 720 \
  --output-dir ./test_outputs/win95_mfm_full
```

Outputs in the analysis folder:
- `<prefix>_disk_surface.png` (Side 0 | Side 1 density per track)
- `<prefix>_instability_map.png` (both sides)
- `<prefix>_side0_report.png`, `<prefix>_side1_report.png`
  - Top: polar surface (density) with optional sector overlays
  - Middle‑left: instability polar
  - Middle‑right: angular heatmap (track × angle)
  - Bottom‑right: aggregated interval histogram (log ns)
- `surface_map.json` and `run.log` for provenance and stats

Optional format overlays:
```powershell
# If you want sector boundary lines only (e.g., for clean visuals)
python src/main.py analyze_disk <dir> ^
  --profile 35HD --angular-bins 720 --format-overlay --overlay-mode mfm ^
  --overlay-sectors-hint 18 ^
  --output-dir .\test_outputs\...
```

## Naming & logging

- Filenames: `label_tNN_sS_rRR_YYYYmmdd_HHMMSS.raw`
- Per-run `run.log` includes DTC path/version, parameters, and all commands.
- In analysis, `surface_map.json` includes `global.inputs` and overlay info used.

## Safety notes for fragile media

- Prefer higher `--revs` over retries; retries re‑seek the head on weak areas.
- Use cooldowns (`--cooldown`) and spin‑ups (`--spinup`) to reduce heat.
- Start experiments on sacrificial media first; outer tracks often tolerate heat better.
- Avoid filtering/post‑processing at capture time; keep flux raw (`-i0`).

## Quick DTC recipes (manual)

- Create STREAM only (folder will be created via `-p`):
```bash
dtc -d 0 -i 0 -p -t <track> -s <side> -r 16 -f '/path/to/out.raw' read
```
- STREAM + MFM image in one pass (optional):
```bash
dtc -d 0 -p -f '/path/to/stream' -i 0 -f '/path/to/image.img' -i 4 -t <track> -s <side> -r 16 read
```
- Deviceless decoding later (from STREAM):
```bash
dtc -f '/path/to/stream' -i 0 -f '/path/to/image.img' -i 4 -m 1
```

## Troubleshooting

- Missing `run.log`/`surface_map.json` in analysis output:
  - Re‑run analyze pointing at the directory with many `.raw` files (not a single file).
  - Check console for `Found N .raw files`, and `global.inputs` in `surface_map.json`.
- Uniform visuals:
  - Increase `--revs` during capture (e.g., 16→32) to enrich angular and interval statistics.
  - Ensure multiple tracks were captured for full disk surfaces and per‑side reports.

---

If you want additional scripts (e.g., compare two captures across time to produce delta reports per side), open an issue or ask and we’ll add them.
