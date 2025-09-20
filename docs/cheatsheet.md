# FloppyAI Cheatsheet

Quick-reference for the most common workflows and flags. Run everything from the repository root using the main script. Module syntax is also supported as an alternate.

- Show help
  ```bash
  python FloppyAI/src/main.py --help
  ```

## One‑liners

- Analyze a single .raw file
  ```bash
  python FloppyAI/src/main.py analyze path\to\file.raw --output-dir .\test_outputs\single
  ```

- Analyze a full disk directory (PC 3.5" HD, MFM overlays)
  ```bash
  python FloppyAI/src/main.py analyze_disk path\to\disk_dir \
    --media-type 35HD --format-overlay --overlay-mode mfm --angular-bins 720 \
    --output-dir .\test_outputs\disk_run
  ```

- Analyze a Mac GCR disk (400K/800K, zoned)
  ```bash
  python FloppyAI/src/main.py analyze_disk path\to\mac_disk \
    --media-type 35DD --format-overlay --overlay-mode gcr \
    --gcr-candidates "12,10,8,9,11,13" --angular-bins 900 \
    --output-dir .\test_outputs\mac_run
  ```

- Compare two reads (after each has a surface_map.json)
  ```bash
  python FloppyAI/src/main.py compare_reads .\test_outputs\win95_0 .\test_outputs\win95_1 \
    --output-dir .\test_outputs\diff_win95
  ```

- Build a corpus, auto‑generating per‑disk maps first
  ```bash
  python FloppyAI/src/main.py analyze_corpus .\stream_dumps \
    --generate-missing --media-type 35HD \
    --format-overlay --overlay-mode mfm --angular-bins 720 \
    --output-dir .\test_outputs\corpus
  ```

## Overlay quick recipes

- MFM (IBM‑like, PC):
  ```bash
  --format-overlay --overlay-mode mfm --angular-bins 720
  ```
  - Typical k values: 8, 9, 15, 18

- GCR (Apple Mac 400K/800K):
  ```bash
  --format-overlay --overlay-mode gcr --gcr-candidates "12,10,8,9,11,13" --angular-bins 900
  ```
  - Try a wider candidate list if unsure: `"6,7,8,9,10,11,12,13,14"`

- Visibility tweaks:
  ```bash
  --overlay-color "#ff3333" --overlay-alpha 0.8
  # or bright yellow
  --overlay-color "#ffd60a" --overlay-alpha 0.7
  ```

- Fallback hint when detection is weak:
  ```bash
  --overlay-sectors-hint 18   # force k=18 spokes if needed
  ```

## Media & RPM

- Force media classification (overrides heuristic):
  ```bash
  --media-type 35HD|35DD|525HD|525DD
  ```
- RPM defaults by profile (used if `--rpm` not provided):
  - 35HD / 35DD → 300 RPM
  - 525HD → 360 RPM
  - 525DD → 300 RPM

## Common workflows

- PC disk (1.44MB) two‑read comparison
  ```bash
  # Read A
  python FloppyAI/src/main.py analyze_disk FloppyAI\stream_dumps\1.44\win95boot\0 \
    --media-type 35HD --format-overlay --overlay-mode mfm --angular-bins 720 \
    --output-dir .\test_outputs\win95_0

  # Read B
  python FloppyAI/src/main.py analyze_disk FloppyAI\stream_dumps\1.44\win95boot\1 \
    --media-type 35HD --format-overlay --overlay-mode mfm --angular-bins 720 \
    --output-dir .\test_outputs\win95_1

  # Compare
  python FloppyAI/src/main.py compare_reads .\test_outputs\win95_0 .\test_outputs\win95_1 \
    --output-dir .\test_outputs\diff_win95
  ```

- Mac corpus (generate + aggregate)
  ```bash
  python FloppyAI/src/main.py analyze_corpus .\stream_dumps\mac \
    --generate-missing --media-type 35DD \
    --format-overlay --overlay-mode gcr --gcr-candidates "12,10,8,9,11,13" \
    --angular-bins 900 --output-dir .\test_outputs\corpus_mac
  ```

## Outputs at a glance

- Per‑run (in your `--output-dir` or `test_outputs/<timestamp>/`):
  - `surface_map.json` — per‑track/side stats and analysis
  - `overlay_debug.json` — the overlay block only
  - `<label>_composite_report.png` — single composite image
  - `<label>_surface_disk_surface.png` — polar map (both sides)
  - `<label>_surface_side0.png`, `<label>_surface_side1.png`
  - `<label>_instability_summary.csv`

- Corpus (under `run_dir/corpus/`):
  - `corpus_summary.json`, histograms/boxplots/scatter
  - `corpus_inputs.txt`, `corpus_missing_inputs.txt`
  - `corpus_surfaces_grid.png`, side grids

- Diff (under `run_dir/diff/`):
  - `diff_summary.json` — global stats and overlay comparison
  - `diff_densities.csv` — per‑track densities across reads

## Troubleshooting

- "No surface_map.json found" when comparing reads:
  - Run `analyze_disk` on each read directory first, then rerun `compare_reads`.

- Overlays seem off:
  - Increase `--angular-bins` (e.g., 900+), adjust `--overlay-color`/`--overlay-alpha`.
  - For Mac disks, ensure `--overlay-mode gcr` and pass good `--gcr-candidates`.

- Package not found when using `-m`:
  - Run from the repository root (the folder that contains `FloppyAI/`).
