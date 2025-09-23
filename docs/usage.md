# FloppyAI CLI Usage Guide

This guide shows how to run FloppyAI directly from the FloppyAI directory using the main script, and includes practical examples for all commands.

## Invocation

- Always run from the `FloppyAI/` directory (the folder containing `src/`, `docs/`, etc.).
- Preferred: direct-run the main script

```bash
python src/main.py --help
```

- Alternate: module syntax also works

```bash
python -m src.main --help
```

## Environment

- Python 3.9+ recommended
- `pip install -r requirements.txt`
- Optional: LM Studio (or compatible OpenAI API server) for `--summarize`
- Optional: KryoFlux hardware + DTC for read/write flows

## Global Tips

- Use `--output-dir` to direct outputs to a stable folder; otherwise a timestamped directory under `test_outputs/` is created.
- Prefer `--profile` to set RPM and overlay defaults. Supported: `35HD`, `35DD`, `525HD`, `525DD`, plus GCR variants `35HDGCR`, `35DDGCR`, `525DDGCR`.
- Overlay mode now defaults to `auto` and is profile‑driven (GCR profiles pick GCR; others pick MFM). You can still force with `--overlay-mode`.
- Instability visuals are now bright by default (inverted magma colormap with auto‑contrast).

---

## Commands

### 1) analyze — single .raw file

Parse a single KryoFlux `.raw` and generate basic statistics and plots.

```bash
python src/main.py analyze path\to\file.raw --output-dir .\test_outputs\single
```

Outputs: `<stem>_intervals.png`, `<stem>_hist.png`, and `<stem>_heatmap.png` (if multi‑rev).

### 2) read — capture a track/side

```bash
python src/main.py read <track> <side> [--revs 3] [--simulate] [--analyze] [--output-dir DIR]
```

- `--simulate` avoids real hardware access
- `--analyze` automatically runs `analyze` on the captured `.raw`

### 3) write — write a `.raw` to hardware

```bash
python src/main.py write <input.raw> <track> <side> [--simulate] [--output-dir DIR]
```

### 4) generate — synthetic stream for testing

```bash
python src/main.py generate <track> <side> [--revs 1] [--cell 4000] [--analyze] [--output-dir DIR]
```

- Increase `--revs` to fill more revolutions
- Reduce `--cell` (ns) to raise density

### 5) encode — binary → .raw

```bash
python src/main.py encode <input.bin> <track> <side> \
  [--density 1.0] [--variable] [--revs 1] [--output OUT.raw] [--write] [--simulate] [--analyze] [--output-dir DIR]
```

### 6) decode — recover binary data from `.raw`

```bash
python src/main.py decode <input.raw> \
  [--density 1.0] [--variable] [--revs 1] [--output out.bin] [--expected orig.bin] [--output-dir DIR]
```

---

## Disk‑level analysis

### 7) analyze_disk — build a full surface map

```bash
python src/main.py analyze_disk <dir_or_file> \
  [--rpm FLOAT] [--profile 35HD|35DD|35HDGCR|35DDGCR|525HD|525DD|525DDGCR] \
  [--format-overlay] [--overlay-mode mfm|gcr|auto] [--angular-bins N] \
  [--overlay-sectors-hint INT] [--overlay-alpha F] [--overlay-color HEX] \
  [--summarize] [--lm-host HOST[:PORT]] [--lm-model NAME] [--lm-temperature F] \
  [--output-dir DIR]
```

- RPM: provide `--rpm`, or let `--profile` set one and bias overlay mode automatically
- MFM overlays (PC): `--format-overlay` (auto from profile); optional `--align-to-sectors auto` and `--label-sectors`
- GCR overlays (Mac): `--format-overlay --profile 35DDGCR` (auto candidates `12,11,10,9,8`)

Outputs include `surface_map.json`, `overlay_debug.json`, composite and surface PNGs, and an instability summary CSV.

#### Reading the Instability Panels

- Bright = angles where the flux transition pattern varies more across revolutions (less repeatable).
- Dark = angles with consistent behavior (more repeatable).
- This is a statistical repeatability map; it is not a decoded data or audio‑waveform view. Format structures (sectors/gaps) and mechanical effects can create wedge‑like patterns without implying synthetic content.

Verification steps:
- Open `surface_map.json`:
  - Per track/side: `analysis.instability_theta` (0..1 angular profile), `analysis.instability_features`, and `analysis.instability_score` (scalar).
  - Check that bright regions in the panel correspond to higher values in `instability_theta` for the same track/angles.
- See the concept page for details: [docs/instability.md](./instability.md)

### 8) analyze_corpus — aggregate many maps

```bash
python src/main.py analyze_corpus <root_or_map.json> \
  [--generate-missing] [--rpm FLOAT] [--profile 35HD|35DD|35HDGCR|35DDGCR|525HD|525DD|525DDGCR] \
  [--format-overlay] [--overlay-mode mfm|gcr|auto] [--angular-bins N] [--overlay-sectors-hint INT] \
  [--overlay-alpha F] [--overlay-color HEX] \
  [--summarize] [--lm-host HOST[:PORT]] [--lm-model NAME] [--lm-temperature F] \
  [--output-dir DIR]
```

- With `--generate-missing`, the tool will first run `analyze_disk` on directories that contain `.raw` files but no `surface_map.json` and then aggregate
- Writes `corpus_inputs.txt` and `corpus_missing_inputs.txt` for transparency

### 9) compare_reads — compare multiple reads of the same disk

```bash
python src/main.py compare_reads <path_or_dir1> <path_or_dir2> [<...>] [--output-dir DIR]
```

- Each argument may be a `surface_map.json` path or a directory containing it
- Outputs `diff/diff_summary.json` and `diff/diff_densities.csv`

### 10) classify_surface — quick blank-like vs written-like

```bash
python src/main.py classify_surface <surface_map.json> [--blank-density-thresh 1000] [--output-dir DIR]
```

### 11) plan_pool — select top‑quality tracks as a bit pool

```bash
python src/main.py plan_pool <surface_map.json> [--min-density 2000] [--top-percent 0.2] [--output-dir DIR]
```

---

## Cross‑machine (Linux DTC) workflow

When KryoFlux DTC hardware is attached to a Linux host that requires sudo, do not attempt to orchestrate DTC from Windows. Use a manual, script-based flow on Linux and analyze on Windows.

Recommended steps:
- Generate streams on Windows with FloppyAI (you can test with `--simulate` first):
  ```bash
  python src/main.py generate 80 0 --revs 1 --density 1.2 --pattern prbs7 --output-dir .\test_outputs\to_linux
  ```
- Transfer the generated `.raw` file(s) to the Linux DTC host (USB/share/etc.).
- On Linux, run your bash script(s) to write and then read back captures with dtc (sudo as needed). For example:
  ```bash
  # Example only; adapt to your environment
  sudo dtc -d 0 -i 0 -t 80 -s 0 -f /path/to/generated.raw write
  sudo dtc -d 0 -i 0 -t 80 -s 0 -r 3 -f /path/to/captured_80_0.raw read
  ```
- Transfer captured `.raw` back to Windows and analyze:
  ```bash
  python src/main.py analyze path\to\captured_80_0.raw --output-dir .\test_outputs\captures
  ```

Notes:
- `read`/`write` CLI commands in this repo assume local DTC availability; on Windows, prefer `--simulate` or use them only if you have local DTC.
- Cross‑machine orchestration (e.g., SSH) is intentionally out of scope.

## Overlay Notes

- The overlay block is saved in `surface_map.json` under `global.insights.overlay` and per‑track under `<track>.overlay`.
- If confidence is weak, the tool falls back to either profile defaults, your `--overlay-sectors-hint`, or the median per‑track sector count.
- Increase `--angular-bins` for finer phase resolution; tune `--overlay-color` and `--overlay-alpha` for visibility.

## Troubleshooting

- Compare reads says it found no `surface_map.json`:
  - Run `analyze_disk` for each read first (pointing to the read directory), then re-run `compare_reads` using those run output folders or the direct JSON paths.

- Module cannot be found:
  - Ensure you run commands from the `FloppyAI/` directory with `python -m src.main ...`.
