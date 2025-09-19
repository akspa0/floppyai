# FloppyAI: Advanced Flux Analysis and Encoding Tool

## Overview
FloppyAI is a Python-based tool for interfacing with KryoFlux DTC CLI to read/write flux streams (.raw files), analyze magnetic media characteristics (noise, anomalies on blanks), and prototype custom flux encoding for higher data density. It builds on the existing src/ for flux handling and extends it for AI-enhanced error correction.

## Installation
1. Ensure Python 3.8+ and pip.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
   (NumPy, SciPy, Matplotlib for analysis/visualization.)

3. For hardware: Connect KryoFlux to PC; ensure drivers installed. DTC.exe is in ../lib/kryoflux_3.50_windows_r2/dtc/.

## Usage
Run from FloppyAI/ directory or src/ subdirectory:
```
cd FloppyAI/src  # Or run from root with python -m FloppyAI.src.main
python main.py --help
```

Note: Commands like analyze_disk do not take positional file arguments; use specific subcommands for single files.

### Commands

1. **Analyze a Stream File** (Decode and visualize .raw):
   ```
   python main.py analyze <input.raw>
   ```
   - Parses flux transitions, computes stats (mean interval, noise variance).
   - Detects anomalies (short/long cells, weak bits via rev inconsistencies).
   - Generates plots: <input>_intervals.png (time series), <input>_hist.png (distribution), <input>_heatmap.png (if multi-rev).
   - Example: `python main.py analyze ../../example_stream_data/unknown-stream00.0.raw`
     - Output: Stats like "Mean Interval: 4000.50 ns", "Short Cells: 5", visualizations saved.
   - For blanks: High variance indicates surface irregularities; low anomalies = clean media.

2. **Read Track from Hardware**:
   ```
   python main.py read <track> <side> <output.raw> [--revs 3] [--simulate] [--analyze]
   ```
   - Reads track/side (e.g., 0 0) to .raw using DTC.
   - --revs: Revolutions (default 3 for better analysis).
   - --simulate: Dry-run without hardware (no-op).
   - --analyze: Auto-analyze output .raw.
   - Use for blanks: Reveals baseline flux noise for custom encoding.

3. **Write Stream to Hardware**:
   ```
   python main.py write <input.raw> <track> <side> [--simulate]
   ```
   - Writes .raw to track/side.
   - --simulate: Dry-run (logs command).
   - Test custom flux on blanks to explore readability.

4. **Generate Dummy Stream** (Build custom flux for testing):
   ```
   python main.py generate <track> <side> <output.raw> [--revs 1] [--cell 4000] [--analyze]
   ```
   - Creates .raw with uniform intervals + noise (simulate one revolution).
   - --revs: Number of revolutions.
   - --cell: Nominal cell length ns (vary for density: shorter = higher density).
   - --analyze: Auto-analyze output.
   - Example: `python main.py generate 0 0 dummy.raw --cell 2000 --analyze`
     - Generates denser flux; analyze to see interval distribution.
 
 5. **Encode Binary Data to Custom Stream** (Prototype higher density encoding):
    ```
    python main.py encode <input.bin> <track> <side> [--density 1.0] [--variable] [--revs 1] [--output <output.raw>] [--write] [--simulate] [--analyze]
    ```
    - Encodes binary file to .raw using Manchester or variable RLL-like flux encoding.
    - --density: Scaling factor (>1.0 shortens cells for higher density; e.g., 2.0 for ~2x bits).
    - --variable: Use RLL-like variable cell lengths (short for 0s, long for 1s) for advanced packing.
    - --revs: Revolutions to fill (repeats data to embed continuously).
    - --output: Custom .raw path (default: encoded_track_X_Y.raw in timestamp dir).
    - --write: Auto-write .raw to hardware track/side after generation.
    - --simulate: Dry-run for --write (no hardware).
    - --analyze: Auto-analyze generated .raw (check density estimate vs. achieved).
    - Outputs achieved density (bits/rev) based on input size.
    - Example: `python main.py encode test_data.bin 0 0 --density 2.0 --variable --analyze`
      - Encodes 1KB data at 2x density with variable cells; prints ~8192 bits/rev (vs. standard ~4000); analyzes for noise/readability.
    - For density testing: Compare bits/rev in output to standard (analyze dummy at density=1.0); higher = success if low anomalies.
 
 6. **Decode Custom Stream** (Recover binary data from encoded .raw):
    ```
    python main.py decode <input.raw> [--density 1.0] [--variable] [--revs 1] [--output <output.bin>] [--expected <original.bin>] [--output-dir]
    ```
    - Decodes flux to binary using matching parameters.
    - --density: Expected density used in encoding.
    - --variable: Assume RLL-like variable cells.
    - --revs: Number of revolutions.
    - --output: Custom .bin path.
    - --expected: Original .bin for verification (reports % match, byte errors).
    - Example: `python main.py decode test_encoded.raw --density 2.0 --variable --expected test_data.bin`
      - Outputs test_decoded.bin; verifies 100% recovery for all-zero data (perfect for blanks).
 
 7. **Analyze Disk Surface** (Batch process streams for full disk map):
    ```
    python main.py analyze_disk [input] [--track N] [--side 0|1] [--output-dir]
    ```
   - input: Optional directory or single .raw file (default: ../example_stream_data/). Globs all *.raw if dir; auto-batches parent dir if single numbered file and siblings exist.
   - Parses track/side from filename ending in \d+\.\d+\.raw (e.g., BugsBunnyHare00.0.raw → track 0 side 0).
   - Filters to tracks 0-83, sides 0-1; processes in order (00.0, 00.1, ..., 83.1), logging found vs expected (up to 168 files).
   - Use --track/--side for manual override if unparsable (applies to all files if no pattern).
   - Outputs surface_map.json: Per-track/side list of individual files with noise/anomalies/density; includes paths and aggregate stats if multiple files.
   - Generates combined PNG visualizations per track/side (intervals, histogram, heatmap if multi-rev) using all flux data for that track/side.
   - Examples:
     - Default batch: `python main.py analyze_disk` (processes example_stream_data/)
     - Single file (auto-batch if siblings): `python main.py analyze_disk ..\stream_dumps\BugsBunny\bugsbunny\BugsBunnyHare00.0.raw`
     - Full disk dir: `python main.py analyze_disk ..\stream_dumps\BugsBunny\bugsbunny` (sorts/analyzes 00.0.raw to 83.1.raw)
     - With override: `python main.py analyze_disk /path/to/unpatterned --track 5 --side 0`
   - Processes blanks/dumps; JSON/viz show high variance areas (e.g., static noise >track 80) for targeted encoding (weak bits as ternary states).
   - Use for full disk: Maps entire surface, identifies coercivity variations for adaptive density.
   - Note: For single-file focus with per-file viz, use 'analyze'. Binary files like test_data.bin are for encode/decode, not flux analysis.
 
 ## Testing the Tooling
 - **Round-trip:** Encode test_data.bin → test_encoded.raw, decode back; expect 100% match.
 - **Density:** At 2.0, ~8192 bits/rev vs. standard ~4000; decoder recovers data despite 5% noise.
 - **Disk Surface:** Run analyze_disk on blanks to explore full surface; use map for custom encoding placement.
 
 ## Next Steps
 - Integrate AI EC (phase 3): Train models on surface map for error prediction, puzzle-like patterns.
 - See [`plan.md`](plan.md) for full roadmap.
 
 For issues, ensure running from src/ or use `python -m FloppyAI.src.main` from FloppyAI/.