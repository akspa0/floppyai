# FloppyAI: Advanced Flux Analysis and Encoding Tool

## Overview
FloppyAI is a Python-based tool for interfacing with KryoFlux DTC CLI to read/write flux streams (.raw files), analyze magnetic media characteristics (noise, anomalies on blanks), and prototype custom flux encoding for higher data density. It builds on the existing src/ for flux handling and extends it for AI-enhanced error correction.

Recent enhancements include RPM normalization for accurate density estimates, copy protection detection via protection_scores and anomalies, surface modeling with side-specific heatmaps, and local LLM integration for human-readable summaries.

## Installation
1. Ensure Python 3.8+ and pip.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
   (NumPy, SciPy, Matplotlib for analysis/visualization; scikit-learn/Torch for ML; lmstudio for official LM Studio SDK integration.)

3. For hardware: Connect KryoFlux to PC; ensure drivers installed. DTC.exe is in ../lib/kryoflux_3.50_windows_r2/dtc/.

4. For LLM summaries (optional): Run LM Studio on localhost:1234 (or specify --lm-host) with a local model loaded. Recommended models for clean technical summaries (avoid models that show internal thinking tags):
   - **Qwen2-7B-Instruct**: Produces clean, technical summaries without thinking tags
   - **Gemma 7B Instruct**: Good for structured technical analysis
   - **Llama 3 8B Instruct**: Reliable but may occasionally show reasoning
   - **Mistral 7B Instruct**: Efficient but test for clean output

   The system prompt explicitly instructs models to avoid thinking tags and provide only final professional summaries. For best results, test models to ensure they follow the "output only final analysis" instruction.

## Usage
Run from FloppyAI/ directory or src/ subdirectory:
```
cd FloppyAI/src  # Or run from root with python -m FloppyAI.src.main
python main.py --help
```
Global options (place before subcommand): --lm-host IP (default: localhost, port fixed to 1234), --lm-model MODEL (default: local-model, name of loaded model in LM Studio).
Example: python main.py --lm-host 192.168.1.131 --lm-model deepseek-r1-0528-qwen3-8b analyze_disk [path] --summarize

Note: Commands like analyze_disk do not take positional file arguments; use specific subcommands for single files.

### Commands

1. **Analyze a Stream File** (Decode and visualize .raw):
   ```
   python main.py analyze <input.raw> [--output-dir]
   ```
   - Parses flux transitions, computes stats (mean interval, noise variance).
   - Detects anomalies (short/long cells, weak bits via rev inconsistencies).
   - Generates plots: <input>_intervals.png (time series), <input>_hist.png (distribution), <input>_heatmap.png (if multi-rev).
   - Example: `python main.py analyze ../../example_stream_data/unknown-stream00.0.raw`
     - Output: Stats like "Mean Interval: 4000.50 ns", "Short Cells: 5", visualizations saved.
   - For blanks: High variance indicates surface irregularities; low anomalies = clean media.

2. **Read Track from Hardware**:
   ```
   python main.py read <track> <side> [--revs 3] [--simulate] [--analyze] [--output-dir]
   ```
   - Reads track/side (e.g., 0 0) to .raw using DTC.
   - --revs: Revolutions (default 3 for better analysis).
   - --simulate: Dry-run without hardware (no-op).
   - --analyze: Auto-analyze output .raw.
   - --rpm: Known RPM for normalization (default 360).
   - Use for blanks: Reveals baseline flux noise for custom encoding.

3. **Write Stream to Hardware**:
   ```
   python main.py write <input.raw> <track> <side> [--simulate] [--output-dir]
   ```
   - Writes .raw to track/side.
   - --simulate: Dry-run (logs command).
   - Test custom flux on blanks to explore readability.

4. **Generate Dummy Stream** (Build custom flux for testing):
   ```
   python main.py generate <track> <side> [--revs 1] [--cell 4000] [--analyze] [--output-dir]
   ```
   - Creates .raw with uniform intervals + noise (simulate one revolution).
   - --revs: Number of revolutions.
   - --cell: Nominal cell length ns (vary for density: shorter = higher density).
   - --analyze: Auto-analyze output.
   - --rpm: Known RPM for normalization (default 360).
   - Example: `python main.py generate 0 0 dummy.raw --cell 2000 --analyze`
     - Generates denser flux; analyze to see interval distribution.

5. **Encode Binary Data to Custom Stream** (Prototype higher density encoding):
   ```
   python main.py encode <input.bin> <track> <side> [--density 1.0] [--variable] [--revs 1] [--output <output.raw>] [--write] [--simulate] [--analyze] [--rpm 360] [--output-dir]
   ```
   - Encodes binary file to .raw using Manchester or variable RLL-like flux encoding.
   - --density: Scaling factor (>1.0 shortens cells for higher density; e.g., 2.0 for ~2x bits).
   - --variable: Use RLL-like variable cell lengths (short for 0s, long for 1s) for advanced packing.
   - --revs: Revolutions to fill (repeats data to embed continuously).
   - --output: Custom .raw path (default: encoded_track_X_Y.raw in timestamp dir).
   - --write: Auto-write .raw to hardware track/side after generation.
   - --simulate: Dry-run for --write (no hardware).
   - --analyze: Auto-analyze generated .raw (check density estimate vs. achieved).
   - --rpm: Known RPM for normalization (default 360).
   - Outputs achieved density (bits/rev) based on input size.
   - Example: `python main.py encode test_data.bin 0 0 --density 2.0 --variable --rpm 360 --analyze`
     - Encodes 1KB data at 2x density with variable cells; prints ~8192 bits/rev (vs. standard ~4000); analyzes for noise/readability.
   - For density testing: Compare bits/rev in output to standard (analyze dummy at density=1.0); higher = success if low anomalies.

6. **Decode Custom Stream** (Recover binary data from encoded .raw):
   ```
   python main.py decode <input.raw> [--density 1.0] [--variable] [--revs 1] [--output <output.bin>] [--expected <original.bin>] [--rpm 360] [--output-dir]
   ```
   - Decodes flux to binary using matching parameters.
   - --density: Expected density used in encoding.
   - --variable: Assume RLL-like variable cells.
   - --revs: Number of revolutions.
   - --output: Custom .bin path.
   - --expected: Original .bin for verification (reports % match, byte errors).
   - --rpm: Known RPM for normalization (default 360).
   - Example: `python main.py decode test_encoded.raw --density 2.0 --variable --rpm 360 --expected test_data.bin`
     - Outputs test_decoded.bin; verifies 100% recovery for all-zero data (perfect for blanks).

7. **Analyze Disk Surface** (Batch process streams for full disk map):
   ```
   python main.py analyze_disk [input] [--track N] [--side 0|1] [--rpm 360] [--output-dir] [--summarize]
   ```
   - input: Optional directory or single .raw file (default: ../example_stream_data/). Globs all *.raw if dir; auto-batches parent dir if single numbered file and siblings exist.
   - Parses track/side from filename ending in \d+\.\d+\.raw (e.g., Goofy00.0.raw → track 0 side 0); handles concatenated prefixes (e.g., blank180.1.raw → track 80 side 1).
   - Filters to tracks 0-83, sides 0-1; processes in order (00.0, 00.1, ..., 83.1), logging found vs expected (up to 168 files).
   - Use --track/--side for manual override if unparsable (applies to all files if no pattern).
   - --rpm: Known RPM for normalization/validation (default 360; scales partial reads to full rev, computes drift_pct ~0-5%, normalized densities).
   - Outputs surface_map.json: Per-track/side list of files with stats/analysis (normalized mean/std intervals, protection_score 0-1 from anomalies/variance, max_theoretical_density ~rev_time/min_int, is_protected >0.3); includes side summaries (avg_protection, likely_protected) and global (side_diff, packing_potential).
   - Generates combined PNG visualizations for entire disk (intervals, histogram, heatmap) and side_density_heatmap.png (bar charts of density per track/side, highlighting protection asymmetry).
   - --summarize: Auto-generates LLM-powered human-readable report (llm_summary.txt) via LM Studio (--lm-host HOST:1234, --lm-model MODEL); analyzes protection zones, RPM notes, packing suggestions.
   - Examples:
     - Default batch: `python main.py analyze_disk` (processes example_stream_data/)
     - Full disk dir: `python main.py analyze_disk ..\stream_dumps\GoofyExpress\goofy_express\kryoflux_stream`
     - With normalization/summary: `python main.py analyze_disk [path] --rpm 360 --summarize --lm-host localhost --lm-model llama3`
       - Outputs: JSON with protection on side 1 (score ~0.35, high variance/anomalies), heatmap showing ~500 bits/rev vs. 350 on side 0, summary like "Learned: Sparse fluxes for copy protection; pack more with density=1.8".
     - For protected disks: Detects schemes (e.g., weak bits via short_cells >40%, zoned in outer tracks).
   - Use for full disk: Maps surface, identifies coercivity/protection variations for adaptive encoding (e.g., higher density on clean zones).
   - Note: For single-file focus with per-file viz, use 'analyze'. Binary files like test_data.bin are for encode/decode, not flux analysis.

## Testing the Tooling
- **Round-trip:** Encode test_data.bin → test_encoded.raw, decode back; expect 100% match.
- **Density:** At 2.0, ~8192 bits/rev vs. standard ~4000; decoder recovers data despite 5% noise.
- **Disk Surface:** Run analyze_disk on blanks/protected dumps to explore; use map/heatmap for custom encoding placement; --summarize for insights.
- **LLM Summary Example**: With LM Studio running, --summarize produces reports like "Deductions: Side 1 protected... Packing: Try density>1.5 for 700 bits/rev".

## Next Steps
- Integrate AI EC (phase 3): Train models on surface map for error prediction, puzzle-like patterns.
- LLM Enhancements: Customize prompts or integrate more models via LM Studio.
- See [`plan.md`](plan.md) for full roadmap.

For issues, ensure running from src/ or use `python -m FloppyAI.src.main` from FloppyAI/.