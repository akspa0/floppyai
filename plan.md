# FloppyAI: Exploring Advanced Flux Encoding on Floppy Disks

## Introduction

This plan outlines the development of a tool to interface with KryoFlux hardware via its DTC CLI, analyze flux stream files (.raw) from 5.25" floppy disks (including blanks), and explore encoding significantly more data than standard sector-based methods allow. The focus is on leveraging magnetic flux patterns—including noise, weak bits, and non-standard transitions—for dense, readable storage. AI will enhance encoding with novel error correction (EC) that avoids traditional sectors, potentially using ML to generate resilient flux patterns.

The project builds on the existing `src/` Python codebase (e.g., flux handling in `kryoflux/interface.py`, decoding in `decoding/`), extending it for broader flux exploration. We'll create a new `FloppyAI/` subfolder under the root for all new code, docs, and tests. Sample .raw files in `example_stream_data/` will be used for initial analysis; user-provided blanks/samples will enable hardware testing.

**Key Goals:**
- Analyze blank media flux to map surface characteristics (noise, coercivity variations).
- Develop custom flux encoding for >standard density (target: 2-3x via variable cell lengths, embedded redundancy).
- Integrate AI for EC: ML models to predict/compensate flux errors, encode data in "puzzle-like" patterns.
- Ensure readability: Custom decoder must recover data without standard tools.

**Language Choice:** Python 3.8+ (extends existing; easy subprocess for DTC CLI, libraries like NumPy/SciPy for flux analysis, scikit-learn/TensorFlow for AI). C# deferred unless DTC SDK requires it (CLI suffices initially).

## Feasibility Assessment

Floppy disks store data as magnetic flux transitions (reversals) on tracks. Standard 5.25" DD (360 RPM, ~250kbps) holds ~180KB/side; HD ~360KB. Flux resolution via KryoFlux: ~1-4µs timings, yielding ~100-200KB raw flux/track (but compressible to ~1-2KB effective data after encoding).

**Potential for "More Data":**
- **Standard Limits:** MFM/FM encoding + ECC limits density; sectors waste space.
- **Flux Exploitation:** Use full revolution (~0.167s at 360 RPM) for continuous data. Variable-length cells (e.g., run-length limited codes) could pack 2x bits by shortening cells where media allows. Embed data in noise/edges (e.g., weak bits as ternary states).
- **Challenges:** Media imperfections (dropout, bias) limit density; readability requires consistent transitions. Blanks show baseline noise—analyze for "hidden" capacity (e.g., subtle coercivity gradients).
- **AI Role:** Train models on flux samples to generate EC: e.g., autoencoders for dense packing + error prediction (better than Reed-Solomon for analog flux).
- **Viability:** Feasible for 1.5-2x density with custom encoding; 3x+ risky (may not read back reliably). Test on blanks to measure error rates.

**Hardware/Software Context:**
- DTC CLI (in `lib/kryoflux_3.50_windows_r2/dtc/dtc.exe`): Read/write streams (`dtc read`, `dtc write`), analyze (`dtc analyse`).
- Licenses: SPS/KryoFlux for non-commercial; IPF/CAPSImg free for preservation—adhere strictly.
- Samples: `example_stream_data/*.raw` for analysis; user blanks for writes.

Estimated Capacity: ~500KB-1MB/disk (vs. 360KB standard), readable by custom tool.

## Architecture Overview

- **Core Components:**
  - **DTC Wrapper:** Python subprocess calls to DTC for stream I/O (e.g., `dtc -i0 -t<track> -s<side> -f<output.raw> read`).
  - **Flux Parser/Analyzer:** Parse .raw (flux timings as uint32 ns), visualize (Matplotlib heatmaps of transitions/noise), detect anomalies (e.g., weak bits via variance).
  - **Custom Encoder:** Input data → flux transitions (e.g., adaptive RLL code + AI-optimized redundancy).
  - **Custom Decoder:** Flux → bits, with AI EC recovery (e.g., neural net to inpaint errors).
  - **AI Module:** Flux dataset → ML models (scikit-learn for clustering noise; PyTorch for seq2seq encoding).

- **Data Flow:**
  1. Read blank: DTC → .raw → Parse → Analyze surface map (noise profile per track).
  2. Encode: Data + noise map → AI EC → Flux schedule → DTC write.
  3. Decode: Read → Parse → AI recovery → Data.

- **File Structure (in FloppyAI/):**
  ```
  FloppyAI/
  ├── README.md          # Project overview
  ├── plan.md            # This file
  ├── requirements.txt   # Python deps (numpy, scipy, matplotlib, scikit-learn, torch)
  ├── src/
  │   ├── dtc_wrapper.py # Subprocess interface
  │   ├── flux_analyzer.py # Parse/visualize .raw
  │   ├── custom_encoder.py # Flux generation
  │   ├── ai_ec.py       # ML models
  │   └── main.py        # CLI: analyze, encode, decode
  ├── data/              # Processed samples (e.g., noise maps)
  ├── tests/             # Unit tests, sim flux
  └── docs/              # DTC commands, flux specs
  ```

- **Dependencies:** Extend `src/requirements.txt`; add ML libs. No C# needed.

## Implementation Phases

1. **Setup & Analysis (Week 1):**
   - Create FloppyAI/ structure.
   - Implement DTC wrapper: Test read on `example_stream_data/unknown-stream00.0.raw` (verify parsing).
   - Flux analyzer: Load .raw, plot transitions (time vs. flux intervals), compute stats (mean cell length, noise variance). Identify blank patterns (uniform noise vs. recorded).
   - Output: Surface maps (JSON/CSV per track: noise profile, max density estimate).

2. **Custom Encoding Prototype (Weeks 2-3):**
   - Baseline: Extend `src/encoding/` (Manchester/sigma-delta) for data bits → flux.
   - Advanced: Variable encoding—short cells for 0s, long for 1s (RLL(1,7)-like); embed in full track (no gaps).
   - Density test: Encode test data (e.g., 1KB) to flux; simulate write/read to measure bits/revolution.
   - Write tool: `python main.py write --input data.bin --track 0 --side 0` → DTC stream → write to blank.

3. **AI Integration (Weeks 4-5):**
   - Dataset: Analyze 10+ samples (blanks + recorded) → flux vectors (e.g., 1000 transitions/track).
   - Models:
     - Analysis: Unsupervised (KMeans) to cluster flux anomalies.
     - EC: Supervised seq2seq (LSTM) to encode data + redundancy; decoder predicts errors from noisy flux.
     - Train: Augment with simulated noise (add Gaussian to timings).
   - Enhance: AI generates "puzzle" flux (e.g., data hidden in weak-bit patterns, recoverable via ML).

4. **Full Tool & Testing (Week 6):**
   - CLI: `analyze <stream.raw>`, `encode <data.bin> --density high`, `decode <stream.raw>`.
   - Hardware tests: Write to blanks, read back, compute BER (bit error rate). Compare densities.
   - Feasibility metrics: Density gain, readability (90%+ recovery), EC effectiveness (tolerate 10% flux errors).

5. **Documentation & Iteration:**
   - Update README with usage, results.
   - Risks mitigation: Fallback to standard EC if AI underperforms.

## Risks and Limitations

- **Technical:** Flux precision limited by drive/KryoFlux (~1µs); high density may cause write errors or unreadable media. Blanks vary (age, brand).
- **Hardware:** 5.25" drives finicky; ensure alignment. DTC write experimental (per RELEASE.txt).
- **Licenses:** Non-commercial only (SPS/KryoFlux); no distribution of tools/images without permission.
- **AI:** Overfitting to samples; compute-intensive training. Start simple (rule-based EC) before ML.
- **Feasibility Gaps:** If >2x density unreadable, pivot to "puzzle" use (e.g., steganography in flux noise).
- **Time/Cost:** Hardware tests need user blanks; simulate extensively.

## Next Steps

1. User review/approve this plan.
2. Switch to Code mode for implementation (extend src/ as needed).
3. Provide sample blanks paths if not in example_stream_data/.
4. If C# preferred, reassess for DTC .NET bindings.

This plan ensures methodical progress toward a novel flux-storage tool with AI enhancements.