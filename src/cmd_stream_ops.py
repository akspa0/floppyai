import argparse
import os
import random
import numpy as np
from pathlib import Path

from flux_analyzer import FluxAnalyzer
from dtc_wrapper import DTCWrapper
from custom_encoder import CustomEncoder
from utils.io_paths import get_output_dir


def analyze_stream(args):
    """Analyze a .raw stream file."""
    run_dir = get_output_dir(args.output_dir)
    analyzer = FluxAnalyzer()
    try:
        parsed = analyzer.parse(args.input)
        print("Parsed Data:")
        stats = parsed.get('stats', {})
        print(f"  Total Fluxes: {stats.get('total_fluxes', 0)}")
        print(f"  Mean Interval: {stats.get('mean_interval_ns', 0):.2f} ns")
        print(f"  Std Dev: {stats.get('std_interval_ns', 0):.2f} ns")
        print(f"  Num Revolutions: {stats.get('num_revolutions', 0)}")
        if not stats:
            print("  No flux data - possibly unformatted blank or parsing issue.")

        try:
            analysis = analyzer.analyze()
            print("\nAnalysis:")
            anomalies = analysis.get('anomalies', {})
            noise_profile = analysis.get('noise_profile', {})
            print(f"  Short Cells: {len(anomalies.get('short_cells', []))}")
            print(f"  Long Intervals: {len(anomalies.get('long_intervals', []))}")
            print(f"  Avg Variance (Noise): {noise_profile.get('avg_variance', 0):.2f}")
            print(f"  Density Estimate: {analysis.get('density_estimate_bits_per_rev', 0)} bits/rev")
        except Exception as e:
            print(f"  Analysis error: {e}")
            print("  No analysis - check flux data.")

        # Generate visualizations in run_dir
        base_name = str(run_dir / Path(args.input).stem)
        analyzer.visualize(base_name, "intervals")
        analyzer.visualize(base_name, "histogram")
        if len(analyzer.revolutions) > 1:
            analyzer.visualize(base_name, "heatmap")

        print(f"\nVisualizations saved in {run_dir} with prefix '{Path(args.input).stem}_'.")
    except Exception as e:
        print(f"Error analyzing {args.input}: {e}")
        return 1
    return 0


def read_track(args):
    """Read a track from hardware."""
    run_dir = get_output_dir(args.output_dir)
    output_raw = str(run_dir / f"read_track_{args.track}_{args.side}.raw")
    wrapper = DTCWrapper(simulation_mode=args.simulate)
    success = wrapper.read_track(
        track=args.track,
        side=args.side,
        output_raw_path=output_raw,
        revolutions=args.revolutions
    )
    if success:
        print(f"Read saved to {output_raw}")
        # Optionally analyze the new stream
        if getattr(args, 'analyze', False):
            analyze_stream(argparse.Namespace(input=output_raw, output_dir=run_dir))
    else:
        return 1
    return 0


def write_track(args):
    """Write a stream to a track on hardware."""
    wrapper = DTCWrapper(simulation_mode=args.simulate)
    success = wrapper.write_track(
        input_raw_path=args.input,
        track=args.track,
        side=args.side
    )
    if success:
        run_dir = get_output_dir(args.output_dir)
        print(f"Write log in {run_dir}")
    return 0 if success else 1


def _generate_pattern_data(pattern: str, revolutions: int, cell_length: int, density: float = 1.0, seed: int = None) -> np.ndarray:
    """Generate flux data for different test patterns."""
    if seed is not None:
        np.random.seed(seed)

    # Base cell length adjusted by density
    base_cell = int(cell_length / density)
    min_cell = int(base_cell * 0.5)  # 50% minimum
    max_cell = int(base_cell * 2.0)  # 200% maximum

    # Generate multiple revolutions of data
    all_fluxes = []

    for rev in range(revolutions):
        if pattern == 'random':
            # Random flux intervals
            rev_fluxes = np.random.randint(min_cell, max_cell + 1, 1000)

        elif pattern == 'prbs7':
            # PRBS7 pattern - pseudo-random binary sequence
            # Simple LFSR implementation for 7-bit PRBS
            lfsr = 0b1000000 if rev == 0 else ((np.random.randint(0, 127) & 0b1111111) | 0b1000000)
            rev_fluxes = []
            for _ in range(1000):
                bit = lfsr & 1
                lfsr = (lfsr >> 1) | ((bit ^ ((lfsr >> 1) & 1)) << 6)
                cell_len = base_cell if bit == 0 else int(base_cell * 1.5)
                rev_fluxes.append(max(min_cell, min(max_cell, cell_len)))

        elif pattern == 'alt':
            # Alternating pattern
            rev_fluxes = []
            use_long = True
            for _ in range(1000):
                cell_len = int(base_cell * 1.5) if use_long else base_cell
                rev_fluxes.append(max(min_cell, min(max_cell, cell_len)))
                use_long = not use_long

        elif pattern == 'zeros':
            # All zeros pattern (long cells)
            rev_fluxes = np.full(1000, max_cell)

        elif pattern == 'ones':
            # All ones pattern (short cells)
            rev_fluxes = np.full(1000, min_cell)

        elif pattern == 'sweep':
            # Frequency sweep pattern
            rev_fluxes = []
            for i in range(1000):
                # Create a sweep from min to max cell length
                progress = i / 999.0
                cell_len = int(min_cell + (max_cell - min_cell) * (0.5 + 0.5 * np.sin(progress * 2 * np.pi)))
                rev_fluxes.append(cell_len)

        else:
            # Default to random pattern
            rev_fluxes = np.random.randint(min_cell, max_cell + 1, 1000)

        all_fluxes.extend(rev_fluxes)

    return np.array(all_fluxes, dtype=np.uint32)


def generate_dummy(args):
    """Generate a test stream with various patterns for experiments."""
    run_dir = get_output_dir(args.output_dir)

    # Get pattern from args or use default
    pattern = getattr(args, 'pattern', 'random')
    seed = getattr(args, 'seed', None)

    # Generate filename
    safe_pattern = pattern.replace('/', '_')
    filename = f"generated_{safe_pattern}_t{args.track}_s{args.side}.raw"
    if seed is not None:
        filename = f"generated_{safe_pattern}_s{seed}_t{args.track}_s{args.side}.raw"

    output_raw = str(run_dir / filename)

    # Generate flux data based on pattern
    flux_data = _generate_pattern_data(
        pattern=pattern,
        revolutions=args.revolutions,
        cell_length=args.cell_length,
        density=getattr(args, 'density', 1.0),
        seed=seed
    )

    # Create a simple .raw file format
    # In a real implementation, this would use the proper KryoFlux format
    # For now, create a basic binary format with flux intervals
    with open(output_raw, 'wb') as f:
        # Write header (simple format)
        f.write(b'FLUX')  # Magic bytes
        f.write(np.uint32(len(flux_data)))  # Number of intervals
        f.write(np.uint32(args.revolutions))  # Number of revolutions

        # Write flux data
        flux_data.astype(np.uint32).tobytes()
        f.write(flux_data.tobytes())

    print(f"Generated pattern '{pattern}' saved to {output_raw}")
    print(f"Pattern: {pattern}")
    print(f"Intervals: {len(flux_data)}")
    print(f"Revolutions: {args.revolutions}")
    print(f"Cell length: {args.cell_length} ns")
    print(f"Average interval: {np.mean(flux_data):.2f} ns")
    print(f"Std deviation: {np.std(flux_data):.2f} ns")

    # Analyze the generated pattern
    if getattr(args, 'analyze', False):
        analyze_stream(argparse.Namespace(input=output_raw, output_dir=run_dir))

    return 0


def encode_data(args):
    """Encode binary data to .raw using custom encoder."""
    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}")
        return 1

    run_dir = get_output_dir(args.output_dir)
    if args.output is None:
        output_raw = str(run_dir / f"encoded_track_{args.track}_{args.side}.raw")
    else:
        output_raw = args.output

    # Read input data
    with open(args.input, 'rb') as f:
        data_bytes = f.read()

    if len(data_bytes) == 0:
        print("No data in input file")
        return 1

    # Create encoder
    encoder = CustomEncoder(
        density=args.density,
        variable_mode=args.variable
    )

    # Encode
    flux_intervals = encoder.encode_data(data_bytes, num_revs=args.revolutions)

    # Generate .raw
    encoder.generate_raw(
        flux_intervals, args.track, args.side, output_raw,
        num_revs=args.revolutions
    )

    # Calculate and print density
    achieved_density = encoder.calculate_density(data_bytes)
    print(f"Encoded {len(data_bytes)*8} bits to {output_raw}")
    print(f"Achieved density: {achieved_density:.1f} bits per revolution")
    print(f"Total flux intervals: {len(flux_intervals)}")

    # Optionally write
    if getattr(args, 'write', False):
        wrapper = DTCWrapper(simulation_mode=args.simulate)
        wrapper.write_track(
            input_raw_path=output_raw,
            track=args.track,
            side=args.side
        )
        print("Write completed (or simulated)")

    # Optionally analyze
    if getattr(args, 'analyze', False):
        analyze_stream(argparse.Namespace(input=output_raw, output_dir=run_dir))

    return 0
