#!/usr/bin/env python3
"""
FloppyAI Experiments Module
Orchestrates experiment matrices for systematic flux stream analysis.
"""
import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

from utils.io_paths import get_output_dir
from utils.json_io import dump_json

# Direct-run path to main CLI script
MAIN_SCRIPT = str(Path(__file__).with_name("main.py"))


def confirm_hardware_safety(args) -> bool:
    """Confirm with user that they understand hardware safety requirements."""
    if getattr(args, 'simulate', True):
        return True

    print("\n" + "="*60)
    print("HARDWARE SAFETY WARNING")
    print("="*60)
    print("You are about to run experiments on actual floppy disk hardware.")
    print("This may damage or destroy disks if not done properly.")
    print()
    print("SAFETY REQUIREMENTS:")
    print("1. Use SACRIFICIAL MEDIA ONLY - disks you don't mind losing")
    print("2. Start with OUTER TRACKS (0-9) to minimize head damage risk")
    print("3. Monitor disk temperature - stop if disks get too hot")
    print("4. Have spare disks available - experiments may wear out media")
    print("5. Keep drive heads clean and properly aligned")
    print()
    print("RECOMMENDED SETTINGS:")
    print("- Use --simulate mode first to test without hardware")
    print("- Limit to tracks 0-9 initially")
    print("- Use short run times and monitor carefully")
    print("- Allow cooldown periods between experiments")
    print("="*60)

    try:
        response = input("\nDo you acknowledge these safety requirements? (yes/no): ").strip().lower()
        return response in ['yes', 'y']
    except (KeyboardInterrupt, EOFError):
        print("\nOperation cancelled for safety.")
        return False


def validate_experiment_safety(args) -> bool:
    """Validate that experiment parameters are reasonably safe."""
    # Check track range
    tracks = getattr(args, 'tracks', list(range(0, 10)))
    max_track = max(tracks) if tracks else 0

    if max_track > 20:
        print(f"WARNING: Using tracks up to {max_track} - consider starting with outer tracks (0-9)")
        print("Inner tracks have higher risk of head crashes and media damage.")

    # Check repetition count
    repetitions = getattr(args, 'repetitions', 3)
    if repetitions > 5:
        print(f"WARNING: {repetitions} repetitions per combination is high")
        print("Consider starting with fewer repetitions to test stability.")

    # Check if using simulation mode
    if getattr(args, 'simulate', True):
        print("✓ Using simulation mode - safe for testing")
        return True

    # Final safety check
    return confirm_hardware_safety(args)


def run_experiment_matrix(args):
    """Run a matrix of experiments with different parameters."""
    run_dir = get_output_dir(args.output_dir)

    # Validate safety before proceeding
    if not validate_experiment_safety(args):
        print("Experiment cancelled for safety reasons.")
        return 1

    # Default to simulate mode for safety
    simulate = getattr(args, 'simulate', True)
    if not simulate:
        print("WARNING: Hardware mode enabled. Ensure you are using sacrificial media!")
        print("Add --simulate to run in simulation mode.")

    # Experiment configuration
    experiment_config = {
        'experiment_name': getattr(args, 'experiment', 'flux_analysis'),
        'simulate': simulate,
        'timestamp': datetime.datetime.now().isoformat(),
        'parameters': {
            'patterns': getattr(args, 'patterns', ['random', 'prbs7', 'alt']),
            'densities': getattr(args, 'densities', [0.5, 1.0, 1.5, 2.0]),
            'tracks': getattr(args, 'tracks', list(range(0, 10))),  # Default to outer tracks
            'sides': getattr(args, 'sides', [0, 1]),
            'revolutions': getattr(args, 'revolutions', 3),
            'repetitions': getattr(args, 'repetitions', 3),
        }
    }

    # Save experiment manifest
    manifest_path = run_dir / 'experiment_manifest.json'
    dump_json(manifest_path, experiment_config)
    print(f"Experiment manifest saved to {manifest_path}")

    # Create results structure
    results = {
        'experiment_config': experiment_config,
        'matrix_runs': [],
        'summary': {
            'total_runs': 0,
            'successful_runs': 0,
            'failed_runs': 0,
            'errors': []
        }
    }

    # Generate experiment matrix
    matrix = generate_experiment_matrix(experiment_config)

    print(f"Running experiment matrix with {len(matrix)} combinations...")

    for i, combo in enumerate(matrix):
        print(f"\n--- Run {i+1}/{len(matrix)} ---")
        print(f"Parameters: {combo}")

        try:
            # Generate test pattern
            generate_result = run_generate(combo, run_dir, simulate)
            if not generate_result['success']:
                results['summary']['errors'].append(f"Generation failed: {combo}")
                continue

            # Write to disk
            write_result = run_write(generate_result['output_file'], combo, run_dir, simulate)
            if not write_result['success']:
                results['summary']['errors'].append(f"Write failed: {combo}")
                continue

            # Read back multiple times
            read_results = run_read(combo, run_dir, simulate)
            if not read_results['success']:
                results['summary']['errors'].append(f"Read failed: {combo}")
                continue

            # Analyze results
            analyze_result = run_analyze(combo, run_dir)
            if not analyze_result['success']:
                results['summary']['errors'].append(f"Analysis failed: {combo}")
                continue

            # Record successful run
            run_record = {
                'parameters': combo,
                'files': {
                    'generated': generate_result['output_file'],
                    'surface_map': analyze_result['surface_map']
                },
                'metrics': analyze_result['metrics'],
                'success': True
            }

            results['matrix_runs'].append(run_record)
            results['summary']['successful_runs'] += 1

        except Exception as e:
            print(f"Run failed with exception: {e}")
            results['summary']['errors'].append(f"Exception in run {combo}: {e}")
            results['summary']['failed_runs'] += 1

    results['summary']['total_runs'] = len(results['matrix_runs']) + results['summary']['failed_runs']

    # Save results
    results_path = run_dir / 'experiment_results.json'
    dump_json(results_path, results)
    print(f"\nExperiment results saved to {results_path}")

    # Print summary
    print("\n=== Experiment Summary ===")
    print(f"Total runs: {results['summary']['total_runs']}")
    print(f"Successful: {results['summary']['successful_runs']}")
    print(f"Failed: {results['summary']['failed_runs']}")
    if results['summary']['errors']:
        print(f"Errors: {len(results['summary']['errors'])}")

    return 0 if results['summary']['successful_runs'] > 0 else 1


def generate_experiment_matrix(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate all combinations for the experiment matrix."""
    matrix = []

    for pattern in config['parameters']['patterns']:
        for density in config['parameters']['densities']:
            for track in config['parameters']['tracks']:
                for side in config['parameters']['sides']:
                    for rep in range(config['parameters']['repetitions']):
                        combo = {
                            'pattern': pattern,
                            'density': density,
                            'track': track,
                            'side': side,
                            'repetition': rep + 1,
                            'revolutions': config['parameters']['revolutions']
                        }
                        matrix.append(combo)

    return matrix


def run_generate(combo: Dict[str, Any], run_dir: Path, simulate: bool) -> Dict[str, Any]:
    """Generate a test pattern stream."""
    track = combo['track']
    side = combo['side']
    density = combo['density']
    pattern = combo['pattern']

    # Generate filename
    safe_pattern = pattern.replace('/', '_')
    filename = f"exp_{safe_pattern}_d{density}_t{track:02d}_s{side}.raw"
    output_file = run_dir / filename

    # Build command
    cmd = [
        sys.executable, MAIN_SCRIPT, "generate",
        str(track), str(side),
        "--output", str(output_file),
        "--density", str(density),
        "--pattern", pattern,
        "--revs", str(combo['revolutions'])
    ]

    if simulate:
        cmd.append("--simulate")

    print(f"Generating: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return {
            'success': True,
            'output_file': str(output_file),
            'stdout': result.stdout,
            'stderr': result.stderr
        }
    except subprocess.CalledProcessError as e:
        return {
            'success': False,
            'output_file': str(output_file),
            'stdout': e.stdout,
            'stderr': e.stderr,
            'error': str(e)
        }


def run_write(input_file: str, combo: Dict[str, Any], run_dir: Path, simulate: bool) -> Dict[str, Any]:
    """Write generated pattern to disk."""
    track = combo['track']
    side = combo['side']

    # Build command
    cmd = [
        sys.executable, MAIN_SCRIPT, "write",
        input_file, str(track), str(side)
    ]

    if simulate:
        cmd.append("--simulate")

    print(f"Writing: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return {
            'success': True,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
    except subprocess.CalledProcessError as e:
        return {
            'success': False,
            'stdout': e.stdout,
            'stderr': e.stderr,
            'error': str(e)
        }


def run_read(combo: Dict[str, Any], run_dir: Path, simulate: bool) -> Dict[str, Any]:
    """Read back from disk multiple times."""
    track = combo['track']
    side = combo['side']
    revolutions = combo['revolutions']

    # Build command
    cmd = [
        sys.executable, MAIN_SCRIPT, "read",
        str(track), str(side),
        "--revs", str(revolutions)
    ]

    if simulate:
        cmd.append("--simulate")

    print(f"Reading: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return {
            'success': True,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
    except subprocess.CalledProcessError as e:
        return {
            'success': False,
            'stdout': e.stdout,
            'stderr': e.stderr,
            'error': str(e)
        }


def run_analyze(combo: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
    """Analyze the results."""
    track = combo['track']
    side = combo['side']
    pattern = combo['pattern']
    density = combo['density']

    # Build command
    cmd = [
        sys.executable, MAIN_SCRIPT, "analyze",
        str(run_dir / f"exp_{pattern.replace('/', '_')}_d{density}_t{track:02d}_s{side}.raw")
    ]

    print(f"Analyzing: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)

        # Parse output to extract metrics
        metrics = parse_analysis_output(result.stdout)

        return {
            'success': True,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'metrics': metrics
        }
    except subprocess.CalledProcessError as e:
        return {
            'success': False,
            'stdout': e.stdout,
            'stderr': e.stderr,
            'error': str(e)
        }


def parse_analysis_output(output: str) -> Dict[str, Any]:
    """Parse analysis output to extract key metrics."""
    metrics = {}

    # Simple parsing - in a real implementation, this would be more sophisticated
    lines = output.split('\n')
    for line in lines:
        if 'Mean Interval:' in line:
            try:
                metrics['mean_interval_ns'] = float(line.split(':')[1].strip().split()[0])
            except:
                pass
        elif 'Density Estimate:' in line:
            try:
                metrics['density_estimate'] = float(line.split(':')[1].strip().split()[0])
            except:
                pass
        elif 'Total Fluxes:' in line:
            try:
                metrics['total_fluxes'] = int(line.split(':')[1].strip().split()[0])
            except:
                pass

    return metrics


def main():
    """Main entry point for experiments command."""
    parser = argparse.ArgumentParser(
        description="FloppyAI Experiments: Run systematic flux analysis experiments"
    )

    subparsers = parser.add_subparsers(dest="experiment_command", help="Experiment commands")

    # Matrix experiment
    matrix_parser = subparsers.add_parser("matrix", help="Run experiment matrix")
    matrix_parser.add_argument("--experiment", default="flux_analysis", help="Experiment name")
    matrix_parser.add_argument("--patterns", nargs='+', default=['random', 'prbs7', 'alt'],
                              help="Test patterns to use")
    matrix_parser.add_argument("--densities", nargs='+', type=float, default=[0.5, 1.0, 1.5, 2.0],
                              help="Density multipliers to test")
    matrix_parser.add_argument("--tracks", nargs='+', type=int, default=list(range(0, 10)),
                              help="Track numbers to test (default: 0-9 for outer tracks)")
    matrix_parser.add_argument("--sides", nargs='+', type=int, choices=[0, 1], default=[0, 1],
                              help="Sides to test")
    matrix_parser.add_argument("--revolutions", type=int, default=3,
                              help="Revolutions to read/write per test")
    matrix_parser.add_argument("--repetitions", type=int, default=3,
                              help="Number of repetitions per parameter combination")
    matrix_parser.add_argument("--no-simulate", action="store_false", dest="simulate",
                              help="Disable simulation mode (use with caution!)")
    matrix_parser.add_argument("--output-dir", help="Custom output directory")
    matrix_parser.set_defaults(func=run_experiment_matrix)

    args = parser.parse_args()
    if not args.experiment_command:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
