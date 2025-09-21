import subprocess
import os
import sys
from pathlib import Path
import tempfile
import struct
import numpy as np

class DTCWrapper:
    """
    Wrapper for KryoFlux DTC CLI tool.
    Interfaces with dtc.exe to read/write streams from/to hardware or simulate.
    Assumes dtc.exe in ../lib/kryoflux_3.50_windows_r2/dtc/dtc.exe relative to src/.
    """

    def __init__(self, dtc_path=None, simulation_mode=False):
        """
        Args:
            dtc_path (str): Full path to dtc.exe (optional; defaults to relative)
            simulation_mode (bool): If True, simulate operations without hardware
        """
        if dtc_path is None:
            base_dir = Path(__file__).parent.parent
            dtc_path = base_dir / 'lib' / 'kryoflux_3.50_windows_r2' / 'dtc' / 'dtc.exe'
        self.dtc_path = str(dtc_path)
        if not os.path.exists(self.dtc_path) and not simulation_mode:
            raise FileNotFoundError(f"DTC executable not found at {self.dtc_path}")
        self.simulation_mode = simulation_mode
        self.default_revs = 3  # Default revolutions for reads

    def _run_dtc(self, args):
        """
        Run dtc.exe with given arguments.
        
        Args:
            args (list): List of command-line args
            
        Returns:
            tuple: (stdout, stderr, returncode)
        """
        if self.simulation_mode:
            print(f"[SIMULATION] Would run: {self.dtc_path} {' '.join(args)}")
            return "Simulated output", "", 0
        
        cmd = [self.dtc_path] + args
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return result.stdout, result.stderr, result.returncode
        except subprocess.CalledProcessError as e:
            print(f"DTC error: {e.stderr}")
            raise

    def read_track(self, track, side, output_raw_path, revolutions=None, drive=0):
        """
        Read a track from hardware to .raw stream file.
        
        Args:
            track (int): Track number (0-79)
            side (int): Side (0 or 1)
            output_raw_path (str): Path to save .raw file
            revolutions (int): Number of revolutions to read (default: self.default_revs)
            drive (int): Drive index (default: 0)
            
        Returns:
            bool: Success
        """
        if revolutions is None:
            revolutions = self.default_revs
        args = [
            '-d', str(drive),  # Drive
            '-i', '0',  # Input: hardware
            '-t', str(track),
            '-s', str(side),
            '-r', str(revolutions),
            '-f', output_raw_path,  # Output file
            'read'
        ]
        try:
            stdout, stderr, _ = self._run_dtc(args)
            print(f"Read track {track} side {side} to {output_raw_path}")
            print(stdout)
            return True
        except Exception as e:
            print(f"Failed to read track: {e}")
            return False

    def write_track(self, input_raw_path, track, side, drive=0):
        """
        Write a .raw stream file to a track on hardware.
        
        Args:
            input_raw_path (str): Path to input .raw file
            track (int): Track number
            side (int): Side
            drive (int): Drive index
            
        Returns:
            bool: Success
        """
        if not os.path.exists(input_raw_path):
            raise FileNotFoundError(f"Input stream not found: {input_raw_path}")
        
        args = [
            '-d', str(drive),
            '-i', '0',  # Input: hardware (for write, but stream from file)
            '-t', str(track),
            '-s', str(side),
            '-f', input_raw_path,
            'write'
        ]
        try:
            stdout, stderr, _ = self._run_dtc(args)
            print(f"Wrote {input_raw_path} to track {track} side {side}")
            print(stdout)
            return True
        except Exception as e:
            print(f"Failed to write track: {e}")
            return False

    def analyze_stream(self, input_raw_path, output_path=None, format='text'):
        """
        Analyze a .raw stream file using DTC (e.g., generate CT raw or IPF).
        
        Args:
            input_raw_path (str): Input .raw
            output_path (str): Output analysis file (optional; uses temp if None)
            format (str): Output format ('ctr', 'ipf', 'text')
            
        Returns:
            str: Path to analysis output
        """
        if output_path is None:
            output_path = tempfile.mktemp(suffix='.ctr' if format == 'ctr' else '.ipf')
        
        args = [
            '-i', '21',  # Input: raw stream
            '-o', format,  # Output format (21=raw stream in, ctr/ipf out)
            '-f', input_raw_path,
            '-g', output_path,  # Output file
            'convert'
        ]
        try:
            stdout, stderr, _ = self._run_dtc(args)
            print(f"Analyzed {input_raw_path} to {output_path}")
            print(stdout)
            return output_path
        except Exception as e:
            print(f"Failed to analyze stream: {e}")
            return None

    def generate_dummy_stream(self, track, side, output_raw_path, revolutions=1, cell_length_ns=4000):
        """
        Simulation: Generate a dummy .raw with uniform flux intervals.
        For testing custom flux without hardware.
        
        Args:
            track/side: For filename
            output_raw_path: Output path
            revolutions: Number of revs
            cell_length_ns: Nominal cell length (ns)
        """
        print(f"[SIMULATION] Generating dummy stream for track {track} side {side}")
        # Simple header + fluxes
        header = b'KryoFlux Stream\x00\x00\x00\x00' + struct.pack('<II', 1, 0) + b'\x00' * 8  # Basic header
        flux_data = b''
        for _ in range(revolutions):
            flux_data += struct.pack('<I', 1)  # Index count
            num_cells = int(200000000 / cell_length_ns)  # ~0.2s rev at 300RPM
            for _ in range(num_cells):
                interval = cell_length_ns + np.random.normal(0, 100)  # Add noise
                flux_data += struct.pack('<I', int(interval))
            flux_data += struct.pack('<I', 0)  # End rev
        
        with open(output_raw_path, 'wb') as f:
            f.write(header + flux_data)
        print(f"Dummy stream saved to {output_raw_path}")

# Example usage
if __name__ == "__main__":
    wrapper = DTCWrapper(simulation_mode=True)  # Start with sim
    # wrapper.read_track(0, 0, "test_read.raw", revolutions=2)
    # wrapper.write_track("test_read.raw", 0, 0)
    wrapper.generate_dummy_stream(0, 0, "dummy_track00.0.raw")