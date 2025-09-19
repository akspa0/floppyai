import struct
import numpy as np
import matplotlib.pyplot as plt
import os

class FluxAnalyzer:
    """
    Tool for parsing and analyzing KryoFlux .raw stream files.
    Extracts flux transitions, computes statistics, detects anomalies,
    and visualizes patterns to understand media characteristics.
    """

    def __init__(self):
        self.flux_data = None  # np.array of flux intervals in ns
        self.index_positions = []  # List of index pulse positions
        self.stats = {}
        self.revolutions = []

    def parse(self, raw_path):
        """
        Parse a KryoFlux .raw file.
        
        Args:
            raw_path (str): Path to .raw file
            
        Returns:
            dict: Parsed data including fluxes, indices, revolutions
        """
        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"Stream file not found: {raw_path}")
        
        with open(raw_path, 'rb') as f:
            data = f.read()
        
        # Find end of header: null-terminated ASCII
        pos = data.find(b'\x00')
        if pos == -1:
            raise ValueError(f"Invalid .raw file '{raw_path}': no null terminator found in header.")
        flux_offset = pos + 1
        
        # Parse flux intervals, separating revolutions at index pulses (0xFFFFFFFF)
        flux_list = []
        self.revolutions = []
        self.index_positions = []
        current_rev = []
        num_indices = 0
        while flux_offset + 4 <= len(data):
            chunk = data[flux_offset:flux_offset + 4]
            flux_ns = struct.unpack('<I', chunk)[0]
            flux_offset += 4
            if flux_ns == 0xFFFFFFFF:
                num_indices += 1
                self.index_positions.append(flux_offset - 4)  # byte position of index
                if current_rev:
                    self.revolutions.append(np.array(current_rev, dtype=np.uint32))
                    flux_list.extend(current_rev)
                current_rev = []
                continue
            # Valid flux transition interval?
            if 50 <= flux_ns <= 20000000:  # reasonable range: 50ns min, 20ms max
                current_rev.append(flux_ns)
        
        # Append last revolution if any
        if current_rev:
            self.revolutions.append(np.array(current_rev, dtype=np.uint32))
            flux_list.extend(current_rev)
        
        self.flux_data = np.array(flux_list, dtype=np.uint32)
        
        # Compute basic stats
        if len(self.flux_data) > 0:
            total_time_ns = np.sum(self.flux_data)
            rev_time_ns = total_time_ns / max(1, len(self.revolutions))
            self.stats = {
                'total_fluxes': len(self.flux_data),
                'mean_interval_ns': float(np.mean(self.flux_data)),
                'std_interval_ns': float(np.std(self.flux_data)),
                'min_interval_ns': int(np.min(self.flux_data)),
                'max_interval_ns': int(np.max(self.flux_data)),
                'total_measured_time_ns': float(total_time_ns),
                'measured_rev_time_ns': float(rev_time_ns),
                'num_revolutions': len(self.revolutions),
                'inferred_rpm': 60000000000 / rev_time_ns if rev_time_ns > 0 else 0,
            }
        else:
            self.stats = {}
        
        # Extract version from header text if possible
        header_text = data[:flux_offset].decode('ascii', errors='ignore')
        version = header_text.split('version=')[1].split(',')[0] if 'version=' in header_text else 'unknown'
        flags = 0  # Not parsed
        
        return {
            'fluxes': self.flux_data,
            'index_positions': self.index_positions,
            'revolutions': self.revolutions,
            'stats': self.stats,
            'version': version,
            'flags': flags,
            'header_text': header_text
        }

    def analyze(self):
        """
        Analyze flux data for noise, anomalies, weak bits.
        
        Returns:
            dict: Analysis results including noise profile, anomalies
        """
        if self.flux_data is None:
            raise ValueError("Parse a file first with parse()")
        
        if not self.stats:
            return {'anomalies': {}, 'noise_profile': {}, 'density_estimate_bits_per_rev': 0}
        
        mean = self.stats.get('mean_interval_ns', 0)
        std = self.stats.get('std_interval_ns', 0)
        
        # Detect anomalies: intervals >3*std (potential dropouts), <0.5*mean (short cells)
        # For blanks, expect ~2000-4000ns cells; adjust thresholds
        base_cell = 4000  # Typical MFM cell ns
        anomalies = {
            'short_cells': np.where(self.flux_data < 0.5 * mean)[0] if mean > 0 else [],
            'long_intervals': np.where(self.flux_data > mean + 3 * std)[0] if std > 0 else [],
            'weak_bit_candidates': np.where((self.flux_data > base_cell * 1.5) & (self.flux_data < base_cell * 2))[0]  # Potential weak (variable length)
        }
        
        # Simple noise profile: Variance per revolution
        rev_variances = [np.var(rev) for rev in self.revolutions if len(rev) > 0]
        rev_times = [np.sum(rev) for rev in self.revolutions if len(rev) > 0]
        rpm_stability = np.std(rev_times) / np.mean(rev_times) if rev_times else 0  # Relative std of rev times
        noise_profile = {
            'avg_variance': np.mean(rev_variances) if rev_variances else 0,
            'high_noise_revs': [i for i, v in enumerate(rev_variances) if v > np.mean(rev_variances) + std] if rev_variances else [],
            'rpm_stability': rpm_stability
        }
        
        # For weak bits: Compare revolutions (inconsistent timings indicate weak)
        if len(self.revolutions) > 1 and std > 0:
            inconsistencies = []
            for i in range(1, len(self.revolutions)):
                # Simple diff on aligned fluxes (assume same length for demo)
                if len(self.revolutions[0]) == len(self.revolutions[i]):
                    diff = np.abs(self.revolutions[0] - self.revolutions[i])
                    inconsistencies.append(np.where(diff > 2 * std)[0])
            anomalies['weak_bit_candidates'] = inconsistencies
        
        return {
            'anomalies': anomalies,
            'noise_profile': noise_profile,
            'density_estimate_bits_per_rev': int(8 * len(self.flux_data) / len(self.revolutions)) if self.revolutions else 0  # Rough: bits from transitions
        }

    def visualize(self, base_path, plot_type='intervals'):
        """
        Visualize flux data.
        
        Args:
            base_path (str): Base path for saving plots (appends type.png)
            plot_type (str): 'intervals' for time series, 'histogram' for distribution,
                             'heatmap' for multi-rev density
        """
        if self.flux_data is None:
            raise ValueError("Parse a file first with parse()")
        
        plt.figure(figsize=(12, 8))
        
        if plot_type == 'intervals':
            mean_int = self.stats.get('mean_interval_ns', 0)
            std_int = self.stats.get('std_interval_ns', 0)
            if len(self.flux_data) > 0:
                plt.plot(self.flux_data, label='Flux Intervals (ns)')
            plt.axhline(mean_int, color='r', linestyle='--', label='Mean')
            plt.axhline(mean_int + 3 * std_int, color='orange', linestyle='--', label='Anomaly Threshold')
            plt.xlabel('Flux Position')
            plt.ylabel('Interval Length (ns)')
            plt.title('Flux Transition Intervals')
            plt.legend()
            output_path = base_path + '_intervals.png'
        
        elif plot_type == 'histogram':
            mean_int = self.stats.get('mean_interval_ns', 0)
            if len(self.flux_data) > 0:
                plt.hist(self.flux_data, bins=50, alpha=0.7, label='Intervals')
            plt.axvline(mean_int, color='r', linestyle='--', label='Mean')
            plt.xlabel('Interval Length (ns)')
            plt.ylabel('Frequency')
            plt.title('Distribution of Flux Intervals')
            plt.legend()
            output_path = base_path + '_hist.png'
        
        elif plot_type == 'heatmap' and len(self.revolutions) > 1:
            # Heatmap of variances across revolutions
            rev_means = [np.mean(rev) for rev in self.revolutions]
            plt.imshow([rev_means], cmap='hot', aspect='auto')
            plt.colorbar(label='Mean Interval (ns)')
            plt.xlabel('Revolution')
            plt.ylabel('Track Position')
            plt.title('Revolution Consistency Heatmap')
            output_path = base_path + '_heatmap.png'
        else:
            output_path = base_path + f'_{plot_type}.png'
        
        plt.savefig(output_path)
        plt.close()
        print(f"Visualization saved to {output_path}")

# Example usage (for testing)
if __name__ == "__main__":
    analyzer = FluxAnalyzer()
    parsed = analyzer.parse("../../example_stream_data/unknown-stream00.0.raw")
    print("Parsed Stats:", parsed['stats'])
    analysis = analyzer.analyze()
    print("Analysis:", analysis)
    analyzer.visualize("test_flux_plot.png", "intervals")