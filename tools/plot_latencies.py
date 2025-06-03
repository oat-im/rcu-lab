#!/usr/bin/env python3
"""
RCU-Lab Latency Plotter
Visualizes latency distributions from RCU benchmarks
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_latency_distribution(csv_file):
    """Plot latency distribution with percentile markers"""
    
    # Load data
    df = pd.read_csv(csv_file)
    
    # Calculate percentiles
    p50 = df['read_latency_ns'].quantile(0.50)
    p99 = df['read_latency_ns'].quantile(0.99)
    p999 = df['read_latency_ns'].quantile(0.999)
    p9999 = df['read_latency_ns'].quantile(0.9999)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Top plot: Histogram with log scale
    ax1.hist(df['read_latency_ns'], bins=100, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(p50, color='green', linestyle='--', linewidth=2, label=f'P50: {p50:.0f}ns')
    ax1.axvline(p99, color='orange', linestyle='--', linewidth=2, label=f'P99: {p99:.0f}ns')
    ax1.axvline(p999, color='red', linestyle='--', linewidth=2, label=f'P99.9: {p999:.0f}ns')
    ax1.axvline(p9999, color='darkred', linestyle='--', linewidth=2, label=f'P99.99: {p9999:.0f}ns')
    ax1.set_xlabel('Read Latency (ns)')
    ax1.set_ylabel('Count (log scale)')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f'RCU Read Latency Distribution - {Path(csv_file).stem}')
    
    # Bottom plot: CDF
    sorted_latencies = np.sort(df['read_latency_ns'])
    p = np.linspace(0, 100, len(sorted_latencies))
    
    ax2.plot(sorted_latencies, p, linewidth=2)
    ax2.axvline(p50, color='green', linestyle='--', alpha=0.5)
    ax2.axvline(p99, color='orange', linestyle='--', alpha=0.5)
    ax2.axvline(p999, color='red', linestyle='--', alpha=0.5)
    ax2.axvline(p9999, color='darkred', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Read Latency (ns)')
    ax2.set_ylabel('Percentile')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Cumulative Distribution Function (CDF)')
    
    # Add text box with stats
    stats_text = f"""
    Samples: {len(df):,}
    Min: {df['read_latency_ns'].min()}ns
    Max: {df['read_latency_ns'].max()}ns
    Mean: {df['read_latency_ns'].mean():.1f}ns
    Std: {df['read_latency_ns'].std():.1f}ns
    """
    ax1.text(0.02, 0.98, stats_text.strip(), transform=ax1.transAxes,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save and show
    output_file = Path(csv_file).stem + '_distribution.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_file}")
    plt.show()

def plot_numa_comparison(csv_file):
    """Compare local vs remote NUMA access"""
    
    df = pd.read_csv(csv_file)
    
    if 'is_remote' not in df.columns:
        print("No NUMA remote/local data in this file")
        return
    
    local_df = df[df['is_remote'] == 0]
    remote_df = df[df['is_remote'] == 1]
    
    if len(remote_df) == 0:
        print("No remote NUMA accesses in this dataset")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot both distributions
    bins = np.logspace(np.log10(df['read_latency_ns'].min()), 
                      np.log10(df['read_latency_ns'].max()), 50)
    
    ax.hist(local_df['read_latency_ns'], bins=bins, alpha=0.5, 
            label='Local NUMA', color='blue', density=True)
    ax.hist(remote_df['read_latency_ns'], bins=bins, alpha=0.5, 
            label='Remote NUMA', color='red', density=True)
    
    # Add median lines
    local_median = local_df['read_latency_ns'].median()
    remote_median = remote_df['read_latency_ns'].median()
    
    ax.axvline(local_median, color='blue', linestyle='--', 
              label=f'Local median: {local_median:.0f}ns')
    ax.axvline(remote_median, color='red', linestyle='--', 
              label=f'Remote median: {remote_median:.0f}ns')
    
    ax.set_xlabel('Read Latency (ns)')
    ax.set_ylabel('Density')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('NUMA Local vs Remote Access Latency')
    
    # Add penalty factor
    penalty = remote_median / local_median
    ax.text(0.02, 0.98, f'NUMA Penalty: {penalty:.2f}x', 
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    plt.tight_layout()
    output_file = Path(csv_file).stem + '_numa_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved NUMA comparison to {output_file}")
    plt.show()

def main():
    if len(sys.argv) < 2:
        print("Usage: plot_latencies.py <csv_file>")
        print("Example: plot_latencies.py latencies_100000_remote.csv")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    
    if not Path(csv_file).exists():
        print(f"Error: {csv_file} not found")
        sys.exit(1)
    
    print(f"Analyzing {csv_file}...")
    plot_latency_distribution(csv_file)
    plot_numa_comparison(csv_file)

if __name__ == "__main__":
    main()