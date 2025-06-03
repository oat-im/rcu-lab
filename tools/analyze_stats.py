#!/usr/bin/env python3
"""
RCU-Lab Statistical Analyzer
Compares multiple benchmark runs and identifies performance regressions
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import seaborn as sns

def load_and_analyze(csv_file):
    """Load CSV and compute key statistics"""
    df = pd.read_csv(csv_file)
    
    # Group by core and NUMA node
    grouped = df.groupby(['core_id', 'numa_node'])['read_latency_ns'].agg([
        'count',
        'mean',
        'median',
        lambda x: x.quantile(0.99),
        lambda x: x.quantile(0.999),
        lambda x: x.quantile(0.9999)
    ])
    grouped.columns = ['count', 'mean', 'median', 'p99', 'p999', 'p9999']
    
    # Overall statistics
    overall = {
        'file': Path(csv_file).name,
        'total_samples': len(df),
        'median': df['read_latency_ns'].median(),
        'mean': df['read_latency_ns'].mean(),
        'std': df['read_latency_ns'].std(),
        'p99': df['read_latency_ns'].quantile(0.99),
        'p999': df['read_latency_ns'].quantile(0.999),
        'p9999': df['read_latency_ns'].quantile(0.9999),
        'min': df['read_latency_ns'].min(),
        'max': df['read_latency_ns'].max()
    }
    
    # NUMA statistics if available
    if 'is_remote' in df.columns:
        local_latencies = df[df['is_remote'] == 0]['read_latency_ns']
        remote_latencies = df[df['is_remote'] == 1]['read_latency_ns']
        
        if len(remote_latencies) > 0:
            overall['numa_local_median'] = local_latencies.median()
            overall['numa_remote_median'] = remote_latencies.median()
            overall['numa_penalty'] = remote_latencies.median() / local_latencies.median()
    
    return df, grouped, overall

def compare_distributions(files):
    """Compare latency distributions across multiple files"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(files)))
    
    all_stats = []
    
    for i, csv_file in enumerate(files):
        df, grouped, overall = load_and_analyze(csv_file)
        all_stats.append(overall)
        
        label = Path(csv_file).stem
        color = colors[i]
        
        # 1. CDF comparison
        sorted_data = np.sort(df['read_latency_ns'])
        yvals = np.arange(len(sorted_data)) / float(len(sorted_data) - 1)
        axes[0].plot(sorted_data, yvals, label=label, color=color, linewidth=2)
        
        # 2. Violin plot data collection
        sample = df['read_latency_ns'].sample(min(10000, len(df)))
        axes[1].violinplot([sample], positions=[i], showmeans=True, widths=0.7)
        
    # Configure CDF plot
    axes[0].set_xlabel('Read Latency (ns)')
    axes[0].set_ylabel('Cumulative Probability')
    axes[0].set_xscale('log')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].set_title('Cumulative Distribution Comparison')
    
    # Configure violin plot
    axes[1].set_xlabel('Benchmark')
    axes[1].set_ylabel('Read Latency (ns)')
    axes[1].set_yscale('log')
    axes[1].set_xticks(range(len(files)))
    axes[1].set_xticklabels([Path(f).stem for f in files], rotation=45, ha='right')
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_title('Latency Distribution Overview')
    
    # 3. Percentile comparison
    percentiles = ['median', 'p99', 'p999', 'p9999']
    x = np.arange(len(percentiles))
    width = 0.8 / len(files)
    
    for i, stats in enumerate(all_stats):
        values = [stats[p] for p in percentiles]
        axes[2].bar(x + i * width, values, width, label=stats['file'], color=colors[i])
    
    axes[2].set_xlabel('Percentile')
    axes[2].set_ylabel('Latency (ns)')
    axes[2].set_yscale('log')
    axes[2].set_xticks(x + width * (len(files) - 1) / 2)
    axes[2].set_xticklabels(['P50', 'P99', 'P99.9', 'P99.99'])
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, axis='y')
    axes[2].set_title('Percentile Comparison')
    
    # 4. Statistics table
    axes[3].axis('tight')
    axes[3].axis('off')
    
    # Create table data
    table_data = []
    headers = ['File', 'Median', 'P99', 'P99.9', 'P99.99', 'Max']
    
    for stats in all_stats:
        row = [
            Path(stats['file']).stem[:20],  # Truncate long names
            f"{stats['median']:.0f}",
            f"{stats['p99']:.0f}",
            f"{stats['p999']:.0f}",
            f"{stats['p9999']:.0f}",
            f"{stats['max']:.0f}"
        ]
        table_data.append(row)
    
    table = axes[3].table(cellText=table_data, colLabels=headers, 
                         cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    axes[3].set_title('Statistical Summary (nanoseconds)', pad=20)
    
    plt.tight_layout()
    plt.savefig('latency_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved comparison to latency_comparison.png")
    plt.show()
    
    return all_stats

def detect_regressions(baseline_file, test_file, threshold=1.1):
    """Detect performance regressions between baseline and test"""
    
    print(f"\n=== Regression Analysis ===")
    print(f"Baseline: {baseline_file}")
    print(f"Test: {test_file}")
    print(f"Threshold: {threshold}x")
    print("-" * 50)
    
    df_base, _, stats_base = load_and_analyze(baseline_file)
    df_test, _, stats_test = load_and_analyze(test_file)
    
    # Compare key percentiles
    metrics = ['median', 'p99', 'p999', 'p9999']
    
    regressions = []
    for metric in metrics:
        base_val = stats_base[metric]
        test_val = stats_test[metric]
        ratio = test_val / base_val
        
        status = "✓ OK" if ratio < threshold else "✗ REGRESSION"
        
        print(f"{metric.upper():8} | Base: {base_val:7.0f}ns | Test: {test_val:7.0f}ns | "
              f"Ratio: {ratio:5.2f}x | {status}")
        
        if ratio >= threshold:
            regressions.append({
                'metric': metric,
                'baseline': base_val,
                'test': test_val,
                'ratio': ratio
            })
    
    # Statistical significance test
    print("\n--- Statistical Significance ---")
    
    # Sample for Mann-Whitney U test (non-parametric)
    sample_size = min(10000, len(df_base), len(df_test))
    base_sample = df_base['read_latency_ns'].sample(sample_size)
    test_sample = df_test['read_latency_ns'].sample(sample_size)
    
    statistic, pvalue = stats.mannwhitneyu(base_sample, test_sample, alternative='two-sided')
    
    print(f"Mann-Whitney U test p-value: {pvalue:.6f}")
    if pvalue < 0.05:
        print("✗ Statistically significant difference detected")
    else:
        print("✓ No statistically significant difference")
    
    return regressions

def analyze_tail_latency_causes(csv_file):
    """Analyze what causes tail latencies"""
    
    df = pd.read_csv(csv_file)
    
    # Identify tail latency samples (P99+)
    p99_threshold = df['read_latency_ns'].quantile(0.99)
    tail_samples = df[df['read_latency_ns'] > p99_threshold]
    
    print(f"\n=== Tail Latency Analysis for {Path(csv_file).name} ===")
    print(f"Total samples: {len(df):,}")
    print(f"Tail samples (>P99): {len(tail_samples):,}")
    
    # Analyze by core
    print("\n--- Tail Latencies by Core ---")
    core_tails = tail_samples.groupby('core_id').size().sort_values(ascending=False).head(10)
    print(core_tails)
    
    # Analyze by NUMA node
    if 'numa_node' in df.columns:
        print("\n--- Tail Latencies by NUMA Node ---")
        numa_tails = tail_samples.groupby('numa_node').agg({
            'read_latency_ns': ['count', 'mean', 'max']
        })
        print(numa_tails)
    
    # Check for remote NUMA correlation
    if 'is_remote' in df.columns:
        print("\n--- NUMA Locality Impact ---")
        remote_pct_all = df['is_remote'].mean() * 100
        remote_pct_tail = tail_samples['is_remote'].mean() * 100
        
        print(f"Remote accesses in all samples: {remote_pct_all:.1f}%")
        print(f"Remote accesses in tail samples: {remote_pct_tail:.1f}%")
        
        if remote_pct_tail > remote_pct_all * 1.5:
            print("✗ Remote NUMA access is a significant contributor to tail latency")
        else:
            print("✓ Remote NUMA access is not the primary cause of tail latency")
    
    # Time series analysis (if we had timestamps)
    # This would show if tail latencies cluster in time (indicating interference)
    
    return tail_samples

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  analyze_stats.py <csv_file>               # Single file analysis")
        print("  analyze_stats.py <csv1> <csv2> ...        # Compare multiple files")
        print("  analyze_stats.py --regression <baseline> <test>  # Regression detection")
        sys.exit(1)
    
    if sys.argv[1] == '--regression':
        if len(sys.argv) != 4:
            print("Usage: analyze_stats.py --regression <baseline.csv> <test.csv>")
            sys.exit(1)
        
        baseline = sys.argv[2]
        test = sys.argv[3]
        
        regressions = detect_regressions(baseline, test)
        
        if regressions:
            print(f"\n✗ Found {len(regressions)} regression(s)")
            sys.exit(1)
        else:
            print("\n✓ No regressions detected")
            sys.exit(0)
    
    else:
        # Single or multiple file analysis
        files = sys.argv[1:]
        
        # Verify all files exist
        for f in files:
            if not Path(f).exists():
                print(f"Error: {f} not found")
                sys.exit(1)
        
        if len(files) == 1:
            # Single file deep analysis
            df, grouped, overall = load_and_analyze(files[0])
            
            print(f"\n=== Analysis of {files[0]} ===")
            print(f"Total samples: {overall['total_samples']:,}")
            print(f"\nLatency Statistics:")
            print(f"  Minimum:    {overall['min']:>8.0f} ns")
            print(f"  Median:     {overall['median']:>8.0f} ns")
            print(f"  Mean:       {overall['mean']:>8.0f} ns")
            print(f"  P99:        {overall['p99']:>8.0f} ns")
            print(f"  P99.9:      {overall['p999']:>8.0f} ns")
            print(f"  P99.99:     {overall['p9999']:>8.0f} ns")
            print(f"  Maximum:    {overall['max']:>8.0f} ns")
            
            if 'numa_penalty' in overall:
                print(f"\nNUMA Analysis:")
                print(f"  Local median:  {overall['numa_local_median']:>8.0f} ns")
                print(f"  Remote median: {overall['numa_remote_median']:>8.0f} ns")
                print(f"  NUMA penalty:  {overall['numa_penalty']:>8.2f}x")
            
            # Analyze tail latencies
            analyze_tail_latency_causes(files[0])
            
        else:
            # Multiple file comparison
            print(f"Comparing {len(files)} benchmark runs...")
            all_stats = compare_distributions(files)
            
            # Find best/worst
            medians = [s['median'] for s in all_stats]
            best_idx = np.argmin(medians)
            worst_idx = np.argmax(medians)
            
            print(f"\nBest median latency: {all_stats[best_idx]['file']} ({medians[best_idx]:.0f}ns)")
            print(f"Worst median latency: {all_stats[worst_idx]['file']} ({medians[worst_idx]:.0f}ns)")

if __name__ == "__main__":
    main()