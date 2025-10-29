#!/usr/bin/env python3
"""
Analysis script for Multi-Agent Pac-Men simulation results.
Generates figures and metrics for research paper.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from pathlib import Path

def load_data(log_file):
    """Load negotiation log data."""
    return pd.read_csv(log_file)

def plot_conflict_resolution(df, output_dir):
    """Plot conflict resolution rates by strategy."""
    plt.figure(figsize=(10, 6))
    
    strategy_results = df.groupby('Strategy')['Outcome'].value_counts(normalize=True).unstack()
    strategy_results.plot(kind='bar', stacked=True, ax=plt.gca())
    
    plt.title('Conflict Resolution Outcomes by Negotiation Strategy')
    plt.xlabel('Negotiation Strategy')
    plt.ylabel('Proportion of Conflicts')
    plt.legend(title='Outcome')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'conflict_resolution.png'), dpi=300)
    plt.close()

def plot_negotiation_rounds(df, output_dir):
    """Plot distribution of negotiation rounds."""
    plt.figure(figsize=(10, 6))
    
    for strategy in df['Strategy'].unique():
        strategy_data = df[df['Strategy'] == strategy]
        plt.hist(strategy_data['Rounds'], alpha=0.7, label=strategy, bins=range(1, 8))
    
    plt.title('Distribution of Negotiation Rounds by Strategy')
    plt.xlabel('Number of Rounds')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'negotiation_rounds.png'), dpi=300)
    plt.close()

def generate_summary_statistics(df, output_dir):
    """Generate summary statistics table."""
    stats = df.groupby('Strategy').agg({
        'Outcome': lambda x: (x == 'SUCCESS').mean(),
        'Rounds': 'mean',
        'LoserWait': 'mean'
    }).round(3)
    
    stats.columns = ['Success_Rate', 'Avg_Rounds', 'Avg_Loser_Wait']
    
    with open(os.path.join(output_dir, 'summary_statistics.txt'), 'w') as f:
        f.write("Summary Statistics\n")
        f.write("==================\n\n")
        f.write(stats.to_string())
    
    return stats

def main():
    parser = argparse.ArgumentParser(description='Analyze simulation results')
    parser.add_argument('--input', type=str, default='data/negotiation_log.csv',
                       help='Input CSV file with simulation data')
    parser.add_argument('--output', type=str, default='results/figures',
                       help='Output directory for figures')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    # Load and analyze data
    df = load_data(args.input)
    
    print(f"Loaded {len(df)} negotiation records")
    print(f"Strategies: {df['Strategy'].unique()}")
    
    # Generate plots and statistics
    plot_conflict_resolution(df, args.output)
    plot_negotiation_rounds(df, args.output)
    stats = generate_summary_statistics(df, args.output)
    
    print("Analysis complete. Results saved to:", args.output)
    print("\nSummary Statistics:")
    print(stats)

if __name__ == "__main__":
    main()