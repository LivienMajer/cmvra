import argparse
import matplotlib.pyplot as plt
import json
import numpy as np

def load_stats(files):
    combined_stats = []
    for file in files:
        with open(file, 'r') as f:
            stats = json.load(f)
            combined_stats.append(stats)
    return combined_stats

def plot_all_in_one(combined_stats, metrics, modalities, save_path):
    """
    Create a single plot comparing multiple metrics and modalities across different runs.

    Args:
        combined_stats (list): List of dictionaries containing statistics for each run.
        metrics (list): List of metric names to plot.
        modalities (list): List of modalities to include in the plot.
        save_path (str): Path to save the resulting plot.
    """
    plt.figure(figsize=(12, 8))
    markers = ['o', 'v', '^', '<', '>', 's', 'p', '*', '+', 'x']
    lines = ['-', '--', '-.', ':']
    
    # Assign a unique color to each metric
    colors = {
        'train_loss': 'r',
        'train_accuracy': 'g',
        'val_loss': 'b',
        'val_accuracy': 'c',
        'train_balanced_accuracy': 'm',
        'val_balanced_accuracy': 'y'
    }

    for metric in metrics:
        color = colors[metric]
        for modality in modalities:
            for run_idx, stats in enumerate(combined_stats):
                if metric in stats and modality in stats[metric]:
                    epochs = np.arange(1, len(stats[metric][modality]) + 1)
                    marker = markers[run_idx % len(markers)]
                    line = lines[(run_idx + len(modalities) * metrics.index(metric)) % len(lines)]
                    plt.plot(epochs, stats[metric][modality],
                             label=f'Run {run_idx + 1} - {metric} - {modality}',
                             marker=marker, linestyle=line, color=color)

    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Comparison of Metrics and Modalities Across Runs')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    """
    This script plots various metrics for different modalities across multiple runs four our specific experimental setup.
    It takes command-line arguments for input files, metrics to plot, modalities to include, and where to save the plot.
    
    Usage:
    python script_name.py --files file1.json file2.json --metrics train_loss val_accuracy --modalities rgb skeleton --save_path output.png
    """
    parser = argparse.ArgumentParser(description="Plot deep learning statistics.")
    parser.add_argument('--files', nargs='+', help='Paths to the files containing training statistics', required=True)
    parser.add_argument('--metrics', nargs='+', help='Metrics to plot', required=True)
    parser.add_argument('--modalities', nargs='+', help='Modalities to include in the plot', required=True)
    parser.add_argument('--save_path', type=str, help='Path to save the plot', default='/home/bas06400/Thesis/VIP/src/plots/all_stats_comparison_plot.png')

    args = parser.parse_args()

    # Validate that all specified metrics are supported and have a designated color
    for metric in args.metrics:
        if metric not in ['train_loss', 'train_accuracy', 'val_loss', 'val_accuracy', 'train_balanced_accuracy', 'val_balanced_accuracy']:
            raise ValueError(f"Unsupported metric: {metric}. Please ensure all metrics are among the predefined ones.")

    combined_stats = load_stats(args.files)
    plot_all_in_one(combined_stats, args.metrics, args.modalities, args.save_path)
