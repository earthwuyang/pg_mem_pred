import matplotlib.pyplot as plt
import os
# Plot the memory metrics
def plot_memory_metrics(metrics, result_dir, num_queries):
    plt.figure(figsize=(10, 6))
    
    # Convert time to relative
    start_time = min(
        metrics['naive']['time'][0] if metrics['naive']['time'] else float('inf'),
        metrics['memory_based']['time'][0] if metrics['memory_based']['time'] else float('inf')
    )
    naive_time = [t - start_time for t in metrics['naive']['time']]
    memory_based_time = [t - start_time for t in metrics['memory_based']['time']]
    
    # Plot naive
    if naive_time:
        plt.plot(naive_time, metrics['naive']['swap_mem'], label="Default Swap Memory (KB)", linestyle='--')
        plt.plot(naive_time, metrics['naive']['total_mem'], label="Default Total Memory (KB)")

    # Plot memory-based
    if memory_based_time:
        plt.plot(memory_based_time, metrics['memory_based']['swap_mem'], label="FFD Swap Memory (KB)", linestyle='--')
        plt.plot(memory_based_time, metrics['memory_based']['total_mem'], label="FFD Total Memory (KB)")
    
    plt.xlabel("Time (seconds)", fontsize=16)
    plt.ylabel("Memory (KB)", fontsize=16)
    # plt.title("Swap and Total Memory Usage During Execution", fontsize=16)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    # adjust x and y axis ticks font size
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.savefig(os.path.join(result_dir,f'{num_queries}_queries_memory_usage.png'))

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--num_queries', type=int, help='number of queries', required=True)
args = parser.parse_args()

result_dir = './'
with open(f'{args.num_queries}_metrics.pkl', 'rb') as f:
    import pickle
    metrics = pickle.load(f)
plot_memory_metrics(metrics, result_dir, args.num_queries)