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
    # naive_time = [t - start_time for t in metrics['naive']['time']]
    naive_time = [t - metrics['naive']['time'][0] for t in metrics['naive']['time']]
    # memory_based_time = [t - start_time for t in metrics['memory_based']['time']]
    memory_based_time = [t - metrics['memory_based']['time'][0] for t in metrics['memory_based']['time']]
    
    # Plot naive
    if naive_time:
        plt.plot(naive_time, metrics['naive']['swap_mem'], label="Default Swap Memory (KB)", linestyle='--')
        plt.plot(naive_time, metrics['naive']['total_mem'], label="Default Total Memory (KB)")

    # Plot memory-based
    if memory_based_time:
        plt.plot(memory_based_time, metrics['memory_based']['swap_mem'], label="Memory-Based Swap Memory (KB)", linestyle='--')
        plt.plot(memory_based_time, metrics['memory_based']['total_mem'], label="Memory-Based Total Memory (KB)")
    
    plt.xlabel("Time (seconds)", fontsize=14)
    plt.ylabel("Memory (KB)", fontsize=14)
    plt.title("Swap and Total Memory Usage During Execution", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    # adjust x and y axis ticks font size
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.savefig(os.path.join(result_dir,f'{num_queries}_queries_memory_usage.png'))

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--num_queries', type=int, help='number of queries', required=True)
args = parser.parse_args()

result_dir = './'
with open(f'naive_metrics_{args.num_queries}.pkl', 'rb') as f:
    import pickle
    naive_metrics = pickle.load(f)
with open(f"mem_based_metrics_{args.num_queries}.pkl", 'rb') as f:
    mem_based_metrics = pickle.load(f)

metrics = {
    'naive': {
        'time': naive_metrics['naive']['time'],
       'swap_mem': naive_metrics['naive']['swap_mem'],
        'total_mem': naive_metrics['naive']['total_mem']
    },
   'memory_based': {
        'time': mem_based_metrics['memory_based']['time'],
       'swap_mem': mem_based_metrics['memory_based']['swap_mem'],
        'total_mem': mem_based_metrics['memory_based']['total_mem']
    }
}
plot_memory_metrics(metrics, result_dir, args.num_queries)