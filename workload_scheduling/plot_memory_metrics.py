# Plot the memory metrics
def plot_memory_metrics(metrics, result_dir):
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
        plt.plot(naive_time, metrics['naive']['swap_mem'], label="Naive Swap Memory (KB)", linestyle='--')
        plt.plot(naive_time, metrics['naive']['total_mem'], label="Naive Total Memory (KB)")

    # Plot memory-based
    if memory_based_time:
        plt.plot(memory_based_time, metrics['memory_based']['swap_mem'], label="Memory-Based Swap Memory (KB)", linestyle='--')
        plt.plot(memory_based_time, metrics['memory_based']['total_mem'], label="Memory-Based Total Memory (KB)")
    
    plt.xlabel("Time (seconds)")
    plt.ylabel("Memory (KB)")
    plt.title("Swap and Total Memory Usage During Execution")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir,'memory_usage.png'))

import argparse

result_dir = './'
with open(f'{args.num_queries}_metrics.pkl') as f:
    import pickle
    metrics = pickle.load(f)
plot(metrics, result_dir)