import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

# Function to plot performance comparison
def plot_performance(ax):
    x = np.arange(100, 501, 100)
    y_naive = [57.94, 155.84, 219.97, 489.03, 590.13]
    y_mem_based = [51.31, 131.25, 215.12, 259.35, 427.9]

    ax.plot(x, y_naive, label='Naive', color='orange', linewidth=2)
    ax.plot(x, y_mem_based, label='Mem-Based', color='red', linewidth=2)
    ax.set_xlabel('Number of Queries', fontsize=14)
    ax.set_ylabel('Time (seconds)', fontsize=14)
    ax.legend(fontsize=10)  # Set legend font size for left plot
    ax.grid(True)
    ax.set_xticks(np.arange(100, 501, 100))  # Set x-axis ticks every 100
    ax.set_ylim(0, 600)  # Shorten the length by setting y-axis limits

# Function to plot memory usage
def plot_memory_metrics(ax, metrics, num_queries):
    # Convert time to relative
    naive_start_time = metrics['naive']['time'][0]
    mem_based_start_time = metrics['memory_based']['time'][0]

    naive_time = [t - naive_start_time for t in metrics['naive']['time']]
    memory_based_time = [t - mem_based_start_time for t in metrics['memory_based']['time']]

    # Plot naive
    ax.plot(naive_time, metrics['naive']['swap_mem'], label="Naive Swap Memory (KB)", linestyle='--')
    ax.plot(naive_time, metrics['naive']['total_mem'], label="Naive Total Memory (KB)")

    # Plot memory-based
    ax.plot(memory_based_time, metrics['memory_based']['swap_mem'], label="Memory-Based Swap Memory (KB)", linestyle='--')
    ax.plot(memory_based_time, metrics['memory_based']['total_mem'], label="Memory-Based Total Memory (KB)")

    ax.set_xlabel("Time (seconds)", fontsize=14)
    ax.set_ylabel("Memory (KB)", fontsize=14)
    ax.legend(fontsize=10)  # Set legend font size for right plot
    ax.grid(True)

# Load metrics for memory usage
result_dir = './'
num_queries = 500  # Example number of queries
with open(f'naive_metrics_{num_queries}.pkl', 'rb') as f:
    naive_metrics = pickle.load(f)
with open(f"mem_based_metrics_{num_queries}.pkl", 'rb') as f:
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

# Combine the two plots into one figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot the performance comparison on the left
plot_performance(ax1)

# Plot the memory usage on the right
plot_memory_metrics(ax2, metrics, num_queries)

# Add titles below subfigures
fig.text(0.25, 0.02, '(a) Sequential Queries Execution Time: Naive vs. Mem-Based', ha='center', fontsize=14)
fig.text(0.75, 0.02, '(b) Memory Usage: Naive vs. Mem-Based', ha='center', fontsize=14)

# Save the combined figure
combined_fig_path = os.path.join(result_dir, 'combined_performance_memory.png')
plt.tight_layout(rect=[0, 0.05, 1, 1])  # Adjust layout to make space for subfigure titles
plt.savefig(combined_fig_path)
plt.show()
