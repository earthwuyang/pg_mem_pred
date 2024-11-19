import pandas as pd



import matplotlib.pyplot as plt
import numpy as np
# Extract relevant columns for plotting
x = np.arange(100,501, 100)
y_naive = [57.94, 155.84, 219.97, 489.03, 590.13]
y_mem_based = [51.31, 131.25, 215.12, 259.35, 427.9]

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(x, y_naive, label='Naive', color='orange', linewidth=2)
plt.plot(x, y_mem_based, label='Mem-Based', color='red', linewidth=2)
plt.xlabel('Number of Queries')
plt.ylabel('Time (seconds)')
plt.title('Performance Comparison: Naive vs. Mem-Based')
plt.legend()
plt.grid(True)
plt.savefig('pg_sequential_execution.png')
