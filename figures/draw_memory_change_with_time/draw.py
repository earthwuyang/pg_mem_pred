import re
import matplotlib.pyplot as plt
import datetime
from collections import defaultdict

# Read the log file
log_file_path = 'be.INFO.log.20240914-025910'  # Replace with your log file path
with open(log_file_path) as f:
    log_data = f.read()

# Dictionary to store memory usage for each queryid
query_memory_data = defaultdict(list)
from tqdm import tqdm
# Parse the log file to extract queryid, timestamp, and memory usage
for line in tqdm(log_data.splitlines()):
    match = re.search(r'I(\d{8} \d{2}:\d{2}:\d{2}\.\d+).*queryid: (\S+), .*current used memory bytes: (\d+)', line)
    if match:
        timestamp_str = match.group(1)
        query_id = match.group(2)
        memory_bytes = int(match.group(3))
        memory_kb = memory_bytes / 1024  # Convert bytes to KB

        timestamp = datetime.datetime.strptime(timestamp_str, "%Y%m%d %H:%M:%S.%f")
        query_memory_data[query_id].append((timestamp, memory_kb))
import os
if not os.path.exists('figures'):
    os.makedirs('figures')

# Generate a plot for each queryid
from tqdm import tqdm
for query_id, data in tqdm(query_memory_data.items()):
    # Sort data by timestamp
    data.sort(key=lambda x: x[0])
    timestamps, memory_usage_kb = zip(*data)

    # Plot memory usage for the current queryid
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, memory_usage_kb, marker="o", linestyle="-")
    plt.xlabel("Time")
    plt.ylabel("Memory Usage (KB)")
    plt.title(f"Memory Usage Over Time for Query ID: {query_id}")
    plt.grid()
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the figure as a PNG file
    plt.savefig(f"figures/memory_usage_{query_id}.png")

