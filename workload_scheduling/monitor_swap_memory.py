import logging
import psutil
import time
import os

def get_postgres_swap_usage():

    swap_usage = {}
    total_swap_usage = 0

    # For each PID, get the swap memory from /proc/[PID]/status
    for pid in os.listdir('/proc'):
        if not pid.isdigit():
            continue
        try:
            status_file = f'/proc/{pid}/status'
            with open(status_file, 'r') as f:
                for line in f:
                    if 'VmSwap' in line:
                        swap_value = line.split()[1]  # Get the swap value (in kB)
                        swap_usage[pid] = int(swap_value)  # Store it in the dictionary
                        total_swap_usage += int(swap_value)
        except FileNotFoundError:
            print(f"Process {pid} not found.")
        except Exception as e:
            print(f"Error reading {status_file}: {e}")

    return swap_usage, total_swap_usage

while True:
    total_swap_memory = get_postgres_swap_usage()[1]
    print(f"Total swap memory: {total_swap_memory / (1024 ** 2):.2f}")
    time.sleep(0.5)