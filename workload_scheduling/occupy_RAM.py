import numpy as np
import ctypes
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gb', type=float, default=11, help='Memory to allocate in GB')
args = parser.parse_args()

# Allocate 10GB in RAM (float32 takes 4 bytes per element)
mem_holder = np.zeros(int(args.gb * 1024**3 / 4), dtype=np.float32)

# **Force memory to be allocated in RAM**
mem_holder += 1  # ✅ This forces pages to be mapped into physical RAM

# Lock memory to prevent swapping
libc = ctypes.CDLL("libc.so.6")
addr = mem_holder.ctypes.data
size = mem_holder.nbytes
libc.mlock(addr, size)

# Keep process running without using CPU
print("Memory allocated and locked. Running infinite loop...")
while True:
    time.sleep(10)  # Just keeps process alive
