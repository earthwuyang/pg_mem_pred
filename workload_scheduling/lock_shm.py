import ctypes
import mmap
import os

file_path = "/dev/shm/mem_holder/ramfile"

# Open the file and lock it in memory
with open(file_path, "r+b") as f:
    mm = mmap.mmap(f.fileno(), 0)  # Memory-map the file

    libc = ctypes.CDLL("libc.so.6", use_errno=True)  # Enable errno tracking
    libc.mlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    libc.mlock.restype = ctypes.c_int  # Return type is int

    # Get the pointer to the mmap object
    addr = ctypes.addressof(ctypes.c_char.from_buffer(mm))
    size = ctypes.c_size_t(len(mm))

    # Call mlock
    result = libc.mlock(ctypes.c_void_p(addr), size)

    if result == 0:
        print("✅ Memory successfully locked in RAM. Prevented from swapping.")
    else:
        error_code = ctypes.get_errno()  # Correct way to retrieve error
        print(f"❌ mlock failed with error {error_code}: {os.strerror(error_code)}")

# Keep process running to maintain the lock
while True:
    pass
