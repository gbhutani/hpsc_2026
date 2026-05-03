import numpy as np
from numba import cuda
import time

# Set array size
N = 10240000

# CUDA kernel
@cuda.jit
def vector_add(a, b, c, n):
    idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if idx < n:
        c[idx] = a[idx] + b[idx]

def main():
    # Host memory
    h_a = np.arange(N, dtype=np.float32)
    h_b = (N - h_a).astype(np.float32)
    h_c = np.zeros(N, dtype=np.float32)

    # Device memory
    d_a = cuda.to_device(h_a)
    d_b = cuda.to_device(h_b)
    d_c = cuda.device_array_like(h_c)

    # Kernel launch configuration
    threads_per_block = 256
    blocks_per_grid = (N + threads_per_block - 1) // threads_per_block

    # Launch kernel
    vector_add[blocks_per_grid, threads_per_block](d_a, d_b, d_c, N)

    # Optionally wait to see memory in Task Manager (Windows only)
    time.sleep(5)

    # Copy result back to host
    d_c.copy_to_host(h_c)

    # Print first few results
    for i in range(5):
        print(f"{i}: {h_a[i]} + {h_b[i]} = {h_c[i]}")

if __name__ == "__main__":
    main()
