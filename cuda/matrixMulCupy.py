import cupy as cp

N = 4096
A = cp.random.rand(N, N, dtype=cp.float32)
B = cp.random.rand(N, N, dtype=cp.float32)

C = cp.dot(A, B)  # GPU matrix multiplication using cuBLAS

print("Result:\n", C)
print("As NumPy array:\n", cp.asnumpy(C))  # if needed on CPU
