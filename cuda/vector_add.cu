#include <stdio.h>
#include <cuda.h>
#include <windows.h>

#define N 10240000

__global__ void vector_add(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        c[idx] = a[idx] + b[idx];
}

int main() {
    int size = N * sizeof(float);
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;

    h_a = (float *)malloc(size);
    h_b = (float *)malloc(size);
    h_c = (float *)malloc(size);

    for (int i = 0; i < N; i++) {
        h_a[i] = i * 1.0f;
        h_b[i] = (N - i) * 1.0f;
    }

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vector_add<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("CUDA error: %s\n", cudaGetErrorString(err));

    Sleep(5000);  // to see GPU memory usage in Task Manager

    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 5; i++) {
        printf("%d: %f + %f = %f\n", i, h_a[i], h_b[i], h_c[i]);
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
