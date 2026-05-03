#include <stdio.h>
#include <stdlib.h>

__global__ void MatrixMulKernel(float* M, float* N, float* P, int Width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if ((row < Width) && (col < Width)) {
        float Pvalue = 0;
        for (int k = 0; k < Width; ++k) {
            Pvalue += M[row * Width + k] * N[k * Width + col];
        }
        P[row * Width + col] = Pvalue;
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        printf("Usage: %s <Matrix Width> <Block Size>\n", argv[0]);
        return 1;
    }

    int Width = atoi(argv[1]);
    int BlockSize = atoi(argv[2]);

    int size = Width * Width * sizeof(float);

    // Allocate host memory
    float *h_M = (float*)malloc(size);
    float *h_N = (float*)malloc(size);
    float *h_P = (float*)malloc(size);

    // Initialize input matrices
    for (int i = 0; i < Width * Width; ++i) {
        h_M[i] = 1.0f;  // or random values
        h_N[i] = 1.0f;
    }

    // Allocate device memory
    float *d_M, *d_N, *d_P;
    cudaMalloc((void**)&d_M, size);
    cudaMalloc((void**)&d_N, size);
    cudaMalloc((void**)&d_P, size);

    cudaMemcpy(d_M, h_M, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, size, cudaMemcpyHostToDevice);

    // Setup execution configuration
    dim3 dimBlock(BlockSize, BlockSize);
    int blocksPerGridX = (Width + BlockSize - 1) / BlockSize;
    int blocksPerGridY = (Width + BlockSize - 1) / BlockSize;
    dim3 dimGrid(blocksPerGridX, blocksPerGridY);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    // Launch kernel
    MatrixMulKernel<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, Width);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaDeviceSynchronize();

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("KERNEL_TIME: %.3f ms\n", milliseconds);

    // Immediately check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }

    // Optional but useful for debugging and correctness
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_P, d_P, size, cudaMemcpyDeviceToHost);

    // Print first few results (optional)
    printf("Sample output:\n");
    for (int i = 0; i < (Width > 4 ? 4 : Width); ++i) {
        for (int j = 0; j < (Width > 4 ? 4 : Width); ++j) {
            printf("%0.1f ", h_P[i * Width + j]);
        }
        printf("\n");
    }

    // Cleanup
    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);
    free(h_M);
    free(h_N);
    free(h_P);

    return 0;
}
