#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

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
    if (argc != 2) {
        printf("Usage: %s <Matrix Width>\n", argv[0]);
        return 1;
    }

    int Width = atoi(argv[1]);
    int size = Width * Width * sizeof(float);

    float *h_M = (float*)malloc(size);
    float *h_N = (float*)malloc(size);
    float *h_P = (float*)malloc(size);

    for (int i = 0; i < Width * Width; ++i) {
        h_M[i] = 1.0f;
        h_N[i] = 1.0f;
    }

    float *d_M, *d_N, *d_P;
    cudaMalloc((void**)&d_M, size);
    cudaMalloc((void**)&d_N, size);
    cudaMalloc((void**)&d_P, size);

    cudaMemcpy(d_M, h_M, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, size, cudaMemcpyHostToDevice);

    // ==== Auto-tune block size for best occupancy ====
    int minGridSize = 0, bestBlockSize = 0;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &bestBlockSize, MatrixMulKernel, 0, 0);
    printf("Suggested block size for best occupancy: %d threads per block\n", bestBlockSize);

    // Assume square block (close to warp friendly)
    int BlockDim = (int)sqrt((float)bestBlockSize);
    printf("Final Block Size: %d \n", BlockDim);
    dim3 dimBlock(BlockDim, BlockDim);
    int blocksX = (Width + dimBlock.x - 1) / dimBlock.x;
    int blocksY = (Width + dimBlock.y - 1) / dimBlock.y;
    dim3 dimGrid(blocksX, blocksY);

    // ==== Launch kernel ====
    MatrixMulKernel<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, Width);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaDeviceSynchronize();

    cudaMemcpy(h_P, d_P, size, cudaMemcpyDeviceToHost);

    printf("Sample output:\n");
    for (int i = 0; i < (Width > 4 ? 4 : Width); ++i) {
        for (int j = 0; j < (Width > 4 ? 4 : Width); ++j) {
            printf("%0.1f ", h_P[i * Width + j]);
        }
        printf("\n");
    }

    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);
    free(h_M);
    free(h_N);
    free(h_P);

    return 0;
}
