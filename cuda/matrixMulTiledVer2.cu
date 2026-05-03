#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16  // Tile size is now fixed at compile-time

__global__ void matrixMulTiledKernel(float* M, float* N, float* P, int Width) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float Pvalue = 0.0f;

    for (int ph = 0; ph < Width / TILE_WIDTH; ++ph) {
        Mds[ty][tx] = M[Row * Width + ph * TILE_WIDTH + tx];
        Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * Width + Col];
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k)
            Pvalue += Mds[ty][k] * Nds[k][tx];

        __syncthreads();
    }

    P[Row * Width + Col] = Pvalue;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("Usage: %s <Matrix Width>\n", argv[0]);
        return 1;
    }

    int Width = atoi(argv[1]);

    if (Width % TILE_WIDTH != 0) {
        printf("Matrix size must be divisible by TILE_WIDTH (%d).\n", TILE_WIDTH);
        return 1;
    }

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

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid(Width / TILE_WIDTH, Width / TILE_WIDTH);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matrixMulTiledKernel<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, Width);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("KERNEL_TIME: %.3f ms\n", milliseconds);

    cudaMemcpy(h_P, d_P, size, cudaMemcpyDeviceToHost);

    printf("Sample output:\n");
    for (int i = 0; i < (Width > 4 ? 4 : Width); ++i) {
        for (int j = 0; j < (Width > 4 ? 4 : Width); ++j) {
            printf("%0.1f ", h_P[i * Width + j]);
        }
        printf("\n");
    }

    cudaFree(d_M); cudaFree(d_N); cudaFree(d_P);
    free(h_M); free(h_N); free(h_P);

    return 0;
}
