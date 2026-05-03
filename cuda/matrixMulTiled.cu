#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void matrixMulTiledKernel(float* M, float* N, float* P, int Width, int TileSize) {
    extern __shared__ float sharedMem[];  // Dynamic shared memory
    float* Mds = sharedMem;
    float* Nds = &sharedMem[TileSize * TileSize];

    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;

    int Row = by * TileSize + ty;
    int Col = bx * TileSize + tx;

    float Pvalue = 0.0f;

    for (int ph = 0; ph < Width / TileSize; ++ph) {
        // Collaborative load into shared memory
        Mds[ty * TileSize + tx] = M[Row * Width + ph * TileSize + tx];
        Nds[ty * TileSize + tx] = N[(ph * TileSize + ty) * Width + Col];
        __syncthreads();

        for (int k = 0; k < TileSize; ++k)
            Pvalue += Mds[ty * TileSize + k] * Nds[k * TileSize + tx];

        __syncthreads();
    }

    P[Row * Width + Col] = Pvalue;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        printf("Usage: %s <Matrix Width> <Tile Size>\n", argv[0]);
        return 1;
    }

    int Width = atoi(argv[1]);
    int TileSize = atoi(argv[2]);

    if (Width % TileSize != 0) {
        printf("Matrix size must be divisible by tile size.\n");
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

    dim3 dimBlock(TileSize, TileSize);
    dim3 dimGrid(Width / TileSize, Width / TileSize);

    size_t sharedMemSize = 2 * TileSize * TileSize * sizeof(float); // Mds + Nds

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matrixMulTiledKernel<<<dimGrid, dimBlock, sharedMemSize>>>(d_M, d_N, d_P, Width, TileSize);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaDeviceSynchronize();

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
