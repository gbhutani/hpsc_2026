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

void printDeviceProperties() {
    int device;
    cudaGetDevice(&device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    printf("==== CUDA Device Properties ====\n");
    printf("Device Name: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Clock Rate: %.2f GHz\n", prop.clockRate / 1e6f);
    printf("Total Global Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024 * 1024));
    printf("Shared Memory per Block: %.2f KB\n", prop.sharedMemPerBlock / 1024.0);
    printf("Registers per Block: %d\n", prop.regsPerBlock);
    printf("Warp Size: %d\n", prop.warpSize);
    printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("Max Threads Dim: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("Max Grid Size: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("Multiprocessor Count: %d\n", prop.multiProcessorCount);
    printf("Memory Bus Width: %d bits\n", prop.memoryBusWidth);
    printf("Memory Clock Rate: %.2f GHz\n", prop.memoryClockRate / 1e6f);
    printf("L2 Cache Size: %d KB\n", prop.l2CacheSize / 1024);
    printf("=================================\n");
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        printf("Usage: %s <Matrix Width> <Block Size>\n", argv[0]);
        return 1;
    }

    int Width = atoi(argv[1]);
    int BlockSize = atoi(argv[2]);

    printDeviceProperties();

    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    // Safety check
    if (BlockSize * BlockSize > prop.maxThreadsPerBlock) {
        printf("Error: BlockSize^2 (%d) exceeds maxThreadsPerBlock (%d)\n",
               BlockSize * BlockSize, prop.maxThreadsPerBlock);
        return 1;
    }

    int size = Width * Width * sizeof(float);

    // Host memory
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

    dim3 dimBlock(BlockSize, BlockSize);
    dim3 dimGrid((Width + BlockSize - 1) / BlockSize, (Width + BlockSize - 1) / BlockSize);

    MatrixMulKernel<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, Width);
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
