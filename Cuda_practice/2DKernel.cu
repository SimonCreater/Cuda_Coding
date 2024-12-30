#include <stdio.h>
#include <cuda.h>

__global__ void TwoDKernel(float* devPtr, size_t pitch, int width, int height) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < width && y < height) {
        float* row = (float*)((char*)devPtr + y * pitch);
        float element = row[x];
        row[x] = element * 2.0f;
    }
}

int main() {
    int width = 64, height = 64;
    float* devPtr;
    size_t pitch;

    cudaMallocPitch(&devPtr, &pitch, width * sizeof(float), height);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    TwoDKernel<<<numBlocks, threadsPerBlock>>>(devPtr, pitch, width, height);

    cudaFree(devPtr);

    return 0;
}
