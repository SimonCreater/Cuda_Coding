#include <stdio.h>
#include <cuda.h>

#define N 16  


__global__ __cluster_dims__(2,1,1) 
void cluster_kernel(float* input, float* output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    output[idx * N + idy] = input[idx * N + idy];  /
}

int main() {
    float *input, *output;


    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);


    {
        cudaLaunchConfig_t config = {0};
        config.gridDim = numBlocks;
        config.blockDim = threadsPerBlock;


        cudaLaunchAttribute attribute[1];
        attribute[0].id = cudaLaunchAttributeClusterDimension;
        attribute[0].val.clusterDim.x = 2;  
        attribute[0].val.clusterDim.y = 1;
        attribute[0].val.clusterDim.z = 1;
        
        config.attrs = attribute;
        config.numAttrs = 1;

        // 커널 런칭??
        cudaLaunchKernelEx(&config, (void*)cluster_kernel, input, output);
    }
    return 0;
}
