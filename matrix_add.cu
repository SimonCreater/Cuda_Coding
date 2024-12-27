#include <stdio.h>
#include <cuda.h>

#define N 16  


__global__ void MatAdd(float A[N][N], float B[N][N], float C[N][N]) {
    int i = threadIdx.x;
    int j = threadIdx.y;
    
    if (i < N && j < N) {
        C[i][j] = A[i][j] + B[i][j];
    }
}


int main() {
    float A[N][N], B[N][N], C[N][N];
    float (*d_A)[N], (*d_B)[N], (*d_C)[N];


    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = i * 0.1f + j * 0.1f;
            B[i][j] = j * 0.2f + i * 0.2f;
        }
    }


    cudaMalloc((void**)&d_A, N * N * sizeof(float));
    cudaMalloc((void**)&d_B, N * N * sizeof(float));
    cudaMalloc((void**)&d_C, N * N * sizeof(float));


    cudaMemcpy(d_A, A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * N * sizeof(float), cudaMemcpyHostToDevice);


    dim3 threadsPerBlock(N, N);
    int numBlocks = 1;

    MatAdd<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C);

    cudaMemcpy(C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);


    printf("Result matrix C:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%0.1f ", C[i][j]);
        }
        printf("\n");
    }


    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
