#include "matrix_gpu.h"
#include <cublas_v2.h>

__global__ void multiply_naive(const DTYPE* A, const DTYPE* B, DTYPE* C)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    DTYPE sum = 0;
    for (int k = 0; k < MATRIX_W; k++)
        sum += A[tx * MATRIX_W + k] * B[k * MATRIX_W + ty];
    C[tx * MATRIX_W + ty] = sum;
}

__global__ void multiply_block(const DTYPE* A, const DTYPE* B, DTYPE* C)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    DTYPE sum = 0;

    for (int m = 0; m < MATRIX_W; m += BLOCK_SIZE)
        for (int k = 0; k < BLOCK_SIZE; k++)
            sum += A[tx * MATRIX_W + (m + k)] * B[(m + k) * MATRIX_W + ty];

    C[tx * MATRIX_W + ty] = sum;
}

__global__ void multiply_shared(const DTYPE* A, const DTYPE* B, DTYPE* C)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ DTYPE As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ DTYPE Bs[BLOCK_SIZE][BLOCK_SIZE];

    DTYPE sum = 0;

    for (int m = 0; m < MATRIX_W; m += BLOCK_SIZE)
    {
        As[threadIdx.x][threadIdx.y] = A[tx * MATRIX_W + (m + threadIdx.y)];
        Bs[threadIdx.x][threadIdx.y] = B[(m + threadIdx.x) * MATRIX_W + ty];

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; k++)
            sum += As[threadIdx.x][k] * Bs[k][threadIdx.y];

        __syncthreads();
    }

    C[tx * MATRIX_W + ty] = sum;
}

__global__ void multiply_shared_coalesce(const DTYPE* A, const DTYPE* B, DTYPE* C)
{
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    __shared__ DTYPE As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ DTYPE Bs[BLOCK_SIZE][BLOCK_SIZE];

    DTYPE sum = 0;

    for (int m = 0; m < MATRIX_W; m += BLOCK_SIZE)
    {
        As[threadIdx.x][threadIdx.y] = A[row * MATRIX_W + (m + threadIdx.x)];
        Bs[threadIdx.x][threadIdx.y] = B[(m + threadIdx.y) * MATRIX_W + col];
        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; k++)
            sum += As[k][threadIdx.y] * Bs[threadIdx.x][k];

        __syncthreads();
    }

    C[row * MATRIX_W + col] = sum;
}

__global__ void multiply_less_conflict(const DTYPE* A, const DTYPE* B, DTYPE* C)
{
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    __shared__ DTYPE As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ DTYPE Bs[BLOCK_SIZE][BLOCK_SIZE];

    DTYPE sum = 0;

    for (int m = 0; m < MATRIX_W; m += BLOCK_SIZE)
    {
        As[threadIdx.y][threadIdx.x] = A[row * MATRIX_W + (m + threadIdx.x)];
        Bs[threadIdx.y][threadIdx.x] = B[(m + threadIdx.y) * MATRIX_W + col];
        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; k++)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();
    }

    C[row * MATRIX_W + col] = sum;
}

__global__ void multiply_unroll(const DTYPE* A, const DTYPE* B, DTYPE* C)
{
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    __shared__ DTYPE As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ DTYPE Bs[BLOCK_SIZE][BLOCK_SIZE];

    DTYPE sum = 0;

    for (int m = 0; m < MATRIX_W; m += BLOCK_SIZE)
    {
        As[threadIdx.y][threadIdx.x] = A[row * MATRIX_W + (m + threadIdx.x)];
        Bs[threadIdx.y][threadIdx.x] = B[(m + threadIdx.y) * MATRIX_W + col];
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; k++)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();
    }

    C[row * MATRIX_W + col] = sum;
}

#ifdef USE_FLOAT
void multiply_cublas(const DTYPE* A, const DTYPE* B, DTYPE* C)
{
    cublasHandle_t handle;
    cublasCreate(&handle);

    const DTYPE alpha = 1, beta = 0;
    cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        MATRIX_H, MATRIX_W, MATRIX_H,
        &alpha,
        B, MATRIX_H,
        A, MATRIX_W,
        &beta,
        C, MATRIX_H
    );

    cublasDestroy(handle);
}

#endif