#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include "global.h"
#include "matrix.h"
#include "gpu_timer.h"
#include "error_check.h"
#include "matrix_gpu.h"

Matrix a, b, res, tmp;
DTYPE *d_a, *d_b, *d_tmp;

int gpu_check(const char* method, void (*kernel)(const DTYPE*, const DTYPE*, DTYPE*),
              int loop = 7)
{
    GpuTimer timer;
    std::vector<int> time_rec;

    for (int i = 1; i <= loop; i++)
    {
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid(MATRIX_H / BLOCK_SIZE, MATRIX_W / BLOCK_SIZE);

        CHECK(cudaMemset(d_tmp, 0, MATRIX_H * MATRIX_W * sizeof(DTYPE)));

        timer.Start();
        kernel<<<grid, block>>>(d_a, d_b, d_tmp);
        CHECK(cudaDeviceSynchronize());
        timer.Stop();

        CHECK(cudaMemcpy(tmp.a, d_tmp, MATRIX_H * MATRIX_W * sizeof(DTYPE), cudaMemcpyDeviceToHost));

        int time_elapsed = static_cast<int>(timer.Elapsed() * 1000);
        printf("%s: %d us\n", method, time_elapsed);

        if (!check_mat(res, tmp))
        {
            printf("Not correct result! method: %s\n", method);
            throw "wrong result";
        }
        // for (int i = 0; i < 5 ; i++)
        //     for (int j = 0; j < 5; j++)
        //         printf("(%d, %d) %f, %f\n", i, j, res[i][j], tmp[i][j]);

        time_rec.emplace_back(time_elapsed);
    }

    std::sort(time_rec.begin(), time_rec.end());
    return time_rec[(time_rec.size() - 1) >> 1];
}

void Init()
{
    srand(time(NULL));
    a.generate();
    b.generate();
    Multiply(a, b, res);

    CHECK(cudaMalloc((void **)&d_a, MATRIX_H * MATRIX_W * sizeof(DTYPE)));
    CHECK(cudaMalloc((void **)&d_b, MATRIX_H * MATRIX_W * sizeof(DTYPE)));
    CHECK(cudaMalloc((void **)&d_tmp, MATRIX_H * MATRIX_W * sizeof(DTYPE)));
    CHECK(cudaMemcpy(d_a, a.a, MATRIX_H * MATRIX_W * sizeof(DTYPE), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, b.a, MATRIX_H * MATRIX_W * sizeof(DTYPE), cudaMemcpyHostToDevice));
}

void Return()
{
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_tmp);
}

int main()
{
    Init();

    printf("GPU version:\n");
    printf("average: %d us\n\n", gpu_check("Naive", multiply_naive));
    printf("average: %d us\n\n", gpu_check("Block", multiply_block));
    printf("average: %d us\n\n", gpu_check("Shared memory", multiply_shared));
    printf("average: %d us\n\n", gpu_check("Shared memory coalesce", multiply_shared_coalesce));
    printf("average: %d us\n\n", gpu_check("less conflict", multiply_less_conflict));
    printf("average: %d us\n\n", gpu_check("Unroll", multiply_unroll));

#ifdef USE_FLOAT
    printf("average: %d us\n\n", gpu_check("cublas", multiply_cublas));
#endif

    Return();
    return 0;
}