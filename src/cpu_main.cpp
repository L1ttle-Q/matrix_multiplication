#include <cstdlib>
#include <cstdio>
#include <vector>
#include <algorithm>
#include "timer.h"
#include "matrix.h"
#include "matrix_cpu.h"

Matrix a, b, res, tmp;

int cpu_check(const char* method, void (*func)(const Matrix&, const Matrix&, Matrix&),
              int loop = 7)
{
    CpuTimer timer;
    std::vector<int> time_rec;

    for (int i = 1; i <= loop; i++)
    {
        tmp.init();
        timer.Start();
        func(a, b, tmp);
        timer.Stop();

        int time_elapsed = static_cast<int>(timer.Elapsed());
        printf("%s: %d ms\n", method, time_elapsed);

        if (!check_mat(res, tmp))
        {
            printf("Not correct result! method: %s\n", method);
            throw "wrong result";
        }
        time_rec.emplace_back(time_elapsed);
    }

    std::sort(time_rec.begin(), time_rec.end());
    return time_rec[(time_rec.size() - 1) >> 1];
}

int main()
{
    srand(time(NULL));
    a.generate();
    b.generate();
    res.init();
    Multiply(a, b, res);

    printf("CPU version:\n");
    printf("average: %d ms\n\n", cpu_check("Naive", multiply_naive));
    printf("average: %d ms\n\n", cpu_check("Reordered", multiply_reordered));
    printf("average: %d ms\n\n", cpu_check("Unroll", multiply_unroll));
    printf("average: %d ms\n\n", cpu_check("Unroll + ptr", multiply_unroll_ptr));
    printf("average: %d ms\n\n", cpu_check("Unroll 8", multiply_unroll2));
    printf("average: %d ms\n\n", cpu_check("Subblock", multiply_block));

    printf("average: %d ms\n\n", cpu_check("Naive(openmp)", multiply_naive_omp));
    printf("average: %d ms\n\n", cpu_check("Unroll(openmp)", multiply_unroll_omp));
    printf("average: %d ms\n\n", cpu_check("Unroll_ptr(openmp)", multiply_unroll_ptr_omp));
    printf("average: %d ms\n\n", cpu_check("Subblock(openmp)", multiply_block_omp));

    return 0;
}