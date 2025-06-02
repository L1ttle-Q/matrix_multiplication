#include <omp.h>
#include "matrix.h"

void multiply_naive(const Matrix& a, const Matrix& b, Matrix& res)
{
    for (int i = 0; i < MATRIX_H; i++)
        for (int j = 0; j < MATRIX_W; j++)
            for (int k = 0; k < MATRIX_W; k++)
                res[i][j] += a[i][k] * b[k][j];
}

void multiply_reordered(const Matrix& a, const Matrix& b, Matrix& res)
{
    for (int i = 0; i < MATRIX_H; i++)
        for (int k = 0; k < MATRIX_W; k++)
        {
            DTYPE t = a[i][k];
            for (int j = 0; j < MATRIX_W; j++)
                res[i][j] += t * b[k][j];
        }
}

void multiply_unroll(const Matrix& a, const Matrix& b, Matrix& res)
{
    for (int i = 0; i < MATRIX_H; i++)
        for (int k = 0; k < MATRIX_W; k++)
        {
            DTYPE t = a[i][k];

            int upper = MATRIX_W;
            for (int j = 0; j < upper;)
            {
                res[i][j] += t * b[k][j]; j++;
                res[i][j] += t * b[k][j]; j++;
                res[i][j] += t * b[k][j]; j++;
                res[i][j] += t * b[k][j]; j++;
            }
        }
}

void multiply_unroll_ptr(const Matrix& a, const Matrix& b, Matrix& res)
{
    for (int i = 0; i < MATRIX_H; i++)
        for (int k = 0; k < MATRIX_W; k++)
        {
            DTYPE t = a[i][k];
            DTYPE* p0 = res[i];
            const DTYPE* p2 = b[k];

            int upper = MATRIX_W >> 2;
            for (int j = 0; j < upper; j++)
            {
                *p0 += t * (*p2); p0++; p2++;
                *p0 += t * (*p2); p0++; p2++;
                *p0 += t * (*p2); p0++; p2++;
                *p0 += t * (*p2); p0++; p2++;
            }
        }
}

void multiply_unroll2(const Matrix& a, const Matrix& b, Matrix& res)
{
    for (int i = 0; i < MATRIX_H; i++)
        for (int k = 0; k < MATRIX_W; k++)
        {
            DTYPE t = a[i][k];
            DTYPE* p0 = res[i];
            const DTYPE* p2 = b[k];

            int upper = MATRIX_W >> 3;
            for (int j = 0; j < upper; j++)
            {
                *p0 += t * (*p2); p0++; p2++;
                *p0 += t * (*p2); p0++; p2++;
                *p0 += t * (*p2); p0++; p2++;
                *p0 += t * (*p2); p0++; p2++;
                *p0 += t * (*p2); p0++; p2++;
                *p0 += t * (*p2); p0++; p2++;
                *p0 += t * (*p2); p0++; p2++;
                *p0 += t * (*p2); p0++; p2++;
            }
        }
}

void multiply_subblock(const Matrix& a, const Matrix& b, Matrix& res,
                      int row_off, int col_off, int k_off, int sub_size)
{
    for (int i = 0; i < sub_size; i++)
        for (int k = 0; k < sub_size; k++)
        {
            DTYPE t = a[row_off + i][k_off + k];
            DTYPE* p0 = res[row_off + i] + col_off;
            const DTYPE* p2 = b[k_off + k] + col_off;

            int upper = sub_size >> 2;
            for (int j = 0; j < upper; j++)
            {
                *p0 += t * (*p2); p0++; p2++;
                *p0 += t * (*p2); p0++; p2++;
                *p0 += t * (*p2); p0++; p2++;
                *p0 += t * (*p2); p0++; p2++;
            }
        }
}

void multiply_block(const Matrix& a, const Matrix& b, Matrix& res)
{
    int block_num = 4;
    int sub_size = MATRIX_H / block_num;
    for (int bi = 0; bi < block_num; bi++)
        for (int bj = 0; bj < block_num; bj++)
            for (int bk = 0; bk < block_num; bk++)
                multiply_subblock(
                    a, b, res,
                    bi * sub_size, bj * sub_size, bk * sub_size, sub_size
                );
}

void multiply_naive_omp(const Matrix& a, const Matrix& b, Matrix& res)
{
    #pragma omp parallel for shared(a, b, res) collapse(2)
    for (int i = 0; i < MATRIX_H; i++)
        for (int j = 0; j < MATRIX_W; j++)
            for (int k = 0; k < MATRIX_W; k++)
                res[i][j] += a[i][k] * b[k][j];
}

void multiply_unroll_omp(const Matrix& a, const Matrix& b, Matrix& res)
{
    #pragma omp parallel for shared(a, b, res)
    for (int i = 0; i < MATRIX_H; i++)
        for (int k = 0; k < MATRIX_W; k++)
        {
            DTYPE t = a[i][k];
            #pragma unroll(4)
            for (int j = 0; j < MATRIX_W; j++)
                res[i][j] += t * b[k][j];
        }
}

void multiply_unroll_ptr_omp(const Matrix& a, const Matrix& b, Matrix& res)
{
    #pragma omp parallel for shared(a, b, res)
    for (int i = 0; i < MATRIX_H; i++)
        for (int k = 0; k < MATRIX_W; k++)
        {
            DTYPE t = a[i][k];
            DTYPE* p0 = res[i];
            const DTYPE* p2 = b[k];

            int upper = MATRIX_W;
            #pragma unroll
            for (int j = 0; j < upper; j++)
            {
                *p0 += t * (*p2); p0++; p2++;
            }
        }
}

void multiply_block_omp(const Matrix& a, const Matrix& b, Matrix& res)
{
    int block_num = 4;
    int sub_size = MATRIX_H / block_num;
    #pragma omp parallel for collapse(2)
    for (int bi = 0; bi < block_num; bi++)
        for (int bj = 0; bj < block_num; bj++)
            for (int bk = 0; bk < block_num; bk++)
                multiply_subblock(
                    a, b, res,
                    bi * sub_size, bj * sub_size, bk * sub_size, sub_size
                );
}