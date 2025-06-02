#include "matrix.h"
#include <cstring>
#include <iostream>
#include <type_traits>
#include <cmath>

int Rand(const int& upper)
{
    return rand() % upper;
}

void Matrix::generate()
{
    for (int i = 0; i < MATRIX_H; i++)
        for (int j = 0; j < MATRIX_W; j++)
            a[i][j] = static_cast<DTYPE>(Rand(1024));
}

void Matrix::init()
{
    memset(this->a, 0, sizeof(DTYPE) * MATRIX_H * MATRIX_W);
}


void Multiply(const Matrix& a, const Matrix& b, Matrix& res) // unroll and ptr
{
    res.init();
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

bool check_mat(const Matrix& a, const Matrix& b)
{
    if constexpr (std::is_integral<DTYPE>::value)
    {
        for (int i = 0; i < MATRIX_H; i++)
            for (int j = 0; j < MATRIX_W; j++)
                if (a.a[i][j] != b.a[i][j])
                    return false;
        return true;
    }
    else
    {
        static const float eps = 1e-5;
        for (int i = 0; i < MATRIX_H; i++)
            for (int j = 0; j < MATRIX_W; j++)
                if (std::fabs(a.a[i][j] - b.a[i][j]) > eps * a[i][j])
                {
                    fprintf(stderr, "%d, %d: %f %f\n", i, j, a[i][j], b[i][j]);
                    return false;
                }
        return true;
    }
}