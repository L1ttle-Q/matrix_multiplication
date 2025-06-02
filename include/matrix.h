#ifndef __MATRIX_H__
#define __MATRIX_H__

#include "global.h"
#include <random>

class Matrix
{
public:
    Matrix() = default;

    DTYPE a[MATRIX_H][MATRIX_W];

    DTYPE* operator [](const int& x)
    {
        return a[x];
    }

    const DTYPE* operator [](const int& x) const
    {
        return a[x];
    }

    void generate();
    void init();
};

void Multiply(const Matrix&, const Matrix&, Matrix&);
bool check_mat(const Matrix&, const Matrix&);

#endif /* __MATRIX_H__ */