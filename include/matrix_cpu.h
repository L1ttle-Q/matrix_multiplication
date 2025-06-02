#ifndef __MATRIX_CPU_H__
#define __MATRIX_CPU_H__

void multiply_naive(const Matrix&, const Matrix&, Matrix&);
void multiply_reordered(const Matrix&, const Matrix&, Matrix&);
void multiply_unroll(const Matrix&, const Matrix&, Matrix&);
void multiply_unroll_ptr(const Matrix&, const Matrix&, Matrix&);
void multiply_unroll2(const Matrix&, const Matrix&, Matrix&);
void multiply_block(const Matrix&, const Matrix&, Matrix&);
void multiply_naive_omp(const Matrix&, const Matrix&, Matrix&);
void multiply_unroll_omp(const Matrix&, const Matrix&, Matrix&);
void multiply_unroll_ptr_omp(const Matrix& , const Matrix& , Matrix&);
void multiply_block_omp(const Matrix&, const Matrix&, Matrix&);

#endif /* __MATRIX_CPU_H__ */