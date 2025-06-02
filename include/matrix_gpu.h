#ifndef __MATRIX_GPU_H__
#define __MATRIX_GPU_H__

#include "global.h"

__global__ void multiply_naive(const DTYPE*, const DTYPE*, DTYPE*);
__global__ void multiply_block(const DTYPE*, const DTYPE*, DTYPE*);
__global__ void multiply_shared(const DTYPE*, const DTYPE*, DTYPE*);
__global__ void multiply_shared_coalesce(const DTYPE*, const DTYPE*, DTYPE*);
__global__ void multiply_less_conflict(const DTYPE*, const DTYPE*, DTYPE*);
__global__ void multiply_unroll(const DTYPE*, const DTYPE*, DTYPE*);
void multiply_cublas(const DTYPE*, const DTYPE*, DTYPE*);

#endif /* __MATRIX_GPU_H__ */