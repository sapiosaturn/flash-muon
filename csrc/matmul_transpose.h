#pragma once
#include <cute/tensor.hpp>

template <typename T>
void launch_mmt_kernel(T *x, T *y, int M, int K, cudaStream_t stream);
