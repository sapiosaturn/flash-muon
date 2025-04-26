#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "matmul_transpose.h"

// TC: cutlass numeric type, TT: Torch Aten numeric type
template <class TC, class TT>
void compute_mmt_imp(torch::Tensor d_in, torch::Tensor d_out)
{
    d_in = d_in.contiguous();

    int M = d_in.size(0);
    int K = d_in.size(1);

    if (K % 8 != 0)
    {
        std::stringstream error_message;
        error_message << "Error when launching compute_mmt_imp for Tensor `d_in` of shape (MxK): K must be a multiple of 8 (got " << K << ")";
        throw std::runtime_error(error_message.str());
    }

    at::cuda::CUDAGuard device_guard(d_in.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();


    launch_mmt_kernel<TC>(reinterpret_cast<TC *>(d_in.data_ptr<TT>()), reinterpret_cast<TC *>(d_out.data_ptr<TT>()), M, K, stream);
}

void matmul_transpose_assign(torch::Tensor d_in, torch::Tensor d_out)
{
    TORCH_CHECK(d_in.is_cuda(), "Input `d_in` must be a CUDA tensor");
    TORCH_CHECK(d_out.is_cuda(), "Input `d_out` must be a CUDA tensor");
    TORCH_CHECK(d_in.device() == d_out.device(),
                "Inputs `d_in` and `d_out` must be on the same CUDA device");
    TORCH_CHECK(d_in.scalar_type() == d_out.scalar_type(), "Inputs must have the same data type");
    TORCH_CHECK(d_in.dim() == 2, "Input `d_in` must be a 2D tensor");
    TORCH_CHECK(d_out.dim() == 2, "Input `d_out` must be a 2D tensor");
    TORCH_CHECK(d_in.size(0) == d_out.size(0), 
               "First dimension of `d_in` must match first dimension of `d_out`");
    TORCH_CHECK(d_out.size(0) == d_out.size(1),
               "Output `d_out` must be a square matrix (d_out.size(0) == d_out.size(1))");
    switch (d_in.scalar_type())
    {
    case at::ScalarType::Half:
        compute_mmt_imp<cute::half_t, at::Half>(d_in, d_out);
        break;
    case at::ScalarType::BFloat16:
        compute_mmt_imp<cute::bfloat16_t, at::BFloat16>(d_in, d_out); // 或 cute::bfloat16_t（取决于 CUTLASS 定义）
        break;
    default:
        TORCH_CHECK(false, "Unsupported data type. Only fp16 and bf16 are supported");
    }
}

torch::Tensor matmul_transpose(torch::Tensor d_in)
{
    int m = d_in.size(0);
    auto d_out = torch::empty({m, m}, d_in.options());

    matmul_transpose_assign(d_in, d_out);
    return d_out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("matmul_transpose", &matmul_transpose, "Calculate and return `d_in @ d_in.T`");
    m.def("matmul_transpose_assign", &matmul_transpose_assign, "Calculate `d_in @ d_in.T` and assign to output Tensor `d_out`");
}