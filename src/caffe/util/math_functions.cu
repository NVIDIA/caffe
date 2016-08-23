#include <math_functions.h>  // CUDA's, not caffe's, for fabs, signbit
#include <thrust/device_vector.h>
#include <thrust/functional.h>  // thrust::plus
#include <thrust/reduce.h>

#include <algorithm>
#include <cmath>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <>
void caffe_gpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C, cublasHandle_t handle) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasSgemm(handle, cuTransB, cuTransA,
      N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

template <>
void caffe_gpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C, cublasHandle_t handle) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasDgemm(handle, cuTransB, cuTransA,
      N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

template <>
void caffe_gpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y, cublasHandle_t handle) {
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CHECK(cublasSgemv(handle, cuTransA, N, M, &alpha,
      A, N, x, 1, &beta, y, 1));
}

template <>
void caffe_gpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const double alpha, const double* A, const double* x,
    const double beta, double* y, cublasHandle_t handle) {
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CHECK(cublasDgemv(handle, cuTransA, N, M, &alpha,
      A, N, x, 1, &beta, y, 1));
}

template <>
void caffe_gpu_axpy<float>(const int N, const float alpha, const float* X,
    float* Y, cublasHandle_t handle) {
  CUBLAS_CHECK(cublasSaxpy(handle, N, &alpha, X, 1, Y, 1));
}

template <>
void caffe_gpu_axpy<double>(const int N, const double alpha, const double* X,
    double* Y, cublasHandle_t handle) {
  CUBLAS_CHECK(cublasDaxpy(handle, N, &alpha, X, 1, Y, 1));
}

void caffe_gpu_memcpy(const size_t N, const void* X, void* Y,
                      cublasHandle_t handle) {
  cudaStream_t stream;
  CUBLAS_CHECK(cublasGetStream(handle, &stream));
  if (X != Y) {
    CUDA_CHECK(cudaMemcpyAsync(Y, X, N, cudaMemcpyDefault,
      stream));  // NOLINT(caffe/alt_fn)
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

template <>
void caffe_gpu_scal<float>(const int N, const float alpha, float *X,
                           cublasHandle_t handle) {
  CUBLAS_CHECK(cublasSscal(handle, N, &alpha, X, 1));
}

template <>
void caffe_gpu_scal<double>(const int N, const double alpha, double *X,
                            cublasHandle_t handle) {
  CUBLAS_CHECK(cublasDscal(handle, N, &alpha, X, 1));
}

template <>
void caffe_gpu_scal<float>(const int N, const float alpha, float* X,
    cudaStream_t str) {
  cudaStream_t initial_stream;
  CUBLAS_CHECK(cublasGetStream(Caffe::cublas_handle(), &initial_stream));
  CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), str));
  CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), N, &alpha, X, 1));
  CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), initial_stream));
}

template <>
void caffe_gpu_scal<double>(const int N, const double alpha, double* X,
    cudaStream_t str) {
  cudaStream_t initial_stream;
  CUBLAS_CHECK(cublasGetStream(Caffe::cublas_handle(), &initial_stream));
  CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), str));
  CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), N, &alpha, X, 1));
  CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), initial_stream));
}

template <>
void caffe_gpu_axpby<float>(const int N, const float alpha, const float* X,
    const float beta, float* Y, cublasHandle_t handle) {
  caffe_gpu_scal<float>(N, beta, Y, handle);
  caffe_gpu_axpy<float>(N, alpha, X, Y, handle);
}

template <>
void caffe_gpu_axpby<double>(const int N, const double alpha, const double* X,
    const double beta, double* Y, cublasHandle_t handle) {
  caffe_gpu_scal<double>(N, beta, Y, handle);
  caffe_gpu_axpy<double>(N, alpha, X, Y, handle);
}

template <>
void caffe_gpu_dot<float>(const int n, const float* x, const float* y,
    float* out, cublasHandle_t handle) {
  CUBLAS_CHECK(cublasSdot(handle, n, x, 1, y, 1, out));
}

template <>
void caffe_gpu_dot<double>(const int n, const double* x, const double* y,
    double * out, cublasHandle_t handle) {
  CUBLAS_CHECK(cublasDdot(handle, n, x, 1, y, 1, out));
}

template <>
void caffe_gpu_asum<float>(const int n, const float* x, float* y,
                           cublasHandle_t handle) {
  CUBLAS_CHECK(cublasSasum(handle, n, x, 1, y));
}

template <>
void caffe_gpu_asum<double>(const int n, const double* x, double* y,
                            cublasHandle_t handle) {
  CUBLAS_CHECK(cublasDasum(handle, n, x, 1, y));
}

template <>
void caffe_gpu_scale<float>(const int n, const float alpha, const float *x,
                            float* y, cublasHandle_t handle) {
  CUBLAS_CHECK(cublasScopy(handle, n, x, 1, y, 1));
  CUBLAS_CHECK(cublasSscal(handle, n, &alpha, y, 1));
}

template <>
void caffe_gpu_scale<double>(const int n, const double alpha, const double *x,
                             double* y, cublasHandle_t handle) {
  CUBLAS_CHECK(cublasDcopy(handle, n, x, 1, y, 1));
  CUBLAS_CHECK(cublasDscal(handle, n, &alpha, y, 1));
}

template <typename Dtype>
__global__ void set_kernel(const int n, const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = alpha;
  }
}

template <typename Dtype>
void caffe_gpu_set(const int N, const Dtype alpha, Dtype* Y,
    cublasHandle_t handle) {
  cudaStream_t stream;
  CUBLAS_CHECK(cublasGetStream(handle, &stream));
  if (alpha == 0) {
    CUDA_CHECK(cudaMemsetAsync(Y, 0, sizeof(Dtype) * N,
      stream));  // NOLINT(caffe/alt_fn)
    return;
  }
  // NOLINT_NEXT_LINE(whitespace/operators)
  set_kernel<Dtype><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0,
      stream>>>(N, alpha, Y);
}

template void caffe_gpu_set<int>(const int N, const int alpha, int* Y,
    cublasHandle_t handle);
template void caffe_gpu_set<float>(const int N, const float alpha, float* Y,
    cublasHandle_t handle);
template void caffe_gpu_set<double>(const int N, const double alpha, double* Y,
    cublasHandle_t handle);

template <typename Dtype>
__global__ void add_scalar_kernel(const int n, const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] += alpha;
  }
}

template <>
void caffe_gpu_add_scalar(const int N, const float alpha, float* Y,
    cublasHandle_t handle) {
  cudaStream_t stream;
  CUBLAS_CHECK(cublasGetStream(handle, &stream));
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_scalar_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0,
      stream>>>(N, alpha, Y);
}

template <>
void caffe_gpu_add_scalar(const int N, const double alpha, double* Y,
    cublasHandle_t handle) {
  cudaStream_t stream;
  CUBLAS_CHECK(cublasGetStream(handle, &stream));
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_scalar_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS,
      0, stream>>>(N, alpha, Y);
}

template <typename Dtype>
__global__ void add_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] + b[index];
  }
}

template <>
void caffe_gpu_add<float>(const int N, const float* a, const float* b,
    float* y, cublasHandle_t handle) {
  cudaStream_t stream;
  CUBLAS_CHECK(cublasGetStream(handle, &stream));
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0,
      stream>>>(N, a, b, y);
}

template <>
void caffe_gpu_add<double>(const int N, const double* a, const double* b,
    double* y, cublasHandle_t handle) {
  cudaStream_t stream;
  CUBLAS_CHECK(cublasGetStream(handle, &stream));
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0,
      stream>>>(N, a, b, y);
}

template <typename Dtype>
__global__ void sub_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] - b[index];
  }
}

template <>
void caffe_gpu_sub<float>(const int N, const float* a, const float* b,
    float* y, cublasHandle_t handle) {
  cudaStream_t stream;
  CUBLAS_CHECK(cublasGetStream(handle, &stream));
  // NOLINT_NEXT_LINE(whitespace/operators)
  sub_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0,
      stream>>>(N, a, b, y);
}

template <>
void caffe_gpu_sub<double>(const int N, const double* a, const double* b,
    double* y, cublasHandle_t handle) {
  cudaStream_t stream;
  CUBLAS_CHECK(cublasGetStream(handle, &stream));
  // NOLINT_NEXT_LINE(whitespace/operators)
  sub_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0,
      stream>>>(N, a, b, y);
}

template <typename Dtype>
__global__ void mul_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] * b[index];
  }
}

template <>
void caffe_gpu_mul<float>(const int N, const float* a,
    const float* b, float* y, cublasHandle_t handle) {
  cudaStream_t stream;
  CUBLAS_CHECK(cublasGetStream(handle, &stream));
  // NOLINT_NEXT_LINE(whitespace/operators)
  mul_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0,
      stream>>>(N, a, b, y);
}

template <>
void caffe_gpu_mul<double>(const int N, const double* a,
    const double* b, double* y, cublasHandle_t handle) {
  cudaStream_t stream;
  CUBLAS_CHECK(cublasGetStream(handle, &stream));
  // NOLINT_NEXT_LINE(whitespace/operators)
  mul_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0,
      stream>>>(N, a, b, y);
}

template <typename Dtype>
__global__ void div_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] / b[index];
  }
}

template <>
void caffe_gpu_div<float>(const int N, const float* a,
    const float* b, float* y, cublasHandle_t handle) {
  cudaStream_t stream;
  CUBLAS_CHECK(cublasGetStream(handle, &stream));
  // NOLINT_NEXT_LINE(whitespace/operators)
  div_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0,
      stream>>>(N, a, b, y);
}

template <>
void caffe_gpu_div<double>(const int N, const double* a,
    const double* b, double* y, cublasHandle_t handle) {
  cudaStream_t stream;
  CUBLAS_CHECK(cublasGetStream(handle, &stream));
  // NOLINT_NEXT_LINE(whitespace/operators)
  div_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0,
      stream>>>(N, a, b, y);
}

template <typename Dtype>
__global__ void abs_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = abs(a[index]);
  }
}

template <>
void caffe_gpu_abs<float>(const int N, const float* a, float* y,
    cublasHandle_t handle) {
  cudaStream_t stream;
  CUBLAS_CHECK(cublasGetStream(handle, &stream));
  // NOLINT_NEXT_LINE(whitespace/operators)
  abs_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0,
      stream>>>(N, a, y);
}

template <>
void caffe_gpu_abs<double>(const int N, const double* a, double* y,
    cublasHandle_t handle) {
  cudaStream_t stream;
  CUBLAS_CHECK(cublasGetStream(handle, &stream));
  // NOLINT_NEXT_LINE(whitespace/operators)
  abs_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0,
      stream>>>(N, a, y);
}


template <typename Dtype>
__global__ void exp_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = exp(a[index]);
  }
}

template <>
void caffe_gpu_exp<float>(const int N, const float* a, float* y,
    cublasHandle_t handle) {
  cudaStream_t stream;
  CUBLAS_CHECK(cublasGetStream(handle, &stream));
  // NOLINT_NEXT_LINE(whitespace/operators)
  exp_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0,
     stream>>>(N, a, y);
}

template <>
void caffe_gpu_exp<double>(const int N, const double* a, double* y,
    cublasHandle_t handle) {
  cudaStream_t stream;
  CUBLAS_CHECK(cublasGetStream(handle, &stream));
  // NOLINT_NEXT_LINE(whitespace/operators)
  exp_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0,
      stream>>>(N, a, y);
}

template <typename Dtype>
__global__ void log_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = log(a[index]);
  }
}

template <>
void caffe_gpu_log<float>(const int N, const float* a, float* y,
    cublasHandle_t handle) {
  cudaStream_t stream;
  CUBLAS_CHECK(cublasGetStream(handle, &stream));
  // NOLINT_NEXT_LINE(whitespace/operators)
  log_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0,
      stream>>>(N, a, y);
}

template <>
void caffe_gpu_log<double>(const int N, const double* a, double* y,
    cublasHandle_t handle) {
  cudaStream_t stream;
  CUBLAS_CHECK(cublasGetStream(handle, &stream));
  // NOLINT_NEXT_LINE(whitespace/operators)
  log_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0,
      stream>>>(N, a, y);
}

template <typename Dtype>
__global__ void powx_kernel(const int n, const Dtype* a,
    const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = pow(a[index], alpha);
  }
}

template <>
void caffe_gpu_powx<float>(const int N, const float* a,
    const float alpha, float* y, cublasHandle_t handle) {
  cudaStream_t stream;
  CUBLAS_CHECK(cublasGetStream(handle, &stream));
  // NOLINT_NEXT_LINE(whitespace/operators)
  powx_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0,
      stream>>>(N, a, alpha, y);
}

template <>
void caffe_gpu_powx<double>(const int N, const double* a,
    const double alpha, double* y, cublasHandle_t handle) {
  cudaStream_t stream;
  CUBLAS_CHECK(cublasGetStream(handle, &stream));
  // NOLINT_NEXT_LINE(whitespace/operators)
  powx_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0,
      stream>>>(N, a, alpha, y);
}

DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(sign, y[index] = (Dtype(0) < x[index])
                                      - (x[index] < Dtype(0)));
DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(sgnbit, y[index] = signbit(x[index]));

void caffe_gpu_rng_uniform(const int n, unsigned int* r,
    cublasHandle_t handle) {
  CURAND_CHECK(curandGenerate(Caffe::curand_generator(), r, n));
}

template <>
void caffe_gpu_rng_uniform<float>(const int n, const float a, const float b,
                                  float* r, cublasHandle_t handle) {
  CURAND_CHECK(curandGenerateUniform(Caffe::curand_generator(), r, n));
  const float range = b - a;
  if (range != static_cast<float>(1)) {
    caffe_gpu_scal(n, range, r);
  }
  if (a != static_cast<float>(0)) {
    caffe_gpu_add_scalar(n, a, r);
  }
}

template <>
void caffe_gpu_rng_uniform<double>(const int n, const double a, const double b,
                                   double* r, cublasHandle_t handle) {
  CURAND_CHECK(curandGenerateUniformDouble(Caffe::curand_generator(), r, n));
  const double range = b - a;
  if (range != static_cast<double>(1)) {
    caffe_gpu_scal(n, range, r, handle);
  }
  if (a != static_cast<double>(0)) {
    caffe_gpu_add_scalar(n, a, r, handle);
  }
}

template <>
void caffe_gpu_rng_gaussian(const int n, const float mu, const float sigma,
                            float* r, cublasHandle_t handle) {
  CURAND_CHECK(
      curandGenerateNormal(Caffe::curand_generator(), r, n, mu, sigma));
}

template <>
void caffe_gpu_rng_gaussian(const int n, const double mu, const double sigma,
                            double* r, cublasHandle_t handle) {
  CURAND_CHECK(
      curandGenerateNormalDouble(Caffe::curand_generator(), r, n, mu, sigma));
}

template <typename Dtype>
__global__ void caffe_gpu_eltwise_max_kernel(const int N,
    const Dtype alpha, const Dtype* x, const Dtype beta, Dtype* y) {
  CUDA_KERNEL_LOOP(index, N) {
    y[index] = max(alpha * x[index] , beta * y[index]);
  }
}

template <>
void caffe_gpu_eltwise_max<float>(const int N, const float alpha,
    const float* x, const float beta, float* y, cublasHandle_t handle) {
  cudaStream_t stream;
  CUBLAS_CHECK(cublasGetStream(handle, &stream));
  // NOLINT_NEXT_LINE(whitespace/operators)
  caffe_gpu_eltwise_max_kernel<float><<<CAFFE_GET_BLOCKS(N),
                               CAFFE_CUDA_NUM_THREADS, 0, stream>>>(
                               N, alpha, x, beta, y);
}
template <>
void caffe_gpu_eltwise_max<double>(const int N, const double alpha,
    const double* x, const double beta, double* y, cublasHandle_t handle) {
  cudaStream_t stream;
  CUBLAS_CHECK(cublasGetStream(handle, &stream));
  // NOLINT_NEXT_LINE(whitespace/operators)
  caffe_gpu_eltwise_max_kernel<double><<<CAFFE_GET_BLOCKS(N),
                               CAFFE_CUDA_NUM_THREADS, 0, stream>>>(
                               N, alpha, x, beta, y);
}


template <typename Dtype>
__global__ void caffe_gpu_eltwise_min_kernel(const int N,
    const Dtype alpha, const Dtype* x, const Dtype beta, Dtype* y) {
  CUDA_KERNEL_LOOP(index, N) {
    y[index] = min(alpha * x[index] , beta * y[index]);
  }
}

template <>
void caffe_gpu_eltwise_min<float>(const int N, const float alpha,
    const float* x, const float beta, float* y, cublasHandle_t handle) {
  cudaStream_t stream;
  CUBLAS_CHECK(cublasGetStream(handle, &stream));
  // NOLINT_NEXT_LINE(whitespace/operators)
  caffe_gpu_eltwise_min_kernel<float><<<CAFFE_GET_BLOCKS(N),
                               CAFFE_CUDA_NUM_THREADS, 0, stream>>>(
                               N, alpha, x, beta, y);
}
template <>
void caffe_gpu_eltwise_min<double>(const int N, const double alpha,
    const double* x, const double beta, double* y, cublasHandle_t handle) {
  cudaStream_t stream;
  CUBLAS_CHECK(cublasGetStream(handle, &stream));
  // NOLINT_NEXT_LINE(whitespace/operators)
  caffe_gpu_eltwise_min_kernel<double><<<CAFFE_GET_BLOCKS(N),
                               CAFFE_CUDA_NUM_THREADS, 0, stream>>>(
                               N, alpha, x, beta, y);
}

}  // namespace caffe
