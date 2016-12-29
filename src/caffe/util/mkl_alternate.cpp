#include "caffe/util/mkl_alternate.hpp"
#include <glog/logging.h>
#include "caffe/export.hpp"

#define DEFINE_VSL_UNARY_FUNC(name, operation) \
  template<typename Dtype> CAFFE_EXPORT \
  void v##name(const int n, const Dtype* a, Dtype* y) { \
    CHECK_GT(n, 0); CHECK(a); CHECK(y); \
    for (int i = 0; i < n; ++i) { operation; } \
  } \
  template CAFFE_EXPORT void v##name<float>(const int n, \
    const float* a, float* y); \
  template CAFFE_EXPORT void v##name<double>(const int n,  \
    const double* a, double* y);

DEFINE_VSL_UNARY_FUNC(Sqr, y[i] = a[i] * a[i]);
DEFINE_VSL_UNARY_FUNC(Exp, y[i] = exp(a[i]));
DEFINE_VSL_UNARY_FUNC(Ln, y[i] = log(a[i]));
DEFINE_VSL_UNARY_FUNC(Abs, y[i] = fabs(a[i]));

// A simple way to define the vsl unary functions with singular parameter b.
// The operation should be in the form e.g. y[i] = pow(a[i], b)
#define DEFINE_VSL_UNARY_FUNC_WITH_PARAM(name, operation) \
  template<typename Dtype> \
  void v##name(const int n, const Dtype* a, const Dtype b, Dtype* y) { \
    CHECK_GT(n, 0); CHECK(a); CHECK(y); \
    for (int i = 0; i < n; ++i) { operation; } \
  } \
  template CAFFE_EXPORT void v##name<float>(const int n, \
    const float* a, const float b, float* y); \
  template CAFFE_EXPORT void v##name<double>(const int n, \
    const double* a, const double b, double* y); \

DEFINE_VSL_UNARY_FUNC_WITH_PARAM(Powx, y[i] = pow(a[i], b));

// A simple way to define the vsl binary functions. The operation should
// be in the form e.g. y[i] = a[i] + b[i]
#define DEFINE_VSL_BINARY_FUNC(name, operation) \
  template<typename Dtype> CAFFE_EXPORT\
  void v##name(const int n, const Dtype* a, const Dtype* b, Dtype* y) { \
    CHECK_GT(n, 0); CHECK(a); CHECK(b); CHECK(y); \
    for (int i = 0; i < n; ++i) { operation; } \
  } \
  template CAFFE_EXPORT void v##name<float>(const int n, \
    const float* a, const float* b, float* y); \
  template CAFFE_EXPORT void v##name<double>(const int n, \
    const double* a, const double* b, double* y);

DEFINE_VSL_BINARY_FUNC(Add, y[i] = a[i] + b[i]);
DEFINE_VSL_BINARY_FUNC(Sub, y[i] = a[i] - b[i]);
DEFINE_VSL_BINARY_FUNC(Mul, y[i] = a[i] * b[i]);
DEFINE_VSL_BINARY_FUNC(Div, y[i] = a[i] / b[i]);
