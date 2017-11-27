// ------------------------------------------------------------------
// Fast R-CNN
// copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Ross Girshick
// Modified by Wei Liu
// ------------------------------------------------------------------

#include <vector>

#include "caffe/layers/smooth_L1_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Ftype, typename Btype>
void SmoothL1LossLayer<Ftype,Btype>::LayerSetUp(
  const vector<Blob*>& bottom, const vector<Blob*>& top) {
  LossLayer<Ftype,Btype>::LayerSetUp(bottom, top);
  has_weights_ = (bottom.size() == 3);
}

template <typename Ftype, typename Btype>
void SmoothL1LossLayer<Ftype,Btype>::Reshape(
  const vector<Blob*>& bottom, const vector<Blob*>& top) {
  LossLayer<Ftype,Btype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());
  if (has_weights_) {
    CHECK_EQ(bottom[0]->channels(), bottom[2]->channels());
    CHECK_EQ(bottom[0]->height(), bottom[2]->height());
    CHECK_EQ(bottom[0]->width(), bottom[2]->width());
  }
  diff_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
  errors_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
}

template <typename Ftype, typename Btype>
void SmoothL1LossLayer<Ftype,Btype>::Forward_cpu(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
  int count = bottom[0]->count();
  caffe_sub(
      count,
      bottom[0]->cpu_data<Dtype>(),
      bottom[1]->cpu_data<Dtype>(),
      diff_.mutable_cpu_data());
  if (has_weights_) {
    caffe_mul(
        count,
        bottom[2]->cpu_data<Dtype>(),
        diff_.cpu_data(),
        diff_.mutable_cpu_data());  // d := w * (b0 - b1)
  }
  const Dtype* diff_data = diff_.cpu_data();
  Dtype* error_data = errors_.mutable_cpu_data();
  for (int i = 0; i < count; ++i) {
    Dtype val = diff_data[i];
    Dtype abs_val = fabs(val);
    if (abs_val < 1.) {
      error_data[i] = 0.5 * val * val;
    } else {
      error_data[i] = abs_val - 0.5;
    }
  }
  top[0]->mutable_cpu_data<Dtype>()[0] =
      caffe_cpu_asum(count, errors_.cpu_data()) / bottom[0]->num();
}

template <typename Ftype, typename Btype>
void SmoothL1LossLayer<Ftype,Btype>::Backward_cpu(const vector<Blob*>& top,
    const vector<bool>& propagate_down, const vector<Blob*>& bottom) {
  int count = diff_.count();
  Dtype* diff_data = diff_.mutable_cpu_data();
  for (int i = 0; i < count; ++i) {
    Dtype val = diff_data[i];
    // f'(x) = x         if |x| < 1
    //       = sign(x)   otherwise
    if (fabs(val) < 1.) {
      diff_data[i] = val;
    } else {
      diff_data[i] = (Dtype(0) < val) - (val < Dtype(0));
    }
  }
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff<Dtype>()[0] / bottom[i]->num();
      caffe_cpu_axpby(
          bottom[i]->count(),               // count
          alpha,                            // alpha
          diff_.cpu_data(),                 // a
          Dtype(0),                         // beta
          bottom[i]->mutable_cpu_diff<Dtype>());   // b
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(SmoothL1LossLayer);
#endif

INSTANTIATE_CLASS_FB(SmoothL1LossLayer);
REGISTER_LAYER_CLASS(SmoothL1Loss);

}  // namespace caffe
