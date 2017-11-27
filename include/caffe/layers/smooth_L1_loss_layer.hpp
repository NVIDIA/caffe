// ------------------------------------------------------------------
// Fast R-CNN
// copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Ross Girshick
// Modified by Wei Liu
// ------------------------------------------------------------------

#ifndef CAFFE_SMOOTH_L1_LOSS_LAYER_HPP_
#define CAFFE_SMOOTH_L1_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

/**
 * @brief Computes the SmoothL1 loss as introduced in:@f$
 *  Fast R-CNN, Ross Girshick, ICCV 2015.
 */
template <typename Ftype, typename Btype>
class SmoothL1LossLayer : public LossLayer<Ftype,Btype> {
 typedef Ftype Dtype;
 public:
  explicit SmoothL1LossLayer(const LayerParameter& param)
      : LossLayer<Ftype,Btype>(param), diff_() {}
  virtual void LayerSetUp(const vector<Blob*>& bottom,
      const vector<Blob*>& top);
  virtual void Reshape(const vector<Blob*>& bottom,
      const vector<Blob*>& top);

  virtual inline const char* type() const { return "SmoothL1Loss"; }

  virtual inline int MinBottomBlobs() const { return 2; }
  virtual inline int MaxBottomBlobs() const { return 3; }

  /**
   * Unlike most loss layers, in the SmoothL1LossLayer we can backpropagate
   * to both inputs -- override to return true and always allow force_backward.
   */
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
  }

 protected:
  /// @copydoc SmoothL1LossLayer
  virtual void Forward_cpu(const vector<Blob*>& bottom,
      const vector<Blob*>& top);
  virtual void Forward_gpu(const vector<Blob*>& bottom,
      const vector<Blob*>& top);

  virtual void Backward_cpu(const vector<Blob*>& top,
      const vector<bool>& propagate_down, const vector<Blob*>& bottom);
  virtual void Backward_gpu(const vector<Blob*>& top,
      const vector<bool>& propagate_down, const vector<Blob*>& bottom);

  TBlob<Dtype> diff_;
  TBlob<Dtype> errors_;
  bool has_weights_;
};

}  // namespace caffe

#endif  // CAFFE_SMOOTH_L1_LOSS_LAYER_HPP_
