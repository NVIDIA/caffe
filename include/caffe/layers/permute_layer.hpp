#ifndef CAFFE_PERMUTE_LAYER_HPP_
#define CAFFE_PERMUTE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Permute the input blob by changing the memory order of the data.
 *
 * TODO(weiliu89): thorough documentation for Forward, Backward, and proto params.
 */

// The main function which does the permute.
template <typename Dtype>
void Permute(const int count, Dtype* bottom_data, const bool forward,
    const int* permute_order, const int* old_steps, const int* new_steps,
    const int num_axes, Dtype* top_data);

template <typename Ftype, typename Btype>
class PermuteLayer : public Layer<Ftype,Btype> {
 typedef Ftype Dtype;
 public:
  explicit PermuteLayer(const LayerParameter& param)
      : Layer<Ftype,Btype>(param) {}
  virtual void LayerSetUp(const vector<Blob*>& bottom,
      const vector<Blob*>& top);
  virtual void Reshape(const vector<Blob*>& bottom,
      const vector<Blob*>& top);

  virtual inline const char* type() const { return "Permute"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob*>& bottom,
      const vector<Blob*>& top);
  virtual void Forward_gpu(const vector<Blob*>& bottom,
      const vector<Blob*>& top);
  virtual void Backward_cpu(const vector<Blob*>& top,
      const vector<bool>& propagate_down, const vector<Blob*>& bottom);
  virtual void Backward_gpu(const vector<Blob*>& top,
      const vector<bool>& propagate_down, const vector<Blob*>& bottom);

  int num_axes_;
  bool need_permute_;

  // Use Blob because it is convenient to be accessible in .cu file.
  TBlob<int> permute_order_;
  TBlob<int> old_steps_;
  TBlob<int> new_steps_;
};

}  // namespace caffe

#endif  // CAFFE_PERMUTE_LAYER_HPP_
