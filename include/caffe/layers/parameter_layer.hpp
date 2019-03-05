#ifndef CAFFE_PARAMETER_LAYER_HPP_
#define CAFFE_PARAMETER_LAYER_HPP_

#include <vector>

#include "caffe/layer.hpp"

namespace caffe {

//@RefineDet 
//add paramter param 
template <typename Ftype, typename Btype>
class ParameterLayer : public Layer<Ftype, Btype> {
 public:
  explicit ParameterLayer(const LayerParameter& param)
      : Layer<Ftype, Btype>(param) {}
  virtual void LayerSetUp(const vector<Blob*>& bottom,
      const vector<Blob*>& top) {
    if (this->blobs_.size() > 0) {
      LOG(INFO) << "Skipping parameter initialization";
    } else {
      this->blobs_.resize(1);
      this->blobs_[0].reset();
      this->blobs_[0]->Reshape(this->layer_param_.parameter_param().shape());
    }
    top[0]->Reshape(this->layer_param_.parameter_param().shape());
  }
  virtual void Reshape(const vector<Blob*>& bottom,
      const vector<Blob*>& top) { }
  virtual inline const char* type() const { return "Parameter"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob*>& bottom,
      const vector<Blob*>& top) {
    top[0]->ShareData(*(this->blobs_[0]));
    top[0]->ShareDiff(*(this->blobs_[0]));
  }
  virtual void Backward_cpu(const vector<Blob*>& top,
      const vector<bool>& propagate_down, const vector<Blob*>& bottom)
  { }
};

}  // namespace caffe

#endif
