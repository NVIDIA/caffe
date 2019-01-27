#include <vector>

#include "caffe/layers/batch_reindex_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Ftype, typename Btype>
void BatchReindexLayer<Ftype, Btype>::Reshape(const vector<Blob*>& bottom,
                                       const vector<Blob*>& top) {
  CHECK_EQ(1, bottom[1]->num_axes());

  const auto& shape0 = bottom[0]->shape();
  
  vector<int> newshape;
  newshape.reserve(shape0.size());
  
  newshape.push_back(bottom[1]->shape(0));
  for (int i = 1; i < shape0.size(); ++i) {
    newshape.push_back(shape0[i]);
  }
  
  top[0]->Reshape(newshape);
}

template <typename Ftype, typename Btype>
void BatchReindexLayer<Ftype, Btype>::Forward_cpu(const vector<Blob*>& bottom,
                                           const vector<Blob*>& top) {
  check_batch_reindex(bottom[0]->shape(0), bottom[1]->count(),
                      bottom[1]->cpu_data<Ftype>());
  if (top[0]->count() == 0) {
    return;
  }
  int inner_dim = bottom[0]->count() / bottom[0]->shape(0);
  const Ftype* in = bottom[0]->cpu_data<Ftype>();
  const Ftype* permut = bottom[1]->cpu_data<Ftype>();
  Ftype* out = top[0]->mutable_cpu_data<Ftype>();
  for (int index = 0; index < top[0]->count(); ++index) {
    int n = index / (inner_dim);
    int in_n = static_cast<int>(permut[n]);
    out[index] = in[in_n * (inner_dim) + index % (inner_dim)];
  }
}

template <typename Ftype, typename Btype>
void BatchReindexLayer<Ftype, Btype>::Backward_cpu(
    const vector<Blob*>& top, const vector<bool>& propagate_down,
    const vector<Blob*>& bottom) {
  CHECK(!propagate_down[1]) << "Cannot backprop to index.";
  if (!propagate_down[0]) {
    return;
  }
  int inner_dim = bottom[0]->count() / bottom[0]->shape(0);
  Btype* bot_diff = bottom[0]->mutable_cpu_diff<Btype>();
  const Btype* permut = bottom[1]->cpu_data<Btype>();
  const Btype* top_diff = top[0]->cpu_diff<Btype>();
  caffe_set(bottom[0]->count(), Btype(0), bot_diff);
  for (int index = 0; index < top[0]->count(); ++index) {
    int n = index / (inner_dim);
    int in_n = static_cast<int>(permut[n]);
    bot_diff[in_n * (inner_dim) + index % (inner_dim)] += top_diff[index];
  }
}

INSTANTIATE_CLASS_FB(BatchReindexLayer);
REGISTER_LAYER_CLASS(BatchReindex);

}  // namespace caffe
