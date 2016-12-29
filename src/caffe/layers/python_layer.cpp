#ifdef WITH_PYTHON_LAYER
#include "caffe/layers/python_layer.hpp"
#include <vector>

namespace caffe {
template<typename Dtype>
PythonLayer<Dtype>::PythonLayer(PyObject* self, const LayerParameter& param)
    : Layer<Dtype>(param), self_(bp::handle<>(bp::borrowed(self))) { }

template<typename Dtype>
void  PythonLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    // Disallow PythonLayer in MultiGPU training stage, due to GIL issues
    // Details: https://github.com/BVLC/caffe/issues/2936
    if (this->phase_ == TRAIN && Caffe::solver_count() > 1
        && !ShareInParallel()) {
        LOG(FATAL) << "PythonLayer is not implemented in Multi-GPU training";
    }
    self_.attr("param_str") = bp::str(
        this->layer_param_.python_param().param_str());
    self_.attr("phase") = static_cast<int>(this->phase_);
    self_.attr("setup")(bottom, top);
}

template<typename Dtype>
void  PythonLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    self_.attr("reshape")(bottom, top);
}

template<typename Dtype>
bool PythonLayer<Dtype>::ShareInParallel() const {
    return this->layer_param_.python_param().share_in_parallel();
}

template<typename Dtype>
const char* PythonLayer<Dtype>::type() const { return "Python"; }


template<typename Dtype>
void PythonLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    self_.attr("forward")(bottom, top);
}

template<typename Dtype>
void  PythonLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    self_.attr("backward")(top, propagate_down, bottom);
}
INSTANTIATE_CLASS(PythonLayer);

}  // namespace caffe
#endif  // WITH_PYTHON_LAYER
