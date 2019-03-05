#ifndef CAFFE_TRT_LAYER_HPP
#define CAFFE_TRT_LAYER_HPP

#ifdef USE_TRT
#include <vector>
#include <fstream>

#include "caffe/layers/NvCaffeParser.h"
#include "caffe/layers/NvInfer.h"

//#include <NvCaffeParser.h>
//#include <NvInfer.h>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

// NOLINT_NEXT_LINE(build/namespaces)
using namespace nvinfer1;
// NOLINT_NEXT_LINE(build/namespaces)
using namespace nvcaffeparser1;

namespace caffe {

class PreGeneratedCalibrator : public IInt8EntropyCalibrator {
 public:
  explicit PreGeneratedCalibrator(const char* calibrationTableName)
      : mTableName(calibrationTableName) {}

  int getBatchSize() const override {
    return 0;
  }

  bool getBatch(void* bindings[], const char* names[], int nbBindings) override {
    return false;
  }

  const void* readCalibrationCache(size_t& length) override {
    std::ifstream input(mTableName, std::ios::binary);
    CHECK(input.good());
    input >> std::noskipws;

    mCalibrationCache.clear();
    std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(),
        std::back_inserter(mCalibrationCache));

    length = mCalibrationCache.size();
    return mCalibrationCache.data();
  }

  void writeCalibrationCache(const void* cache, size_t length) override {}

 private:
  std::string mTableName;
  std::vector<char> mCalibrationCache;
};


/**
 * @brief TensorRT-based inference executor.
 */
template <typename Ftype, typename Btype>
class TRTLayer : public Layer<Ftype, Btype> {
 public:
  explicit TRTLayer(const LayerParameter& param)
      : Layer<Ftype, Btype>(param),
        batch_size_(1),
        top_k_(0),
        num_labels_(0),
        itr_count_(0),
        engine_(nullptr),
        context_(nullptr) {}
  ~TRTLayer();
  void LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top) override;
  void Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top) override;

  inline const char* type() const override { return "TRT"; }
  inline int ExactNumBottomBlobs() const override { return 2; }

  // If there are two top blobs, then the second blob will contain
  // accuracies per class.
  inline int MinTopBlobs() const override { return 1; }
  inline int MaxTopBlobs() const override { return 1; }

  std::string print_stats() const override;

 protected:
  void Forward_cpu(const vector<Blob*>& bottom, const vector<Blob*>& top) override;

  /// @brief Not implemented -- TRTLayer cannot be used as a loss.
  void Backward_cpu(const vector<Blob*>& top,
      const vector<bool>& propagate_down, const vector<Blob*>& bottom) override {
    for (int i = 0; i < propagate_down.size(); ++i) {
      if (propagate_down[i]) { NOT_IMPLEMENTED; }
    }
  }

  int batch_size_;
  int top_k_;
  int num_labels_;
  int itr_count_;
  ICudaEngine* engine_;
  IExecutionContext* context_;
  std::string input_, output_;
  void* buffers_[2];
  int inputIndex_, outputIndex_;
  TBlob<float> probs_;
  // NOLINT_NEXT_LINE(runtime/int)
  std::vector<long> labels_;
  cudaStream_t stream_;
  shared_ptr<PreGeneratedCalibrator> calibrator_;
};

}  // namespace caffe

#endif // USE_TRT
#endif //CAFFE_TRT_LAYER_HPP
