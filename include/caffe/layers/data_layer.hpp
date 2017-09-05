#ifndef CAFFE_DATA_LAYER_HPP_
#define CAFFE_DATA_LAYER_HPP_

#include <map>
#include <vector>
#include <atomic>

#include "caffe/blob.hpp"
#include "caffe/data_reader.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/thread_pool.hpp"

namespace caffe {

template <typename Ftype, typename Btype>
class DataLayer : public BasePrefetchingDataLayer<Ftype, Btype> {
 public:
  explicit DataLayer(const LayerParameter& param);
  virtual ~DataLayer();
  void DataLayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top) override;
  // DataLayer uses DataReader instead for sharing for parallelism
  bool ShareInParallel() const override {
    return false;
  }
  const char* type() const override {
    return "Data";
  }
  int ExactNumBottomBlobs() const override {
    return 0;
  }
  int MinTopBlobs() const override {
    return 1;
  }
  int MaxTopBlobs() const override {
    return 2;
  }

  Flag* layer_inititialized_flag() override {
    return this->phase_ == TRAIN ? &layer_inititialized_flag_ : nullptr;
  }

 protected:
  void InitializePrefetch() override;
  void load_batch(Batch<Ftype>* batch, int thread_id, size_t queue_id = 0UL) override;
  size_t queue_id(size_t thread_id) const override;

  void init_offsets();
  void start_reading() override {
    reader_->start_reading();
  }

  shared_ptr<DataReader<Datum>> sample_reader_, reader_;
  mutable vector<size_t> parser_offsets_, queue_ids_;
  Flag layer_inititialized_flag_;
  std::atomic_bool sample_only_;
  mutable std::mutex mutex_setup_, mutex_prefetch_;
  const bool cache_, shuffle_;
};

}  // namespace caffe

#endif  // CAFFE_DATA_LAYER_HPP_
