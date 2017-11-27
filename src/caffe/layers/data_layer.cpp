#include "caffe/data_transformer.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/data_layer.hpp"
#include "caffe/parallel.hpp"

namespace caffe {

template<typename Ftype, typename Btype>
DataLayer<Ftype, Btype>::DataLayer(const LayerParameter& param)
  : BasePrefetchingDataLayer<Ftype, Btype>(param),
    cache_(param.data_param().cache()),
    shuffle_(param.data_param().shuffle()) {
  sample_only_.store(this->auto_mode_ && this->phase_ == TRAIN);
  init_offsets();
}

template<typename Ftype, typename Btype>
void
DataLayer<Ftype, Btype>::init_offsets() {
  CHECK_EQ(this->transf_num_, this->threads_num());
  CHECK_LE(parser_offsets_.size(), this->transf_num_);
  CHECK_LE(queue_ids_.size(), this->transf_num_);
  parser_offsets_.resize(this->transf_num_);
  queue_ids_.resize(this->transf_num_);
  for (size_t i = 0; i < this->transf_num_; ++i) {
    parser_offsets_[i] = 0;
    queue_ids_[i] = i * this->parsers_num_;
  }
}

template<typename Ftype, typename Btype>
DataLayer<Ftype, Btype>::~DataLayer() {
  if (layer_inititialized_flag_.is_set()) {
    this->StopInternalThread();
  }
}

template<typename Ftype, typename Btype>
void
DataLayer<Ftype, Btype>::InitializePrefetch() {
  std::lock_guard<std::mutex> lock(mutex_prefetch_);
  if (layer_inititialized_flag_.is_set()) {
    return;
  }
  if (this->auto_mode()) {
    this->AllocatePrefetch();
    P2PManager::dl_bar_wait();
    // Here we try to optimize memory split between prefetching and convolution.
    // All data and parameter blobs are allocated at this moment.
    // Now let's find out what's left...
    size_t current_parsers_num = this->parsers_num_;
    size_t current_transf_num = this->threads_num();
#ifndef CPU_ONLY
    const size_t batch_bytes = this->prefetch_[0]->bytes(this->is_gpu_transform());
    size_t gpu_bytes, total_memory;
    GPUMemory::GetInfo(&gpu_bytes, &total_memory, true);
    gpu_bytes = Caffe::min_avail_device_memory();
    size_t batches_fit = gpu_bytes / batch_bytes;
#else
    size_t batches_fit = this->queues_num_;
#endif
    size_t max_parsers_num = 2;
    size_t max_transf_num = 3;
    float ratio = 5.F;
    Net* pnet = this->parent_net();
    if (pnet != nullptr) {
      Solver* psolver = pnet->parent_solver();
      if (psolver != nullptr) {
        if (pnet->layers().size() < 100) {
          max_transf_num = 4;
          ratio = 2.F; // 1:2 for "i/o bound", 1:5 otherwise
        }
      }
    }
    const float fit = std::min(float(max_parsers_num * max_transf_num),
        std::floor(batches_fit / ratio));
    current_parsers_num = std::min(max_parsers_num, std::max(1UL,
        static_cast<size_t>(std::sqrt(fit))));
    if (cache_ && current_parsers_num > 1UL) {
      LOG(INFO) << this->print_current_device() << " Reduced parser threads count from "
                << current_parsers_num << " to 1 because cache is used";
      current_parsers_num = 1UL;
    }
    current_transf_num = std::min(max_transf_num, std::max(current_transf_num,
        static_cast<size_t>(std::lround(fit / current_parsers_num))));
    this->RestartAllThreads(current_transf_num, true, false, Caffe::next_seed());
    this->transf_num_ = this->threads_num();
    this->parsers_num_ = current_parsers_num;
    this->queues_num_ = this->transf_num_ * this->parsers_num_;
    BasePrefetchingDataLayer<Ftype, Btype>::InitializePrefetch();
    if (current_transf_num > 1) {
      this->next_batch_queue();  // 0th already processed
    }
    if (this->parsers_num_ > 1) {
      parser_offsets_[0]++;  // same as above
    }
    this->go();  // kick off new threads if any
  }

  CHECK_EQ(this->threads_num(), this->transf_num_);
  LOG(INFO) << this->print_current_device() << " Parser threads: "
      << this->parsers_num_ << (this->auto_mode_ ? " (auto)" : "");
  LOG(INFO) << this->print_current_device() << " Transformer threads: "
      << this->transf_num_ << (this->auto_mode_ ? " (auto)" : "");
  layer_inititialized_flag_.set();
}

template<typename Ftype, typename Btype>
size_t DataLayer<Ftype, Btype>::queue_id(size_t thread_id) const {
  const size_t qid = queue_ids_[thread_id] + parser_offsets_[thread_id];
  parser_offsets_[thread_id]++;
  if (parser_offsets_[thread_id] >= this->parsers_num_) {
    parser_offsets_[thread_id] = 0UL;
    queue_ids_[thread_id] += this->parsers_num_ * this->threads_num();
  }
  return qid % this->queues_num_;
};

template<typename Ftype, typename Btype>
void
DataLayer<Ftype, Btype>::DataLayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top) {
  std::lock_guard<std::mutex> lock(mutex_setup_);
  const LayerParameter& param = this->layer_param();
  const int batch_size = param.data_param().batch_size();
  const bool use_gpu_transform = this->is_gpu_transform();
  const bool cache = cache_ && this->phase_ == TRAIN;
  const bool shuffle = cache && shuffle_ && this->phase_ == TRAIN;

  if (this->auto_mode_) {
    if (!sample_reader_) {
      sample_reader_ = make_shared<DataReader<Datum>>(param,
          Caffe::solver_count(),
          this->solver_rank_,
          this->parsers_num_,
          this->threads_num(),
          batch_size,
          true,
          false,
          cache,
          shuffle);
    } else if (!reader_) {
      reader_ = make_shared<DataReader<Datum>>(param,
          Caffe::solver_count(),
          this->solver_rank_,
          this->parsers_num_,
          this->threads_num(),
          batch_size,
          false,
          true,
          cache,
          shuffle);
    }
  } else if (!reader_) {
    reader_ = make_shared<DataReader<Datum>>(param,
        Caffe::solver_count(),
        this->solver_rank_,
        this->parsers_num_,
        this->threads_num(),
        batch_size,
        false,
        false,
        cache,
        shuffle);
    start_reading();
  }
  // Read a data point, and use it to initialize the top blob.
  shared_ptr<Datum> sample_datum = sample_only_ ? sample_reader_->sample() : reader_->sample();
  init_offsets();

  // Calculate the variable sized transformed datum shape.
  vector<int> sample_datum_shape = this->data_transformers_[0]->InferDatumShape(*sample_datum);
#ifdef USE_OPENCV
  if (this->data_transformers_[0]->var_sized_transforms_enabled()) {
    sample_datum_shape =
        this->data_transformers_[0]->var_sized_transforms_shape(sample_datum_shape);
  }
#endif

  // Reshape top[0] and prefetch_data according to the batch_size.
  // Note: all these reshapings here in load_batch are needed only in case of
  // different datum shapes coming from database.
  vector<int> top_shape = this->data_transformers_[0]->InferBlobShape(sample_datum_shape);
  top_shape[0] = batch_size;
  top[0]->Reshape(top_shape);

  vector<int> random_vec_shape(1, batch_size * 3);
  LOG(INFO) << this->print_current_device() << " ReshapePrefetch "
      << top_shape[0] << ", "
      << top_shape[1] << ", "
      << top_shape[2] << ", "
      << top_shape[3];
  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->data_.Reshape(top_shape);
    if (use_gpu_transform) {
      this->prefetch_[i]->gpu_transformed_data_->Reshape(top_shape);
      this->prefetch_[i]->random_vec_.Reshape(random_vec_shape);
    }
  }
  if (use_gpu_transform) {
    LOG(INFO) << this->print_current_device() << " Transform on GPU enabled";
  }
  // label
  vector<int> label_shape(1, batch_size);
  if (this->output_labels_) {
    vector<int> label_shape(1, batch_size);
    top[1]->Reshape(label_shape);
    for (int i = 0; i < this->prefetch_.size(); ++i) {
      this->prefetch_[i]->label_.Reshape(label_shape);
    }
  }
  LOG(INFO) << this->print_current_device() << " Output data size: "
      << top[0]->num() << ", "
      << top[0]->channels() << ", "
      << top[0]->height() << ", "
      << top[0]->width();
}

template<typename Ftype, typename Btype>
void DataLayer<Ftype, Btype>::load_batch(Batch<Ftype>* batch, int thread_id, size_t queue_id) {
  const bool sample_only = sample_only_.load();
  if (!sample_only && !reader_) {
    this->DataLayerSetUp(this->bottom_init_, this->top_init_);
  }
  const bool use_gpu_transform = this->is_gpu_transform();
  // Reshape according to the first datum of each batch
  // on single input batches allows for inputs of varying dimension.
  const int batch_size = this->layer_param_.data_param().batch_size();

  const size_t qid = sample_only ? 0UL : queue_id;
  DataReader<Datum>* reader = sample_only ? sample_reader_.get() : reader_.get();
  shared_ptr<Datum> init_datum = reader->full_peek(qid);
  CHECK(init_datum);

  // Calculate the variable sized transformed datum shape.
  vector<int> datum_shape = this->data_transformers_[thread_id]->InferDatumShape(*init_datum);
#ifdef USE_OPENCV
  if (this->data_transformers_[thread_id]->var_sized_transforms_enabled()) {
    datum_shape = this->data_transformers_[thread_id]->var_sized_transforms_shape(datum_shape);
  }
#endif

  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_shape = this->data_transformers_[thread_id]->InferBlobShape(datum_shape,
      use_gpu_transform);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);
  if (use_gpu_transform) {
    top_shape = this->data_transformers_[thread_id]->InferBlobShape(datum_shape, false);
    top_shape[0] = batch_size;
    batch->gpu_transformed_data_->Reshape(top_shape);
    batch->gpu_transformed_data_->allocate_data();
  }

  size_t out_sizeof_element = 0;
  const bool copy_to_cpu = init_datum->encoded() || !use_gpu_transform;
  Ftype* top_data = nullptr;
  if (copy_to_cpu) {
    top_data = batch->data_.mutable_cpu_data();
  } else {
#ifndef CPU_ONLY
    top_data = batch->data_.mutable_gpu_data();
#else
    NO_GPU;
#endif
  }
  Ftype* top_label = nullptr;
  if (this->output_labels_) {
    vector<int> label_shape(1, batch_size);
    batch->label_.Reshape(label_shape);
    top_label = batch->label_.mutable_cpu_data();
  }
  vector<int> random_vec_shape_(1, batch_size * 3);
  batch->random_vec_.Reshape(random_vec_shape_);
  size_t current_batch_id = 0UL;
  size_t item_id;
  for (size_t entry = 0; entry < batch_size; ++entry) {
    shared_ptr<Datum> datum = reader->full_pop(qid, "Waiting for datum");
#ifdef USE_OPENCV
    // Apply variable-sized transforms.
    if (this->data_transformers_[thread_id]->var_sized_transforms_enabled()) {
      this->data_transformers_[thread_id]->VariableSizedTransforms(datum.get());
    }
#endif
    item_id = datum->record_id() % batch_size;
    if (datum->channels() > 0) {
      CHECK_EQ(top_shape[1], datum->channels())
        << "Number of channels can't vary in the same batch";
    }
    if (!this->data_transformers_[thread_id]->transform_param().has_crop_size()) {
      if (datum->height() > 0) {
        CHECK_EQ(top_shape[2], datum->height())
          << "Image height can't vary in the same batch (crop might help here)";
      }
      if (datum->width() > 0) {
        CHECK_EQ(top_shape[3], datum->width())
          << "Image width can't vary in the same batch (crop might help here)";
      }
    }
    if (item_id == 0UL) {
      current_batch_id = datum->record_id() / batch_size;
    }
    // Copy label.
    Ftype* label_ptr = NULL;
    if (this->output_labels_) {
      label_ptr = &top_label[item_id];
    }
    // Get data offset for this datum to hand off to transform thread
    const size_t offset = batch->data_.offset(item_id);
    Ftype* ptr = top_data + offset;

    if (use_gpu_transform) {
      // store the generated random numbers and enqueue the copy
      this->data_transformers_[thread_id]->Fill3Randoms(
          &batch->random_vec_.mutable_cpu_data()[item_id * 3]);
      this->data_transformers_[thread_id]->CopyPtrEntry(datum, ptr, out_sizeof_element,
          this->output_labels_, label_ptr);
    } else {
      // Precalculate the necessary random draws so that they are
      // drawn deterministically
      std::array<unsigned int, 3> rand;
      this->data_transformers_[thread_id]->Fill3Randoms(&rand.front());
      this->data_transformers_[thread_id]->TransformPtrEntry(datum, ptr, rand,
          this->output_labels_,
          label_ptr);
    }
    reader->free_push(qid, datum);
  }

  if (use_gpu_transform) {
#ifndef CPU_ONLY
    cudaStream_t stream = Caffe::th_stream_aux(Caffe::STREAM_ID_TRANSFORMER);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    this->data_transformers_[thread_id]->TransformGPU(top_shape[0], top_shape[1],
        batch->data_.shape(2),  // non-crop
        batch->data_.shape(3),  // non-crop
        out_sizeof_element, batch->data_.gpu_data(),
        batch->gpu_transformed_data_->template mutable_gpu_data<Ftype>(),
        batch->random_vec_.gpu_data());
#else
    NO_GPU;
#endif
  }
  batch->set_id(current_batch_id);
  sample_only_.store(false);
}

INSTANTIATE_CLASS_FB(DataLayer);
REGISTER_LAYER_CLASS(Data);

}  // namespace caffe
