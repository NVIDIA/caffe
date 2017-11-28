#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <algorithm>
#include <map>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/annotated_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/sampler.hpp"
#include "caffe/parallel.hpp"

namespace caffe {

template <typename Ftype, typename Btype>
AnnotatedDataLayer<Ftype, Btype>::AnnotatedDataLayer(const LayerParameter& param)
  : BasePrefetchingDataLayer<Ftype,Btype>(param),
      cache_(param.data_param().cache()),
    shuffle_(param.data_param().shuffle()) {
  sample_only_.store(this->auto_mode() && this->phase_ == TRAIN);
  init_offsets();
}

template<typename Ftype, typename Btype>
void
AnnotatedDataLayer<Ftype, Btype>::init_offsets() {
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

template <typename Ftype, typename Btype>
AnnotatedDataLayer<Ftype, Btype>::~AnnotatedDataLayer() {
  if (layer_inititialized_flag_.is_set()) {
    this->StopInternalThread();
  }
}

template<typename Ftype, typename Btype>
void
AnnotatedDataLayer<Ftype, Btype>::InitializePrefetch() {
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
      << this->parsers_num_ << (this->auto_mode() ? " (auto)" : "");
  LOG(INFO) << this->print_current_device() << " Transformer threads: "
      << this->transf_num_ << (this->auto_mode() ? " (auto)" : "");
  layer_inititialized_flag_.set();
}

template<typename Ftype, typename Btype>
size_t AnnotatedDataLayer<Ftype, Btype>::queue_id(size_t thread_id) const {
  const size_t qid = queue_ids_[thread_id] + parser_offsets_[thread_id];
  parser_offsets_[thread_id]++;
  if (parser_offsets_[thread_id] >= this->parsers_num_) {
    parser_offsets_[thread_id] = 0UL;
    queue_ids_[thread_id] += this->parsers_num_ * this->threads_num();
  }
  return qid % this->queues_num_;
};

template <typename Ftype, typename Btype>
void AnnotatedDataLayer<Ftype, Btype>::DataLayerSetUp(
    const vector<Blob*>& bottom, const vector<Blob*>& top) {
  std::lock_guard<std::mutex> lock(mutex_setup_);
  const LayerParameter& param = this->layer_param();
  const AnnotatedDataParameter& anno_data_param = param.annotated_data_param();
  const int batch_size = param.data_param().batch_size();
  //const bool use_gpu_transform = this->is_gpu_transform();
  const bool cache = this->cache_ && this->phase_ == TRAIN;
  const bool shuffle = cache && this->shuffle_ && this->phase_ == TRAIN;
  TBlob<Ftype> transformed_datum;
  
  for (int i = 0; i < anno_data_param.batch_sampler_size(); ++i) {
    batch_samplers_.push_back(anno_data_param.batch_sampler(i));
  }
  
  if (this->auto_mode()) {
    if (!sample_reader_) {
	  sample_reader_ = make_shared<DataReader<AnnotatedDatum>>(param,
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
	  reader_ = make_shared<DataReader<AnnotatedDatum>>(param,
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
    reader_ = make_shared<DataReader<AnnotatedDatum>>(param,
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
 
  label_map_file_ = anno_data_param.label_map_file();
  // Make sure dimension is consistent within batch.
  const TransformationParameter& transform_param =
    this->layer_param_.transform_param();
  if (transform_param.has_resize_param()) {
    if (transform_param.resize_param().resize_mode() ==
        ResizeParameter_Resize_mode_FIT_SMALL_SIZE) {
      CHECK_EQ(batch_size, 1)
        << "Only support batch size of 1 for FIT_SMALL_SIZE.";
    }
  }
    		
  // Read a data point, and use it to initialize the top blob.
  shared_ptr<AnnotatedDatum> sample_datum = this->sample_only_ ? this->sample_reader_->sample() : this->reader_->sample();
  AnnotatedDatum& anno_datum = *sample_datum;
  this->init_offsets();

  // Calculate the variable sized transformed datum shape.
  vector<int> sample_datum_shape = this->data_transformers_[0]->InferDatumShape(sample_datum->datum());
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
  transformed_datum.Reshape(top_shape);   	  
  top_shape[0] = batch_size;
  top[0]->Reshape(top_shape);
  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->data_.Reshape(top_shape);
  }
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  if (this->output_labels_) {
    has_anno_type_ = anno_datum.has_type() || anno_data_param.has_anno_type();
    vector<int> label_shape(4, 1);
    if (has_anno_type_) {
      anno_type_ = anno_datum.type();
      if (anno_data_param.has_anno_type()) {
        // If anno_type is provided in AnnotatedDataParameter, replace
        // the type stored in each individual AnnotatedDatum.
        LOG(WARNING) << "type stored in AnnotatedDatum is shadowed.";
        anno_type_ = anno_data_param.anno_type();
      }
      // Infer the label shape from anno_datum.AnnotationGroup().
      int num_bboxes = 0;
      if (anno_type_ == AnnotatedDatum_AnnotationType_BBOX) {
        // Since the number of bboxes can be different for each image,
        // we store the bbox information in a specific format. In specific:
        // All bboxes are stored in one spatial plane (num and channels are 1)
        // And each row contains one and only one box in the following format:
        // [item_id, group_label, instance_id, xmin, ymin, xmax, ymax, diff]
        // Note: Refer to caffe.proto for details about group_label and
        // instance_id.
        for (int g = 0; g < anno_datum.annotation_group_size(); ++g) {
          num_bboxes += anno_datum.annotation_group(g).annotation_size();
        }
        label_shape[0] = 1;
        label_shape[1] = 1;
        // BasePrefetchingDataLayer<Dtype>::LayerSetUp() requires to call
        // cpu_data and gpu_data for consistent prefetch thread. Thus we make
        // sure there is at least one bbox.
        label_shape[2] = std::max(num_bboxes, 1);
        label_shape[3] = 8;
      } else {
        LOG(FATAL) << "Unknown annotation type.";
      }
    } else {
      label_shape[0] = batch_size;
    }
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

// This function is called on prefetch thread
template <typename Ftype, typename Btype>
void AnnotatedDataLayer<Ftype, Btype>::load_batch(Batch<Dtype>* batch, int thread_id, size_t queue_id) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());

  const bool sample_only = sample_only_.load();
  TBlob<Ftype> transformed_datum;

  //const bool use_gpu_transform = false;//this->is_gpu_transform();
  // Reshape according to the first anno_datum of each batch
  // on single input batches allows for inputs of varying dimension.
  const int batch_size = this->layer_param_.data_param().batch_size();
  const AnnotatedDataParameter& anno_data_param =
      this->layer_param_.annotated_data_param();
  const TransformationParameter& transform_param =
    this->layer_param_.transform_param();

  const size_t qid = sample_only ? 0UL : queue_id;
  DataReader<AnnotatedDatum>* reader = sample_only ? sample_reader_.get() : reader_.get();
  shared_ptr<AnnotatedDatum> init_datum = reader->full_peek(qid);
  CHECK(init_datum);
 
  // Calculate the variable sized transformed datum shape.
  vector<int> datum_shape = this->data_transformers_[thread_id]->InferDatumShape(init_datum->datum());
#ifdef USE_OPENCV
  if (this->data_transformers_[thread_id]->var_sized_transforms_enabled()) {
    datum_shape = this->data_transformers_[thread_id]->var_sized_transforms_shape(datum_shape);
  }
#endif
    
  // Use data_transformer to infer the expected blob shape from datum.
  //vector<int> top_shape = this->data_transformers_[thread_id]->InferBlobShape(datum_shape,
  //    use_gpu_transform);  
  vector<int> top_shape =
      this->data_transformers_[thread_id]->InferBlobShape(init_datum->datum());
  transformed_datum.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);

  Dtype* top_data = batch->data_.mutable_cpu_data();
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables
  if (this->output_labels_ && !has_anno_type_) {
    top_label = batch->label_.mutable_cpu_data();
  }

  // Store transformed annotation.
  map<int, vector<AnnotationGroup> > all_anno;
  int num_bboxes = 0;

  for (int item_id = 0; item_id < batch_size; ++item_id) {
    timer.Start();
    // get a anno_datum
    shared_ptr<AnnotatedDatum> anno_datum = reader_->full_pop(queue_id, "Waiting for data");
    read_time += timer.MicroSeconds();
    timer.Start();
    AnnotatedDatum distort_datum;
    AnnotatedDatum* expand_datum = NULL;
    if (transform_param.has_distort_param()) {
      distort_datum.CopyFrom(*anno_datum);
      this->data_transformers_[thread_id]->DistortImage(anno_datum->datum(),
                                            distort_datum.mutable_datum());
      if (transform_param.has_expand_param()) {
        expand_datum = new AnnotatedDatum();
        this->data_transformers_[thread_id]->ExpandImage(distort_datum, expand_datum);
      } else {
        expand_datum = &distort_datum;
      }
    } else {
      if (transform_param.has_expand_param()) {
        expand_datum = new AnnotatedDatum();
        this->data_transformers_[thread_id]->ExpandImage(*anno_datum, expand_datum);
      } else {
        expand_datum = &(*anno_datum);
      }
    }
    AnnotatedDatum* sampled_datum = NULL;
    bool has_sampled = false;
    if (batch_samplers_.size() > 0) {
      // Generate sampled bboxes from expand_datum.
      vector<NormalizedBBox> sampled_bboxes;
      GenerateBatchSamples(*expand_datum, batch_samplers_, &sampled_bboxes);
      if (sampled_bboxes.size() > 0) {
        // Randomly pick a sampled bbox and crop the expand_datum.
        int rand_idx = caffe_rng_rand() % sampled_bboxes.size();
        sampled_datum = new AnnotatedDatum();
        this->data_transformers_[thread_id]->CropImage(*expand_datum,
                                           sampled_bboxes[rand_idx],
                                           sampled_datum);
        has_sampled = true;
      } else {
        sampled_datum = expand_datum;
      }
    } else {
      sampled_datum = expand_datum;
    }
    CHECK(sampled_datum != NULL);
    timer.Start();
    vector<int> shape =
        this->data_transformers_[thread_id]->InferBlobShape(sampled_datum->datum());
    if (transform_param.has_resize_param()) {
      if (transform_param.resize_param().resize_mode() ==
          ResizeParameter_Resize_mode_FIT_SMALL_SIZE) {
        transformed_datum.Reshape(shape);
        batch->data_.Reshape(shape);
        top_data = batch->data_.mutable_cpu_data();
      } else {
        CHECK(std::equal(top_shape.begin() + 1, top_shape.begin() + 4,
              shape.begin() + 1));
      }
    } else {
      CHECK(std::equal(top_shape.begin() + 1, top_shape.begin() + 4,
            shape.begin() + 1));
    }
    // Apply data transformations (mirror, scale, crop...)
    int offset = batch->data_.offset(item_id);
    transformed_datum.set_cpu_data(top_data + offset);
    vector<AnnotationGroup> transformed_anno_vec;
    if (this->output_labels_) {
      if (has_anno_type_) {
        // Make sure all data have same annotation type.
        CHECK(sampled_datum->has_type()) << "Some datum misses AnnotationType.";
        if (anno_data_param.has_anno_type()) {
          sampled_datum->set_type(anno_type_);
        } else {
          CHECK_EQ(anno_type_, sampled_datum->type()) <<
              "Different AnnotationType.";
        }
        // Transform datum and annotation_group at the same time
        transformed_anno_vec.clear();
        this->data_transformers_[thread_id]->Transform(*sampled_datum,
                                           &(transformed_datum),
                                           &transformed_anno_vec);
        if (anno_type_ == AnnotatedDatum_AnnotationType_BBOX) {
          // Count the number of bboxes.
          for (int g = 0; g < transformed_anno_vec.size(); ++g) {
            num_bboxes += transformed_anno_vec[g].annotation_size();
          }
        } else {
          LOG(FATAL) << "Unknown annotation type.";
        }
        all_anno[item_id] = transformed_anno_vec;
      } else {
        this->data_transformers_[thread_id]->Transform(sampled_datum->datum(),
                                           &(transformed_datum));
        // Otherwise, store the label from datum.
        CHECK(sampled_datum->datum().has_label()) << "Cannot find any label.";
        top_label[item_id] = sampled_datum->datum().label();
      }
    } else {
      this->data_transformers_[thread_id]->Transform(sampled_datum->datum(),
                                         &(transformed_datum));
    }
    // clear memory
    if (has_sampled) {
      delete sampled_datum;
    }
    if (transform_param.has_expand_param()) {
      delete expand_datum;
    }
    trans_time += timer.MicroSeconds();

    reader_->free_push(queue_id, anno_datum);
  }

  // Store "rich" annotation if needed.
  if (this->output_labels_ && has_anno_type_) {
    vector<int> label_shape(4);
    if (anno_type_ == AnnotatedDatum_AnnotationType_BBOX) {
      label_shape[0] = 1;
      label_shape[1] = 1;
      label_shape[3] = 8;
      if (num_bboxes == 0) {
        // Store all -1 in the label.
        label_shape[2] = 1;
        batch->label_.Reshape(label_shape);
        caffe_set<Dtype>(8, -1, batch->label_.mutable_cpu_data());
      } else {
        // Reshape the label and store the annotation.
        label_shape[2] = num_bboxes;
        batch->label_.Reshape(label_shape);
        top_label = batch->label_.mutable_cpu_data();
        int idx = 0;
        for (int item_id = 0; item_id < batch_size; ++item_id) {
          const vector<AnnotationGroup>& anno_vec = all_anno[item_id];
          for (int g = 0; g < anno_vec.size(); ++g) {
            const AnnotationGroup& anno_group = anno_vec[g];
            for (int a = 0; a < anno_group.annotation_size(); ++a) {
              const Annotation& anno = anno_group.annotation(a);
              const NormalizedBBox& bbox = anno.bbox();
              top_label[idx++] = item_id;
              top_label[idx++] = anno_group.group_label();
              top_label[idx++] = anno.instance_id();
              top_label[idx++] = bbox.xmin();
              top_label[idx++] = bbox.ymin();
              top_label[idx++] = bbox.xmax();
              top_label[idx++] = bbox.ymax();
              top_label[idx++] = bbox.difficult();
            }
          }
        }
      }
    } else {
      LOG(FATAL) << "Unknown annotation type.";
    }
  }
  timer.Stop();
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS_FB(AnnotatedDataLayer);
REGISTER_LAYER_CLASS(AnnotatedData);

}  // namespace caffe
