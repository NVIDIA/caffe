#include <boost/thread.hpp>
#include <sys/sysinfo.h>

#include "caffe/util/rng.hpp"
#include "caffe/parallel.hpp"
#include "caffe/data_reader.hpp"

#ifdef USE_MPI
#include "caffe/clusters.hpp"
#endif

namespace caffe {

template<typename DatumType>
std::mutex DataReader<DatumType>::db_mutex_;
// https://stackoverflow.com/questions/26935824/gcc-gives-an-undefined-reference-error-to-static-data-members-in-templated-cla

template<typename DatumType>
std::mutex DataReader<DatumType>::DataCache::cache_mutex_{};

template<typename DatumType>
DataReader<DatumType>::DataReader(const LayerParameter& param,
    size_t solver_count,
    size_t solver_rank,
    size_t parser_threads_num,
    size_t transf_threads_num,
    size_t queue_depth,
    bool sample_only,
    bool skip_one_batch,
    bool cache,
    bool shuffle,
    bool epoch_count_required)
    : InternalThread(Caffe::current_device(),
          solver_rank, sample_only ? 1U : parser_threads_num, false),
      parser_threads_num_(threads_num()),
      transf_threads_num_(sample_only ? 1U : transf_threads_num),
      queues_num_(parser_threads_num_ * transf_threads_num_),
      queue_depth_(queue_depth),
      solver_count_(solver_count),
      solver_rank_(solver_rank),
      skip_one_batch_(skip_one_batch),
      current_rec_(0),
      current_queue_(0),
      sample_only_(sample_only),
      cache_(cache && !sample_only),
      shuffle_(cache_ && shuffle),
      epoch_count_required_(epoch_count_required) {
  CHECK(queues_num_);
  CHECK(queue_depth_);
  batch_size_ = param.data_param().batch_size();
  backend_ = param.data_param().backend();
  if (backend_ == DataParameter_DB_LEVELDB) {
    CHECK_EQ(parser_threads_num_, 1) << "LevelDB doesn't support multiple connections";
  }
  if (cache_) {
    // This is singleton, we cache TRAIN db only
    data_cache_ = DataCache::data_cache_inst(parser_threads_num_ * solver_count_, shuffle_);
  }

  free_.resize(queues_num_);
  full_.resize(queues_num_);
  LOG(INFO) << (sample_only ? "Sample " : "") << "Data Reader threads: "
      << this->threads_num() << ", out queues: " << queues_num_ << ", depth: " << queue_depth_;
  for (size_t i = 0; i < queues_num_; ++i) {
    full_[i] = make_shared<BlockingQueue<shared_ptr<DatumType>>>();
    free_[i] = make_shared<BlockingQueue<shared_ptr<DatumType>>>();
    for (size_t j = 0; j < queue_depth_; ++j) {
      free_[i]->push(make_shared<DatumType>());
    }
  }
  db_source_ = param.data_param().source();
  init_ = make_shared<BlockingQueue<shared_ptr<DatumType>>>();
  StartInternalThread(false, Caffe::next_seed());
}

template<typename DatumType>
DataReader<DatumType>::~DataReader() {
  StopInternalThread();
}

template<typename DatumType>
void DataReader<DatumType>::InternalThreadEntry() {
  InternalThreadEntryN(0U);
}

template<typename DatumType>
void DataReader<DatumType>::InternalThreadEntryN(size_t thread_id) {
  if (cache_) {
    data_cache_->check_db(db_source_);
    data_cache_->register_new_thread();
  }

  unique_ptr<db::DB> db;
  {
    std::lock_guard<std::mutex> lock(db_mutex_);
    db.reset(db::GetDB(backend_));
    db->Open(db_source_, db::READ);
  }

  CursorManager cm(db.get(),
      this,
      solver_count_,
      solver_rank_,
      parser_threads_num_,
      thread_id,
      batch_size_,
      cache_ && !sample_only_,
      shuffle_ && !sample_only_,
      epoch_count_required_);
  shared_ptr<DatumType> init_datum = make_shared<DatumType>();
  cm.fetch(init_datum.get());
  init_->push(init_datum);

  if (!sample_only_) {
    start_reading_flag_.wait();
  }
  cm.rewind();
  size_t skip = skip_one_batch_ ? batch_size_ : 0UL;

  size_t queue_id, ranked_rec, batch_on_solver, sample_count = 0UL;
  shared_ptr<DatumType> datum = make_shared<DatumType>();
  try {
    while (!must_stop(thread_id)) {
      cm.next(datum);
      // See comment below
      ranked_rec = (size_t) datum->record_id() / cm.full_cycle();
      batch_on_solver = ranked_rec * parser_threads_num_ + thread_id;
      queue_id = batch_on_solver % queues_num_;

      if (thread_id == 0 && skip > 0U) {
        --skip;
        continue;
      }

      full_push(queue_id, datum);

      if (sample_only_) {
        ++sample_count;
        if (sample_count >= batch_size_) {
          // sample batch complete
          break;
        }
      }
      datum = free_pop(queue_id);
    }
  } catch (boost::thread_interrupted&) {
  }
}

template<typename DatumType>
shared_ptr<DatumType>& DataReader<DatumType>::DataCache::next_new() {
  std::lock_guard<std::mutex> lock(cache_mutex_);
  cache_buffer_.emplace_back(make_shared<DatumType>());
  return cache_buffer_.back();
}

template<typename DatumType>
shared_ptr<DatumType>& DataReader<DatumType>::DataCache::next_cached(DataReader& reader) {
  if (just_cached_.load()) {
    cache_bar_.wait();
    just_cached_.store(false);
    LOG_FIRST_N(INFO, 1) << "Cached " << cache_buffer_.size() << " records by "
          << cached_flags_.size() << " threads";
//#ifdef DEBUG
//    {
//      std::lock_guard<std::mutex> lock(cache_mutex_);
//      std::multiset<size_t> pk;
//      for (auto &entry : cache_buffer_) {
//        pk.insert(entry->record_id());
//        if (pk.count(entry->record_id()) > 1) {
//          LOG(ERROR) << "Record " << entry->record_id() << " duplicated "
//              << entry->record_id() << " times";
//        }
//      }
//      LOG(INFO) << "Recorded " << pk.size() << " from " << *pk.begin() << " to " << *pk.rbegin();
//    }
//#endif
    cache_bar_.wait();
  }
  std::lock_guard<std::mutex> lock(cache_mutex_);
  if (shuffle_ && cache_idx_== 0UL) {
    LOG(INFO) << "Shuffling " << cache_buffer_.size() << " records...";
    caffe::shuffle(cache_buffer_.begin(), cache_buffer_.end());
  }
  shared_ptr<DatumType>& datum = cache_buffer_[cache_idx_++];
  if (cache_idx_ >= cache_buffer_.size()) {
    cache_idx_= 0UL;
  }
  return datum;
}

template<typename DatumType>
void DataReader<DatumType>::DataCache::just_cached() {
  just_cached_.store(true);
  cached_flags_[lwp_id()]->set();
}

template<typename DatumType>
bool DataReader<DatumType>::DataCache::check_memory() {
#ifdef __APPLE__
  return true;
#else
  if (cache_buffer_.size() == 0UL || cache_buffer_.size() % 1000UL != 0UL) {
    return true;
  }
  std::lock_guard<std::mutex> lock(cache_mutex_);
  bool mem_ok = true;
  struct sysinfo sinfo;
  sysinfo(&sinfo);
  if (sinfo.totalswap > 0UL && sinfo.freeswap < sinfo.totalswap / 2UL) {
    LOG_FIRST_N(WARNING, 1) << "Data Reader cached " << cache_buffer_.size()
        << " records so far but it can't continue because it used more than half"
        << " of swap buffer. Free swap memory left: " << sinfo.freeswap << " of total "
        << sinfo.totalswap << ". Cache and shuffling are now disabled.";
    mem_ok = false;
  }
//  else {
//    unsigned long ram_avail = 0UL;  // NOLINT(runtime/int)
//    char buf[128];
//    char *e;
//    FILE *fp = fopen("/proc/meminfo", "r");
//    while (fgets(buf, sizeof(buf) - 1, fp) != nullptr) {
//      if (strstr(buf, "vailable") != nullptr) {
//        char *p = strchr(buf, ':');
//        if (p != nullptr) {
//          ++p;
//          ram_avail = strtoull(p, &e, 10) * 1024UL;
//          break;
//        }
//      }
//      if (feof(fp)) {
//        break;
//      }
//    }
//    fclose(fp);
//    if (ram_avail == 0UL) {
//      // 2nd attempt
//      ram_avail = sinfo.freeram + sinfo.bufferram + sinfo.sharedram;
//    }
//    if (sinfo.totalswap == 0UL && ram_avail < sinfo.totalram / 50UL) {
//      LOG_FIRST_N(WARNING, 1) << "Data Reader cached " << cache_buffer_.size()
//          << " records so far but it can't continue because it used more than 98%"
//          << " of RAM and there is no swap space available. RAM available: "
//          << ram_avail << " of total " << sinfo.totalram
//          << ". Cache and shuffling are now disabled.";
//      mem_ok = false;
//    }
//  }
  if (!mem_ok) {
    cache_buffer_.clear();
    shuffle_ = false;
  }
  return mem_ok;
#endif
}

size_t cycle_count(size_t rank_cycle, size_t solver_count) {
#ifdef USE_MPI
return rank_cycle * solver_count * Clusters::node_count();
#else
return rank_cycle * solver_count;
#endif
}

template<typename DatumType>
DataReader<DatumType>::CursorManager::CursorManager(db::DB* db, DataReader<DatumType>* reader,
    size_t solver_count, size_t solver_rank, size_t parser_threads, size_t parser_thread_id,
    size_t batch_size, bool cache, bool shuffle, bool epoch_count_required)
    : db_(db),
      cursor_(db->NewCursor()),
      reader_(reader),
      solver_count_(solver_count),
      solver_rank_(solver_rank),
      batch_size_(batch_size),
      parser_threads_(parser_threads),
      parser_thread_id_(parser_thread_id),
      rank_cycle_(parser_threads_ * batch_size_),
      full_cycle_(cycle_count(rank_cycle_,  solver_count_)),
      rec_id_(0UL),
      rec_end_(0UL),
      cache_(cache),
      shuffle_(shuffle),
      cached_all_(false),
      epoch_count_(0UL),
      epoch_count_required_(epoch_count_required) {}

template<typename DatumType>
DataReader<DatumType>::CursorManager::~CursorManager() {
  cursor_.reset();
  db_->Close();
}

template<typename DatumType>
void DataReader<DatumType>::CursorManager::next(shared_ptr<DatumType>& datum) {
  if (cached_all_) {
    datum = reader_->next_cached();
  } else {
    while (cache_) {
      if (!reader_->check_memory()) {
        cache_ = false;
        shuffle_ = false;
        break;
      }
      datum = reader_->next_new();
      break;
    }
    fetch(datum.get());
  }

  datum->set_record_id(rec_id_);
  size_t old_id = rec_id_;
  ++rec_id_;
  if (rec_id_ == rec_end_) {
    rec_id_ += full_cycle_ - batch_size_;
    rec_end_ += full_cycle_;
  }
  if (cached_all_) {
    return;
  }
  for (size_t i = old_id; i < rec_id_; ++i) {
    cursor_->Next();
    if (!cursor_->valid()) {
      if (epoch_count_required_ && epoch_count_ == 0UL) {  // only once if required
        epoch_count_ = rec_id_;
        Caffe::report_epoch_count(epoch_count_);
      }
      if (cache_) {
        cached_all_ = true;
        reader_->just_cached();
        break;  // we cache first epoch, then we just read it from cache
      }
      LOG_IF(INFO, solver_rank_ == 0 && parser_thread_id_ == 0) << "Restarting data pre-fetching";
      cursor_->SeekToFirst();
    }
  }
}

/*
  Example: 2 solvers (rank 0, rank 1), 3 parser threads per solver (pt0, pt1, pt2),
           2 transformer threads per solver (tr0, tr1) - each transformer owns queue set
           with number of queues equal to the number of parser threads)

        B0    B1    B2    B3    B4    B5    B6    B7
      --------------------------------------------------  --> S0.tr0 S0.q0
   |                                                      --> S0.tr0 S0.q1
   |  r0pt0.q0                            r0pt0.q3        --> S0.tr0 S0.q2
S0 |        r0pt1.q1                            r0pt1.q4  --> S0.tr1 S0.q3
   |              r0pt2.q2                                --> S0.tr1 S0.q4
   |                                                      --> S0.tr1 S0.q5
      ..................................................
   |                                                      --> S1.tr0 S1.q0
   |                    r1pt0.q0                          --> S1.tr0 S1.q1
S1 |                          r1pt1.q1                    --> S1.tr0 S1.q2
   |                                r1pt2.q2              --> S1.tr1 S1.q3
   |                                                      --> S1.tr1 S1.q4
      --------------------------------------------------  --> S1.tr1 S1.q5
      <-- rank cycle ->
      <---------- full cycle ----------->
*/
template<typename DatumType>
void DataReader<DatumType>::CursorManager::rewind() {
  CHECK(parser_threads_);
#ifdef USE_MPI
  size_t rank_cycle_per_solver = parser_threads_ * batch_size_;
  size_t rank_cycle_per_node = rank_cycle_per_solver * solver_count_;
  size_t rank_cycle_begin = rank_cycle_per_solver * solver_rank_
                        + rank_cycle_per_node * Clusters::node_rank();
#else
  size_t rank_cycle_begin = rank_cycle_ * solver_rank_;
#endif
  rec_id_ = rank_cycle_begin + parser_thread_id_ * batch_size_;
  rec_end_ = rec_id_ + batch_size_;
  cursor_->SeekToFirst();
  for (size_t i = 0; i < rec_id_; ++i) {
    cursor_->Next();
    if (!cursor_->valid()) {
      cursor_->SeekToFirst();
    }
  }
}

template<>
void DataReader<Datum>::CursorManager::fetch(Datum* datum) {
  C2TensorProtos protos;
  if (cursor_->parse(&protos) && protos.protos_size() >= 2) {
    C2TensorProto* image_proto = protos.mutable_protos(0);
    C2TensorProto* label_proto = protos.mutable_protos(1);
    if (image_proto->data_type() == C2TensorProto::STRING) {
      // encoded image string.
      DCHECK_EQ(image_proto->string_data_size(), 1);
      datum->mutable_data()->assign(image_proto->string_data(0));
      datum->set_encoded(true);
    } else if (image_proto->data_type() == C2TensorProto::BYTE) {
      // raw image content.
      datum->set_allocated_data(image_proto->release_byte_data());
      datum->set_encoded(false);
      datum->set_channels(image_proto->dims_size() == 3 ? image_proto->dims(2) : 1);
      datum->set_height(image_proto->dims_size() > 1 ? image_proto->dims(0) : 0);
      datum->set_width(image_proto->dims_size() > 1 ? image_proto->dims(1) : 0);
    } else {
      LOG(FATAL) << "Unknown C2 image data type.";
    }
    if (label_proto->data_type() == C2TensorProto::INT32) {
      DCHECK_EQ(label_proto->int32_data_size(), 1);
      datum->set_label(label_proto->int32_data(0));
    } else {
      LOG(FATAL) << "Unsupported C2 label data type.";
    }
  } else if (!cursor_->parse(datum)) {
    LOG(ERROR) << "Database cursor failed to parse Datum record";
  }
  // DLOG(INFO) << cursor_->key() << " " << datum->label();
}

template<>
void DataReader<AnnotatedDatum>::CursorManager::fetch(AnnotatedDatum* datum) {
  if (!cursor_->parse(datum)) {
    LOG(ERROR) << "Database cursor failed to parse Datum record";
  }
}

template class DataReader<Datum>;
template class DataReader<AnnotatedDatum>;

}  // namespace caffe
