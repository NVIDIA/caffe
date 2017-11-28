#ifndef CAFFE_DATA_READER_HPP_
#define CAFFE_DATA_READER_HPP_

#include <algorithm>
#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/util/blocking_queue.hpp"
#include "caffe/util/db.hpp"
#include "caffe/util/thread_pool.hpp"

namespace caffe {

/**
 * @brief Reads data from a source to queues available to data layers.
 * Few reading threads are created per source, every record gets it's unique id
 * to allow deterministic ordering down the road. Data is distributed to solvers
 * in a round-robin way to keep parallel training deterministic.
 */
template <typename DatumType>
class DataReader : public InternalThread {
 private:
  class CursorManager {
    shared_ptr<db::DB> db_;
    unique_ptr<db::Cursor> cursor_;
    DataReader* reader_;
    const size_t solver_count_, solver_rank_, batch_size_;
    const size_t parser_threads_, parser_thread_id_;
    const size_t rank_cycle_, full_cycle_;
    size_t rec_id_, rec_end_;
    bool cache_, shuffle_;
    bool cached_all_;

   public:
    CursorManager(shared_ptr<db::DB> db, DataReader* reader, size_t solver_count,
        size_t solver_rank, size_t parser_threads, size_t parser_thread_id, size_t batch_size_,
        bool cache, bool shuffle);
    ~CursorManager();
    void next(shared_ptr<DatumType>& datum);
    void fetch(DatumType* datum);
    void rewind();

    size_t full_cycle() const {
      return full_cycle_;
    }

    DISABLE_COPY_MOVE_AND_ASSIGN(CursorManager);
  };

  class DataCache {
   public:
    static DataCache* data_cache_inst(size_t threads, bool shuffle) {
      if (!data_cache_inst_) {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        if (!data_cache_inst_) {
          data_cache_inst_.reset(new DataCache(threads, shuffle));
        }
      }
      return data_cache_inst_.get();
    }

    shared_ptr<DatumType>& next_new();
    shared_ptr<DatumType>& next_cached();
    bool check_memory();
    void check_db(const std::string& db_source) {
      std::lock_guard<std::mutex> lock(cache_mutex_);
      if (db_source_.empty()) {
        db_source_ = db_source;
      } else {
        CHECK_EQ(db_source_, db_source) << "Caching of two DB sources is not supported";
      }
    }

    void just_cached();
    void register_new_thread() {
      std::lock_guard<std::mutex> lock(cache_mutex_);
      cached_flags_.emplace(std::this_thread::get_id(), make_shared<Flag>());
    }

   private:
    DataCache(size_t threads, bool shuffle)
        : cache_idx_(0UL),
          cache_bar_(threads),
          shuffle_(shuffle),
          just_cached_(false) {}

    std::string db_source_;
    vector<shared_ptr<DatumType>> cache_buffer_;
    size_t cache_idx_;
    boost::barrier cache_bar_;
    bool shuffle_;
    std::atomic_bool just_cached_;
    std::unordered_map<std::thread::id, shared_ptr<Flag>> cached_flags_;
    static std::mutex cache_mutex_;
    static unique_ptr<DataCache> data_cache_inst_;
  };

 public:
  DataReader(const LayerParameter& param,
      size_t solver_count,
      size_t solver_rank,
      size_t parser_threads_num,
      size_t transf_threads_num,
      size_t queue_depth,
      bool sample_only,
      bool skip_one_batch,
      bool cache,
      bool shuffle);
  virtual ~DataReader();

  void start_reading() {
    start_reading_flag_.set();
  }

  void free_push(size_t queue_id, const shared_ptr<DatumType>& datum) {
    if (!sample_only_) {
      free_[queue_id]->push(datum);
    }
  }

  shared_ptr<DatumType> free_pop(size_t queue_id) {
    return free_[queue_id]->pop();
  }

  shared_ptr<DatumType> sample() {
    return init_->peek();
  }

  void full_push(size_t queue_id, const shared_ptr<DatumType>& datum) {
    full_[queue_id]->push(datum);
  }

  shared_ptr<DatumType> full_peek(size_t queue_id) {
    return full_[queue_id]->peek();
  }

  shared_ptr<DatumType> full_pop(size_t queue_id, const char* log_on_wait) {
    return full_[queue_id]->pop(log_on_wait);
  }

  shared_ptr<DatumType>& next_new() {
    return data_cache_->next_new();
  }

  shared_ptr<DatumType>& next_cached() {
    return data_cache_->next_cached();
  }

  bool check_memory() {
    return data_cache_->check_memory();
  }

  void just_cached() {
    data_cache_->just_cached();
  }

 protected:
  void InternalThreadEntry() override;
  void InternalThreadEntryN(size_t thread_id) override;

  const size_t parser_threads_num_, transf_threads_num_;
  const size_t queues_num_, queue_depth_;
  string db_source_;
  const size_t solver_count_, solver_rank_;
  size_t batch_size_;
  const bool skip_one_batch_;
  DataParameter_DB backend_;

  shared_ptr<BlockingQueue<shared_ptr<DatumType>>> init_;
  vector<shared_ptr<BlockingQueue<shared_ptr<DatumType>>>> free_;
  vector<shared_ptr<BlockingQueue<shared_ptr<DatumType>>>> full_;

 private:
  int current_rec_;
  int current_queue_;
  Flag start_reading_flag_;
  bool sample_only_;
  const bool cache_, shuffle_;

  DataCache* data_cache_;

  DISABLE_COPY_MOVE_AND_ASSIGN(DataReader);
};

}  // namespace caffe

#endif  // CAFFE_DATA_READER_HPP_
