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
class DataReader : public InternalThread {
 private:
  class CursorManager {
    shared_ptr<db::DB> db_;
    unique_ptr<db::Cursor> cursor_;
    const size_t solver_count_, solver_rank_, batch_size_;
    const size_t parser_threads_, parser_thread_id_;
    const size_t rank_cycle_, full_cycle_;
    size_t rec_id_, rec_end_;
    size_t num_skips_;

   public:
    CursorManager(shared_ptr<db::DB> db, size_t solver_count, size_t solver_rank,
        size_t parser_threads, size_t parser_thread_id, size_t batch_size_, size_t num_skips_);
    ~CursorManager();
    void next(Datum* datum);
    void fetch(Datum* datum);
    void rewind();

    size_t full_cycle() const {
      return full_cycle_;
    }

    DISABLE_COPY_MOVE_AND_ASSIGN(CursorManager);
  };

 public:
  DataReader(const LayerParameter& param,
      size_t solver_count,
      size_t solver_rank,
      size_t parser_threads_num,
      size_t transf_threads_num,
      size_t queue_depth,
      bool sample_only,
      bool skip_one_batch);
  virtual ~DataReader();

  void start_reading() {
    start_reading_flag_.set();
  }

  void free_push(size_t queue_id, const shared_ptr<Datum>& datum) {
    if (!sample_only_) {
      free_[queue_id]->push(datum);
    }
  }

  shared_ptr<Datum> free_pop(size_t queue_id) {
    return free_[queue_id]->pop();
  }

  shared_ptr<Datum> sample() {
    return init_->peek();
  }

  bool sample_only() const {
    return sample_only_;
  }

  void full_push(size_t queue_id, const shared_ptr<Datum>& datum) {
    full_[queue_id]->push(datum);
  }

  shared_ptr<Datum> full_peek(size_t queue_id) {
    return full_[queue_id]->peek();
  }

  shared_ptr<Datum> full_pop(size_t queue_id, const char* log_on_wait) {
    return full_[queue_id]->pop(log_on_wait);
  }

 protected:
  void InternalThreadEntry() override;
  void InternalThreadEntryN(size_t thread_id) override;

  const size_t parser_threads_num_, transf_threads_num_;
  const size_t queues_num_, queue_depth_;
  string db_source_;
  const size_t solver_count_, solver_rank_;
  size_t batch_size_;
  size_t num_skips_;
  const bool skip_one_batch_;
  DataParameter_DB backend_;

  shared_ptr<BlockingQueue<shared_ptr<Datum>>> init_;
  vector<shared_ptr<BlockingQueue<shared_ptr<Datum>>>> free_;
  vector<shared_ptr<BlockingQueue<shared_ptr<Datum>>>> full_;

 private:
  int current_rec_;
  int current_queue_;
  Flag start_reading_flag_;
  bool sample_only_;

  DISABLE_COPY_MOVE_AND_ASSIGN(DataReader);
};

}  // namespace caffe

#endif  // CAFFE_DATA_READER_HPP_
