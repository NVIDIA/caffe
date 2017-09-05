#include <boost/thread.hpp>
#include <string>

#include "caffe/data_reader.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/parallel.hpp"
#include "caffe/util/blocking_queue.hpp"

namespace caffe {

template<typename T>
BlockingQueue<T>::BlockingQueue() {}

template<typename T>
BlockingQueue<T>::~BlockingQueue() {}

template<typename T>
void BlockingQueue<T>::push(const T& t) {
  boost::mutex::scoped_lock lock(mutex_);
  queue_.push(t);
  lock.unlock();
  condition_.notify_one();
}

template<typename T>
bool BlockingQueue<T>::try_pop(T* t) {
  boost::mutex::scoped_lock lock(mutex_);
  if (queue_.empty()) {
    return false;
  }
  *t = queue_.front();
  queue_.pop();
  return true;
}

template<typename T>
T BlockingQueue<T>::pop(const char* log_on_wait) {
  boost::mutex::scoped_lock lock(mutex_);
  while (queue_.empty()) {
    LOG_EVERY_N(INFO, 10000) << log_on_wait;
    condition_.wait(lock);
  }
  T t = queue_.front();
  queue_.pop();
  return t;
}

template<typename T>
T BlockingQueue<T>::pop() {
  boost::mutex::scoped_lock lock(mutex_);
  while (queue_.empty()) {
    condition_.wait(lock);
  }
  T t(queue_.front());
  queue_.pop();
  return t;
}

template<typename T>
bool BlockingQueue<T>::try_peek(T* t) {
  boost::mutex::scoped_lock lock(mutex_);
  if (queue_.empty()) {
    return false;
  }
  *t = queue_.front();
  return true;
}

template<typename T>
T BlockingQueue<T>::peek() {
  boost::mutex::scoped_lock lock(mutex_);
  while (queue_.empty()) {
    condition_.wait(lock);
  }
  return queue_.front();
}

template<typename T>
size_t BlockingQueue<T>::size() const {
  boost::mutex::scoped_lock lock(mutex_);
  return queue_.size();
}

template<typename T>
bool BlockingQueue<T>::nonblocking_size(size_t* size) const {
  boost::mutex::scoped_lock lock(mutex_, boost::try_to_lock);
  if (lock.owns_lock()) {
    *size = queue_.size();
    return true;
  }
  return false;
}

template class BlockingQueue<int>;
template class BlockingQueue<shared_ptr<Batch<float>>>;
template class BlockingQueue<shared_ptr<Batch<double>>>;
#ifndef CPU_ONLY
template class BlockingQueue<shared_ptr<Batch<float16>>>;
#endif
template class BlockingQueue<shared_ptr<Datum>>;
template class BlockingQueue<shared_ptr<AnnotatedDatum>>;
template class BlockingQueue<P2PSync*>;

}  // namespace caffe
