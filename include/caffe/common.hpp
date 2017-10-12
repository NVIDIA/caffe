#ifndef CAFFE_COMMON_HPP_
#define CAFFE_COMMON_HPP_

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <boost/version.hpp>
#if BOOST_VERSION >= 106100
// error: class "boost::common_type<long, long>" has no member "type"
#define BOOST_NO_CXX11_VARIADIC_TEMPLATES
#endif

#include <boost/thread.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

#include <atomic>
#include <condition_variable>
#include <climits>
#include <cmath>
#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <map>
#include <unordered_map>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <utility>  // pair
#include <vector>

#ifndef CPU_ONLY
#  ifdef USE_CUDNN
#    include <cudnn.h>
#  endif
#  include "caffe/util/device_alternate.hpp"
#endif

#if defined(CPU_ONLY) && defined(USE_CUDNN)
  #error "USE_CUDNN mode is not compatible with CPU_ONLY"
#endif

#include "caffe/util/float16.hpp"
#ifndef CPU_ONLY
#  if CUDA_VERSION >= 8000
#    define CAFFE_DATA_HALF CUDA_R_16F
#  else
#    define CAFFE_DATA_HALF CUBLAS_DATA_HALF
#  endif
#endif
// Convert macro to string
#define STRINGIFY(m) #m
#define AS_STRING(m) STRINGIFY(m)

// gflags 2.1 issue: namespace google was changed to gflags without warning.
// Luckily we will be able to use GFLAGS_GFLAGS_H_ to detect if it is version
// 2.1. If yes, we will add a temporary solution to redirect the namespace.
// TODO(Yangqing): Once gflags solves the problem in a more elegant way, let's
// remove the following hack.
#ifndef GFLAGS_GFLAGS_H_
namespace gflags = google;
#endif  // GFLAGS_GFLAGS_H_

// Disable the copy and assignment operator for a class.
#define DISABLE_COPY_MOVE_AND_ASSIGN(classname) \
private:\
  classname(const classname&) = delete;\
  classname(classname&&) = delete;\
  classname& operator=(const classname&) = delete; \
  classname& operator=(classname&&) = delete

#define INSTANTIATE_CLASS_CPU(classname) \
  char gInstantiationGuard##classname; \
  template class classname<float>; \
  template class classname<double>

#define INSTANTIATE_CLASS_CPU_FB(classname) \
  char gInstantiationGuard##classname; \
  template class classname<float, float>; \
  template class classname<float, double>; \
  template class classname<double, float>; \
  template class classname<double, double>

// Instantiate a class with float and double specifications.
#ifdef CPU_ONLY

# define INSTANTIATE_CLASS(classname)    INSTANTIATE_CLASS_CPU(classname)
# define INSTANTIATE_CLASS_FB(classname) INSTANTIATE_CLASS_CPU_FB(classname)

#else

# define INSTANTIATE_CLASS(classname) \
    INSTANTIATE_CLASS_CPU(classname); \
    template class classname<float16>

# define INSTANTIATE_CLASS_FB(classname) \
    INSTANTIATE_CLASS_CPU_FB(classname); \
    template class classname<float16, float>; \
    template class classname<float, float16>; \
    template class classname<float16, double>; \
    template class classname<double, float16>; \
    template class classname<float16, float16>

# define INSTANTIATE_LAYER_GPU_FORWARD(classname) \
  template void classname<float>::Forward_gpu( \
      const std::vector<Blob*>& bottom, \
      const std::vector<Blob*>& top); \
  template void classname<double>::Forward_gpu( \
      const std::vector<Blob*>& bottom, \
      const std::vector<Blob*>& top)

# define INSTANTIATE_LAYER_GPU_FORWARD_F16(classname) \
  template void classname<float16>::Forward_gpu( \
      const std::vector<Blob*>& bottom, \
      const std::vector<Blob*>& top)

# define INSTANTIATE_LAYER_GPU_FORWARD_F16_FB(classname, member) \
  template void classname<float16, float>::member( \
      const std::vector<Blob*>& bottom, \
      const std::vector<Blob*>& top); \
  template void classname<float, float16>::member( \
      const std::vector<Blob*>& bottom, \
      const std::vector<Blob*>& top); \
  template void classname<float16, double>::member( \
      const std::vector<Blob*>& bottom, \
      const std::vector<Blob*>& top); \
  template void classname<double, float16>::member( \
      const std::vector<Blob*>& bottom, \
      const std::vector<Blob*>& top); \
  template void classname<float16, float16>::member( \
      const std::vector<Blob*>& bottom, \
      const std::vector<Blob*>& top)

# define INSTANTIATE_LAYER_GPU_BACKWARD_F16_FB(classname, member) \
  template void classname<float16, float>::member( \
      const std::vector<Blob*>& top, \
      const std::vector<bool>& propagate_down, \
      const std::vector<Blob*>& bottom); \
  template void classname<float, float16>::member( \
      const std::vector<Blob*>& top, \
      const std::vector<bool>& propagate_down, \
      const std::vector<Blob*>& bottom); \
  template void classname<float16, double>::member( \
      const std::vector<Blob*>& top, \
      const std::vector<bool>& propagate_down, \
      const std::vector<Blob*>& bottom); \
  template void classname<double, float16>::member( \
      const std::vector<Blob*>& top, \
      const std::vector<bool>& propagate_down, \
      const std::vector<Blob*>& bottom); \
  template void classname<float16, float16>::member( \
      const std::vector<Blob*>& top, \
      const std::vector<bool>& propagate_down, \
      const std::vector<Blob*>& bottom)

# define INSTANTIATE_LAYER_GPU_FORWARD_FB(classname, member) \
  template void classname<float, float>::member( \
      const std::vector<Blob*>& bottom, \
      const std::vector<Blob*>& top); \
  template void classname<float, double>::member( \
      const std::vector<Blob*>& bottom, \
      const std::vector<Blob*>& top); \
  template void classname<double, float>::member( \
      const std::vector<Blob*>& bottom, \
      const std::vector<Blob*>& top); \
  template void classname<double, double>::member( \
      const std::vector<Blob*>& bottom, \
      const std::vector<Blob*>& top);

# define INSTANTIATE_LAYER_GPU_BACKWARD_FB(classname, member) \
  template void classname<float, float>::member( \
      const std::vector<Blob*>& top, \
      const std::vector<bool>& propagate_down, \
      const std::vector<Blob*>& bottom); \
  template void classname<float, double>::member( \
      const std::vector<Blob*>& top, \
      const std::vector<bool>& propagate_down, \
      const std::vector<Blob*>& bottom); \
  template void classname<double, float>::member( \
      const std::vector<Blob*>& top, \
      const std::vector<bool>& propagate_down, \
      const std::vector<Blob*>& bottom); \
  template void classname<double, double>::member( \
      const std::vector<Blob*>& top, \
      const std::vector<bool>& propagate_down, \
      const std::vector<Blob*>& bottom)

#  define INSTANTIATE_LAYER_GPU_FORWARD_ONLY_FB(classname) \
    INSTANTIATE_LAYER_GPU_FORWARD_FB(classname, Forward_gpu); \
    INSTANTIATE_LAYER_GPU_FORWARD_F16_FB(classname, Forward_gpu)

#  define INSTANTIATE_LAYER_GPU_BACKWARD_ONLY_FB(classname) \
    INSTANTIATE_LAYER_GPU_BACKWARD_FB(classname, Backward_gpu); \
    INSTANTIATE_LAYER_GPU_BACKWARD_F16_FB(classname, Backward_gpu)

#  define INSTANTIATE_LAYER_GPU_FUNCS_FB(classname) \
    INSTANTIATE_LAYER_GPU_FORWARD_FB(classname, Forward_gpu); \
    INSTANTIATE_LAYER_GPU_FORWARD_F16_FB(classname, Forward_gpu); \
    INSTANTIATE_LAYER_GPU_BACKWARD_FB(classname, Backward_gpu); \
    INSTANTIATE_LAYER_GPU_BACKWARD_F16_FB(classname, Backward_gpu)

#  define INSTANTIATE_LAYER_GPU_FW_MEMBER_FB(classname, member) \
    INSTANTIATE_LAYER_GPU_FORWARD_FB(classname, member); \
    INSTANTIATE_LAYER_GPU_FORWARD_F16_FB(classname, member)

#  define INSTANTIATE_LAYER_GPU_BW_MEMBER_FB(classname, member) \
    INSTANTIATE_LAYER_GPU_BACKWARD_FB(classname, member); \
    INSTANTIATE_LAYER_GPU_BACKWARD_F16_FB(classname, member)

#endif

// A simple macro to mark codes that are not implemented, so that when the code
// is executed we will see a fatal log.
#define NOT_IMPLEMENTED LOG(FATAL) << "Not Implemented Yet"

// See PR #1236
namespace cv { class Mat; }

namespace caffe {

// Common functions and classes from std that caffe often uses.
using std::fstream;
using std::ios;
using std::isnan;
using std::isinf;
using std::iterator;
using std::make_pair;
using std::map;
using std::unordered_map;
using std::ostringstream;
using std::pair;
using std::set;
using std::string;
using std::stringstream;
using std::vector;
using std::unique_ptr;
using std::mutex;
using std::lock_guard;
// std::shared_ptr would be better but pycaffe breaks
using boost::shared_ptr;
using boost::weak_ptr;
using boost::make_shared;
using boost::shared_mutex;
using boost::shared_lock;
using boost::upgrade_lock;
using boost::unique_lock;
using boost::upgrade_to_unique_lock;

#ifndef CPU_ONLY
// Shared CUDA Stream for correct life cycle management
class CudaStream {
  explicit CudaStream(bool high_priority);

 public:
  ~CudaStream();

  static shared_ptr<CudaStream> create(bool high_priority = false) {
    shared_ptr<CudaStream> pstream(new CudaStream(high_priority));
    return pstream;
  }

  cudaStream_t get() const {
    return stream_;
  }

 private:
  cudaStream_t stream_;

  static int current_device() {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    return device;
  }
  DISABLE_COPY_MOVE_AND_ASSIGN(CudaStream);
};

#ifdef USE_CUDNN
struct CuDNNHandle {
  explicit CuDNNHandle(cudaStream_t stream);
  ~CuDNNHandle();

  cudnnHandle_t get() const {
    return handle_;
  }
 private:
  cudnnHandle_t handle_;
  DISABLE_COPY_MOVE_AND_ASSIGN(CuDNNHandle);
};
#endif
#endif

// A global initialization function that you should call in your main function.
// Currently it initializes google flags and google logging.
void GlobalInit(int* pargc, char*** pargv);

// A singleton class to hold common caffe stuff, such as the handler that
// caffe is going to use for cublas, curand, etc.
class Caffe {
 public:
  ~Caffe();

  // Thread local context for Caffe. Moved to common.cpp instead of
  // including boost/thread.hpp to avoid a boost/NVCC issues (#1009, #1010)
  // on OSX. Also fails on Linux with CUDA 7.0.18.
  static Caffe& Get();

  enum Brew { CPU, GPU };

  // This random number generator facade hides boost and CUDA rng
  // implementation from one another (for cross-platform compatibility).
  class RNG {
   public:
    RNG();
    explicit RNG(uint64_t seed);
    RNG(const RNG&);
    RNG& operator = (const RNG&);
    void* generator();
   private:
    class Generator;
    shared_ptr<Generator> generator_;
  };

  // Getters for boost rng, curand, and cublas handles
  static RNG& rng_stream() {
    if (!Get().random_generator_) {
      Get().random_generator_.reset(new RNG());
    }
    return *(Get().random_generator_);
  }
#ifndef CPU_ONLY
  static shared_ptr<CudaStream> thread_pstream(int group = 0) {
    return Get().device_pstream(group);
  }
  static cudaStream_t thread_stream(int group = 0) {
    return Get().device_pstream(group)->get();
  }
  static shared_ptr<CudaStream> th_pstream_aux(int id) {
    return Get().device_pstream_aux(id);
  }
  static cudaStream_t th_stream_aux(int id) {
    return Get().device_pstream_aux(id)->get();
  }
  static cublasHandle_t cublas_handle(int group = 0) {
    return Get().device_cublas_handle(group);
  }
  static curandGenerator_t curand_generator() {
    return Get().device_curand_generator();
  }
#ifdef USE_CUDNN
  static cudnnHandle_t cudnn_handle(int group = 0) {
    return Get().device_cudnn_handle(group);
  }
#endif
#endif

  // Returns the mode: running on CPU or GPU.
  static Brew mode() {
    return Get().mode_;
  }
  // The setters for the variables
  // Sets the mode. It is recommended that you don't change the mode halfway
  // into the program since that may cause allocation of pinned memory being
  // freed in a non-pinned way, which may cause problems - I haven't verified
  // it personally but better to note it here in the header file.
  static void set_mode(Brew mode) {
    if (mode == Caffe::GPU && !has_device()) {
      LOG(ERROR) << "Cannot use GPU: no device detected";
    }
    Get().mode_ = mode;
  }
  // Next seed. It's deterministic if root seed is already set.
  static uint64_t next_seed();
  // Sets the random seed of both boost and curand
  // Uses system generated one if -1 passed
  static void set_random_seed(uint64_t random_seed = SEED_NOT_SET);
  // For correct determinism user should set a seed for a root solver
  // Note: it invokes set_random_seed internally
  static void set_root_seed(uint64_t random_seed);
  // Sets the root device. Function name remains the same for backward compatibility.
  static void SetDevice(const int device_id);
  static int root_device() {
    return root_device_;
  }
  // Prints the current GPU status.
  static void DeviceQuery();
  // Check if specified device is available
  static bool CheckDevice(const int device_id);
  // Search from start_id to the highest possible device ordinal,
  // return the ordinal of the first available device.
  static int FindDevice(const int start_id = 0);
  // Parallel training info
  static int solver_count() { return Get().solver_count_; }
  static void set_solver_count(int val) { Get().solver_count_ = val; }
  static bool root_solver() { return Get().root_solver_; }
  static void set_root_solver(bool val) { Get().root_solver_ = val; }
  static int restored_iter() { return restored_iter_; }
  static void set_restored_iter(int val);

  static bool has_device() {
#ifndef CPU_ONLY
    static bool first_time = true;
    static int device_count = 0;
    if (first_time) {
      // in case of failure of cudaGetDeviceCount, device_count == 0
      if (cudaGetDeviceCount(&device_count) != cudaSuccess) {
        LOG(WARNING) << "Failed to access GPU count. Assuming has_device == false";
      }
      first_time = false;
    }
    return device_count > 0;
#else
    return false;
#endif
  }
  static void set_gpus(const std::vector<int>& gpus) {
    props().gpus_ = gpus;
  }
  static const std::vector<int>& gpus() {
    return props().gpus_;
  }
  static const std::string& caffe_version() {
    return props().caffe_version();
  }
  static const std::string& cudnn_version() {
    return props().cudnn_version();
  }
  static const std::string& cublas_version() {
    return props().cublas_version();
  }
  static const std::string& cuda_version() {
    return props().cuda_version();
  }
  static const std::string& cuda_driver_version() {
    return props().cuda_driver_version();
  }
  static std::thread::id main_thread_id() {
    return props().main_thread_id();
  }
  static bool is_main_thread() {
    return props().main_thread_id() == std::this_thread::get_id();
  }
  static std::string start_time() {
    return props().start_time();
  }
  static std::time_t init_time() {
    return props().init_time();
  }
  static std::string time_from_init();
  static int device_capability(int device) {
    return props().device_capability(device);
  }

  static int current_device() {
#ifndef CPU_ONLY
    int device = 0;
    cudaGetDevice(&device); // in case of fail, return same value as CPU_ONLY
    return device;
#else
    return 0;
#endif
  }

#ifndef CPU_ONLY
  /**
   * Minimum memory available across all deviced currently used
   * @return size_t
   */
  static size_t min_avail_device_memory();
#endif

    static int thread_count() {
    return thread_count_;
  }

  static constexpr int STREAM_ID_ASYNC_PUSH = 0;
  static constexpr int STREAM_ID_TRANSFORMER = 1;
  static constexpr uint64_t SEED_NOT_SET = static_cast<uint64_t>(-1);

 protected:
#ifndef CPU_ONLY
  static std::list<shared_ptr<CudaStream>> all_streams_;
  vector<vector<shared_ptr<CudaStream>>> device_streams_;  // [device][group]
  vector<vector<shared_ptr<CudaStream>>> device_streams_aux_;  // [device][id]
  vector<vector<cublasHandle_t>> cublas_handles_;  // [device][group]
  vector<curandGenerator_t> curand_generators_;
  shared_ptr<CudaStream> device_pstream(int group = 0);
  shared_ptr<CudaStream> device_pstream_aux(int id);
  cublasHandle_t device_cublas_handle(int group = 0);
  curandGenerator_t device_curand_generator();

#ifdef USE_CUDNN
  vector<vector<shared_ptr<CuDNNHandle>>> cudnn_handles_;  // [device][group]
  cudnnHandle_t device_cudnn_handle(int group = 0);
#endif
#endif
  shared_ptr<RNG> random_generator_;

  Brew mode_;
  int solver_count_;
  bool root_solver_;

  // Default device chosen by a user and associated with the main thread.
  // For example, if user runs `caffe train -gpu=1,0,3` then it has to be set to 1.
  static int root_device_;
  static int thread_count_;
  static int restored_iter_;
  static std::atomic<uint64_t> root_seed_;
  static std::mutex props_mutex_, caffe_mutex_, pstream_mutex_, cublas_mutex_, seed_mutex_;

 private:
  // The private constructor to avoid duplicate instantiation.
  Caffe();

  DISABLE_COPY_MOVE_AND_ASSIGN(Caffe);

  // Caffe Properties singleton
  class Properties {
    friend class Caffe;
   public:
    const std::string& caffe_version() const {
      return caffe_version_;
    }
    const std::string& cudnn_version() const {
      return cudnn_version_;
    }
    const std::string& cublas_version() const {
      return cublas_version_;
    }
    const std::string& cuda_version() const {
      return cuda_version_;
    }
    const std::string& cuda_driver_version() const {
      return cuda_driver_version_;
    }
    std::thread::id main_thread_id() const {
      return main_thread_id_;
    }
    std::string start_time() const {
      // NOLINT_NEXT_LINE(runtime/threadsafe_fn)
      return std::ctime(&init_time_);
    }
    std::time_t init_time() const {
      return init_time_;
    }
    int device_capability(int device) const {
      return compute_capabilities_[device];
    }

    ~Properties();

   private:
    std::vector<int> gpus_;
    std::time_t init_time_;
    std::thread::id main_thread_id_;
    std::string caffe_version_;
    std::string cudnn_version_;
    std::string cublas_version_;
    std::string cuda_version_;
    std::string cuda_driver_version_;
    std::vector<int> compute_capabilities_;

    Properties();
    DISABLE_COPY_MOVE_AND_ASSIGN(Properties);
  };

  static Properties props_;

  static Properties& props() {
    return props_;
  }
};

// Yet another Event implementation
class Flag {
  mutable std::mutex m_;
  mutable std::condition_variable cv_;
  bool flag_, disarmed_;

  DISABLE_COPY_MOVE_AND_ASSIGN(Flag);

 public:
  explicit Flag(bool state = false)
      : flag_(state), disarmed_(false) {}

  bool is_set() const {
    std::lock_guard<std::mutex> lock(m_);
    return flag_;
  }

  void set() {
    {
      std::lock_guard<std::mutex> lock(m_);
      flag_ = true;
    }
    cv_.notify_all();
  }

  void reset() {
    {
      std::lock_guard<std::mutex> lock(m_);
      flag_ = false;
    }
    cv_.notify_all();
  }

  void wait() const {
    std::unique_lock<std::mutex> lock(m_);
    cv_.wait(lock, [this] { return flag_; } );
  }

  void disarm() {
    {
      std::lock_guard<std::mutex> lock(m_);
      disarmed_ = true;
    }
    cv_.notify_all();
  }

  void wait_reset() {
    {
      std::unique_lock<std::mutex> lock(m_);
      cv_.wait(lock, [this] { return flag_ || disarmed_; } );
      if (!disarmed_) {
        flag_ = false;
      }
    }
    cv_.notify_all();
  }
};

class MutexVec {
  static constexpr size_t TOP_ORDINAL = 128;
  vector<shared_ptr<mutex>> v_;

 public:
  MutexVec() : v_(TOP_ORDINAL, make_shared<mutex>()) {}
  mutex& operator[] (size_t dev) {
    while (v_.size() <= dev) {
      v_.resize(v_.size(), make_shared<mutex>());
    }
    return *v_[dev];
  }
};

template <typename M>
class ThreadSafeMap {
 public:
  explicit ThreadSafeMap(std::mutex& m) : m_(m) {
    std::lock_guard<std::mutex> lock(m_);
    map_.reset(new M());
  }
  ~ThreadSafeMap() = default;

  using iterator = typename M::iterator;
  using const_iterator = typename M::const_iterator;
  using value_type = typename M::value_type;
  using mapped_type = typename M::mapped_type;
  using key_type = typename M::key_type;
  using size_type = typename M::size_type;

  size_type size() const {
    std::lock_guard<std::mutex> lock(m_);
    return map_->size();
  }
  std::pair<iterator, bool> insert(const value_type& entry) {
    std::lock_guard<std::mutex> lock(m_);
    return map_->insert(entry);
  }
  template<class... Args>
  std::pair<iterator, bool> emplace(Args&&... args) {
    std::lock_guard<std::mutex> lock(m_);
    return map_->emplace(args...);
  }
  mapped_type& operator[](const key_type& key) {
    std::lock_guard<std::mutex> lock(m_);
    return (*map_)[key];
  }
  iterator find(const key_type& key) {
    std::lock_guard<std::mutex> lock(m_);
    return map_->find(key);
  }
  const_iterator find(const key_type& key) const {
    std::lock_guard<std::mutex> lock(m_);
    return map_->find(key);
  }
  iterator begin() {
    std::lock_guard<std::mutex> lock(m_);
    return map_->begin();
  }
  const_iterator begin() const {
    std::lock_guard<std::mutex> lock(m_);
    return map_->begin();
  }
  const_iterator cbegin() const {
    std::lock_guard<std::mutex> lock(m_);
    return map_->cbegin();
  }
  iterator end() {
    std::lock_guard<std::mutex> lock(m_);
    return map_->end();
  }
  const_iterator end() const {
    std::lock_guard<std::mutex> lock(m_);
    return map_->end();
  }
  const_iterator cend() const {
    std::lock_guard<std::mutex> lock(m_);
    return map_->cend();
  }
  void clear() {
    std::lock_guard<std::mutex> lock(m_);
    map_->clear();
  }
  void insert_max(const key_type& key, const mapped_type& value) {
    std::lock_guard<std::mutex> lock(m_);
    iterator it = map_->find(key);
    if (it == map_->end()) {
      map_->emplace(key, value);
    } else if (value > it->second) {
      it->second = value;
    }
  }
  bool remove_top(key_type& key, mapped_type& value) {
    std::lock_guard<std::mutex> lock(m_);
    iterator it = map_->begin();
    if (it == map_->end()) {
      return false;
    }
    key = it->first;
    value = it->second;
    map_->erase(it);
    return true;
  }
  size_type erase(const key_type& key) {
    std::lock_guard<std::mutex> lock(m_);
    return map_->erase(key);
  }
  iterator erase(const_iterator pos) {
    std::lock_guard<std::mutex> lock(m_);
    return map_->erase(pos);
  }

 private:
  std::unique_ptr<M> map_;
  std::mutex& m_;
};


///> the biggest number n which is not greater than val and divisible by 2^power
template<int power, typename T>
inline T align_down(T val) {
  return val & ~((1 << power) - 1);
}

///> the smallest number n which is not less than val and divisible by 2^power
template<int power, typename T>
inline T align_up(T val) {
  return !(val & ((1 << power) - 1)) ? val : (val | ((1 << power) - 1)) + 1;
}

template <typename T>
inline bool is_even(T val) { return (val & 1) == 0; }

///> the smallest even number n which is not less than val
template <typename T>
inline T even(T val) { return val & 1 ? val + 1 : val; }


// Unlike dataType<> this one keeps typed values to be used in caffe_* calls.
template <typename Dtype> class TypedConsts;
template<> class TypedConsts<double>  {
 public:
  static const double zero, one;
};
template<> class TypedConsts<float>  {
 public:
  static const float zero, one;
};
#ifndef CPU_ONLY
template<> class TypedConsts<float16>  {
 public:
  static const float16 zero, one;
};
#endif
template<> class TypedConsts<int>  {
 public:
  static const int zero, one;
};


template <typename Dtype>
CAFFE_UTIL_IHD Dtype max_dtype();
template <>
CAFFE_UTIL_IHD double max_dtype<double>() {
  return DBL_MAX;
}
template <>
CAFFE_UTIL_IHD float max_dtype<float>() {
  return FLT_MAX;
}
#ifndef CPU_ONLY
template <>
CAFFE_UTIL_IHD float16 max_dtype<float16>() {
  float16 ret;
  // Exponent all ones except LSB (0x1e), mantissa is all ones (0x3ff)
  ret.setx(0x7bffU);
  return ret;
}
// Largest positive FP16 value, corresponds to 6.5504e+04
#ifdef __CUDACC__
template <>
CAFFE_UTIL_IHD half max_dtype<half>() {
    half ret;
    // Exponent all ones except LSB (0x1e), mantissa is all ones (0x3ff)
    ret.setx(0x7bffU);
    return ret;
}
#endif
#endif

// Normalized minimums:
template <typename Dtype>
CAFFE_UTIL_IHD Dtype min_dtype();
template <>
CAFFE_UTIL_IHD double min_dtype<double>() {
  return DBL_MIN;
}
template <>
CAFFE_UTIL_IHD float min_dtype<float>() {
  return FLT_MIN;
}
#ifndef CPU_ONLY
template <>
CAFFE_UTIL_IHD float16 min_dtype<float16>() {
  float16 ret;
  // Exponent is 0x01 (5 bits), mantissa is all zeros (10 bits)
  ret.setx(0x0400U);
  return ret;
}
// Smallest positive (normalized) FP16 value, corresponds to 6.1035e-05
#ifdef __CUDACC__
template <>
CAFFE_UTIL_IHD half min_dtype<half>() {
  half ret;
  // Exponent is 0x01 (5 bits), mantissa is all zeros (10 bits)
  ret.setx(0x0400U);
  return ret;
}
#endif
#endif

template <typename Dtype>
CAFFE_UTIL_IHD Dtype epsilon_dtype();
template <>
CAFFE_UTIL_IHD double epsilon_dtype<double>() {
  return DBL_EPSILON;
}
template <>
CAFFE_UTIL_IHD float epsilon_dtype<float>() {
  return FLT_EPSILON;
}

#ifndef CPU_ONLY
template <>
CAFFE_UTIL_IHD float16 epsilon_dtype<float16>() {
  float16 ret;
  ret.setx(0x1001U);
  return ret;
}
#ifdef __CUDACC__
template <>
CAFFE_UTIL_IHD half epsilon_dtype<half>() {
  half ret;
  ret.setx(0x1001U);
  return ret;
}
#endif
#endif

template <typename Dtype> constexpr
inline bool is_precise() {
  return false;
}
template <> constexpr
inline bool is_precise<double>() {
  return true;
}
template <> constexpr
inline bool is_precise<float>() {
  return true;
}

template <typename Dtype>
CAFFE_UTIL_IHD Dtype tol(Dtype fine, Dtype coarse) {
  return is_precise<Dtype>() ? fine : coarse;
}

template<typename Dtype>
void atomic_maximum(std::atomic<Dtype>& max_val, Dtype const& new_val) noexcept {
    Dtype prev_val = std::atomic_load(&max_val);
    while (prev_val < new_val &&
        !max_val.compare_exchange_weak(prev_val, new_val)) {}
}

template<typename Dtype>
void atomic_minimum(std::atomic<Dtype>& min_val, Dtype const& new_val) noexcept {
  Dtype prev_val = std::atomic_load(&min_val);
  while (prev_val > new_val &&
         !min_val.compare_exchange_weak(prev_val, new_val)) {}
}

template <typename Dtype>
float gb_round2(Dtype val) {
  return std::round(val * 1.e-7) * 0.01F;
}

template <typename Dtype>
float f_round2(Dtype val) {
  return std::round(val * 100.F) * 0.01F;
}

}  // namespace caffe

#endif  // CAFFE_COMMON_HPP_
