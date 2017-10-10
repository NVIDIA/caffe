#ifndef CAFFE_UTIL_GPU_MEMORY_HPP_
#define CAFFE_UTIL_GPU_MEMORY_HPP_

#include <thread>
#include <unordered_map>
#include <vector>

#include "caffe/common.hpp"

#ifndef CPU_ONLY

namespace cub {
  class CachingDeviceAllocator;
}

namespace caffe {

struct GPUMemory {
  static void GetInfo(size_t* free_mem, size_t* used_mem, bool with_update = false) {
    return mgr_.GetInfo(free_mem, used_mem, with_update);
  }

  static int current_device() {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    return device;
  }

  static void* pinned_buffer(size_t size, int group = 0) {
    return mgr_.pinned_buffer(size, current_device(), group);
  }

  static void* thread_pinned_buffer(size_t size, int group = 0);

  template <class Any>
  static void allocate(Any** ptr, size_t size, int device = current_device(), int group = 0) {
    if (!try_allocate(reinterpret_cast<void**>(ptr), size, device, group)) {
      LOG(FATAL) << "Failed to allocate " << size << " bytes on device " << device
          << ". " << mgr_.report_dev_info(device);
    }
  }

  static void deallocate(void* ptr, int device = current_device()) {
    mgr_.deallocate(ptr, device);
  }

  static bool try_allocate(void** ptr, size_t size, int device = current_device(), int group = 0) {
    return mgr_.try_allocate(ptr, size, device, group);
  }

  static shared_mutex& read_write_mutex() {
    return mgr_.read_write_mutex();
  }

  enum Mode {
    CUDA_MALLOC,   // Straight CUDA malloc/free (may be expensive)
    CUB_ALLOCATOR  // CUB caching allocator
  };

  // Scope initializes global Memory Manager for a given scope.
  // It's instantiated in test(), train() and time() Caffe brewing functions
  // as well as in unit tests main().
  struct Scope {
    Scope(const std::vector<int>& gpus, Mode m = CUB_ALLOCATOR,
          bool debug = false) {
      mgr_.init(gpus, m, debug);
    }
    ~Scope() {
      mgr_.reset();
    }
  };

  // Workspace's release() functionality depends on global pool availability
  // If pool is available, it returns memory to the pool and sets ptr to nullptr
  // If pool is not available, it retains memory.
  struct Workspace {
    Workspace()
      : ptr_(nullptr), size_(0), device_(current_device()) {}
    Workspace(size_t size, int device = current_device())
      : ptr_(nullptr), size_(size), device_(device) {
      reserve(size_, device);
    }
    ~Workspace() {
      if (ptr_ != nullptr) {
        mgr_.deallocate(ptr_, device_);
      }
    }

    void* data() const {
      CHECK(ptr_ != nullptr) << "(device_: " << device_
                             << ", current device: " << Caffe::current_device() << ")";
      return ptr_;
    }

    size_t size() const { return size_; }
    int device() const { return device_; }
    bool empty() const { return ptr_ == nullptr; }
    bool safe_reserve(size_t size, int device = current_device());
    bool try_reserve(size_t size, int device = current_device());

    void reserve(size_t size, int device = current_device()) {
      if (!try_reserve(size, device)) {
        LOG(FATAL) << "Out of memory: failed to allocate " << size
            << " bytes on device " << device;
      }
    }

    void release() {
      if (mgr_.using_pool() && ptr_ != nullptr) {
        mgr_.deallocate(ptr_, device_);
        ptr_ = nullptr;
        size_ = 0;
      }
      // otherwise (no pool) - we retain memory in the buffer
    }

    void zero_mem() {
      cudaStream_t stream = Caffe::thread_stream();
      CUDA_CHECK(cudaMemsetAsync(ptr_, 0, size_, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }

   private:
    void* ptr_;
    size_t size_;
    int device_;

    DISABLE_COPY_MOVE_AND_ASSIGN(Workspace);
  };

 private:
  struct Manager {
    Manager();
    ~Manager();
    void lazy_init(int device);
    void GetInfo(size_t* free_mem, size_t* used_mem, bool with_update);
    void deallocate(void* ptr, int device);
    bool try_allocate(void** ptr, size_t size, int device, int group = 0);
    const char* pool_name() const;
    bool using_pool() const { return mode_ != CUDA_MALLOC; }
    void init(const std::vector<int>&, Mode, bool);
    void reset();
    void* pinned_buffer(size_t size, int device, int group);
    std::string report_dev_info(int device);
    shared_mutex& read_write_mutex() { return mutex_; }

    // keep a reference to which streams have been used, so that we can call
    // deallocate before the streams themselves have been destroyed
    std::set<shared_ptr<CudaStream>> used_streams_;

    Mode mode_;
    bool debug_;

   private:
    struct DevInfo {
      DevInfo() {
        free_ = total_ = flush_count_ = 0;
      }
      size_t free_;
      size_t total_;
      unsigned flush_count_;
    };

    void update_dev_info(int device);
    bool resize_buffers(int device, int group);

    vector<DevInfo> dev_info_;
    bool initialized_;
    std::unique_ptr<cub::CachingDeviceAllocator> cub_allocator_;
    vector<vector<void*>> pinned_host_buffers_;
    vector<vector<void*>> pinned_device_buffers_;
    vector<vector<size_t>> pinned_buffer_sizes_;
    vector<size_t> update_thresholds_;
    shared_mutex mutex_;

    static const unsigned int BIN_GROWTH;  ///< Geometric growth factor
    static const unsigned int MIN_BIN;  ///< Minimum bin
    static const unsigned int MAX_BIN;  ///< Maximum bin
    static const size_t MAX_CACHED_BYTES;  ///< Maximum aggregate cached bytes
    static const size_t MAX_CACHED_SIZE;  ///< 2^MAX_BIN
    static const size_t INITIAL_PINNED_BYTES;
  };

  static Manager mgr_;
  static const int INVALID_DEVICE;  ///< Default is invalid: CUB takes care
};

}  // namespace caffe

#endif

#endif
