#ifndef CAFFE_UTIL_GPU_MEMORY_HPP_
#define CAFFE_UTIL_GPU_MEMORY_HPP_

#include <vector>
#include "caffe/common.hpp"

namespace caffe {

class gpu_memory {
 public:
  enum PoolMode { NoPool, CnMemPool, CubPool };

  static const char* getPoolName();
  static bool usingPool() {
    return mode_ != NoPool;
  }

  class arena {
   public:
    arena(const std::vector<int>& gpus, PoolMode m = CubPool, bool debug = false) {
      init(gpus, m, debug);
    }
    ~arena() {
      destroy();
     }
  };

 private:
  static void init(const std::vector<int>&, PoolMode, bool);
    static void destroy();

    static bool     initialized_;
    static PoolMode mode_;
    static size_t   poolsize_;
  static bool       debug_;
#ifndef CPU_ONLY

 public:
  static void allocate(void **ptr, size_t size,
                       cudaStream_t stream = cudaStreamDefault);
  static void deallocate(void *ptr, cudaStream_t = cudaStreamDefault);
  static void registerStream(cudaStream_t stream);
  static void getInfo(size_t *free_mem, size_t *used_mem);

 private:
  static void initMEM(const std::vector<int>& gpus, PoolMode m);

#endif
};

}  // namespace caffe

# endif
