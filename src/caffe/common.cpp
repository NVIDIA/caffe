#include <glog/logging.h>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <ios>
#include <memory>

#include "caffe/common.hpp"
#include "caffe/util/device_alternate.hpp"
#include "caffe/util/gpu_memory.hpp"
#include "caffe/util/rng.hpp"
#if defined(USE_CUDNN)
#include "caffe/util/cudnn.hpp"
#endif

namespace caffe {

// Must be set before brewing
int Caffe::root_device_ = -1;
int Caffe::thread_count_ = 0;
shared_mutex Caffe::caffe_mutex_;
std::mutex Caffe::mutex_;
std::mutex Caffe::mutex2_;
int Caffe::restored_iter_;


#ifndef CPU_ONLY
// Lifecycle management for CUDA streams
std::list<shared_ptr<CudaStream>> Caffe::all_streams_;
#endif

Caffe& Caffe::Get() {
  upgrade_lock<shared_mutex> lock(caffe_mutex_);
  // Make sure each thread can have different values.
  static thread_local shared_ptr<Caffe> thread_instance_;
  if (!thread_instance_.get()) {
    upgrade_to_unique_lock<shared_mutex> write_lock(lock);
    if (!thread_instance_.get()) {
      thread_instance_.reset(new Caffe());
      ++thread_count_;
    }
  }
  return *(thread_instance_.get());
}

// random seeding
int64_t cluster_seedgen(void) {
  int64_t s, seed, pid;
  FILE* f = fopen("/dev/urandom", "rb");
  if (f && fread(&seed, 1, sizeof(seed), f) == sizeof(seed)) {
    fclose(f);
    return seed;
  }

  LOG(INFO) << "System entropy source not available, "
              "using fallback algorithm to generate seed instead.";
  if (f)
    fclose(f);

  pid = getpid();
  s = time(NULL);
  seed = std::abs(((s * 181) * ((pid - 83) * 359)) % 104729);
  return seed;
}


void GlobalInit(int* pargc, char*** pargv) {
  // Google flags.
  ::gflags::ParseCommandLineFlags(pargc, pargv, true);
  // Google logging.
  ::google::InitGoogleLogging(*(pargv)[0]);
  // Provide a backtrace on segfault.
  ::google::InstallFailureSignalHandler();
}

#ifdef CPU_ONLY  // CPU-only Caffe.

Caffe::Caffe()
    : random_generator_(), mode_(Caffe::CPU),
      solver_count_(1), root_solver_(true) { }

Caffe::~Caffe() { }

void Caffe::set_random_seed(const unsigned int seed) {
  // RNG seed
  Get().random_generator_.reset(new RNG(seed));
}

void Caffe::SetDevice(const int device_id) {
  NO_GPU;
}

void Caffe::DeviceQuery() {
  NO_GPU;
}

bool Caffe::CheckDevice(const int device_id) {
  NO_GPU;
  return false;
}

int Caffe::FindDevice(const int start_id) {
  NO_GPU;
  return -1;
}

class Caffe::RNG::Generator {
 public:
  Generator() : rng_(new caffe::rng_t(cluster_seedgen())) {}
  explicit Generator(unsigned int seed) : rng_(new caffe::rng_t(seed)) {}
  caffe::rng_t* rng() { return rng_.get(); }
 private:
  shared_ptr<caffe::rng_t> rng_;
};

Caffe::RNG::RNG() : generator_(new Generator()) { }

Caffe::RNG::RNG(unsigned int seed) : generator_(new Generator(seed)) { }

Caffe::RNG& Caffe::RNG::operator=(const RNG& other) {
  generator_ = other.generator_;
  return *this;
}

void* Caffe::RNG::generator() {
  return static_cast<void*>(generator_->rng());
}

#else  // Normal GPU + CPU Caffe.

Caffe::Caffe()
    : random_generator_(), mode_(Caffe::CPU), solver_count_(1), root_solver_(true) {
  int count;
  CUDA_CHECK(cudaGetDeviceCount(&count));
  device_streams_.resize(count);
  device_streams_aux_.resize(count);
  cublas_handles_.resize(count);
  curand_generators_.resize(count);
#ifdef USE_CUDNN
  cudnn_handles_.resize(count);
#endif
}

Caffe::~Caffe() {
  for (vector<cublasHandle_t>& group_cublas_handles : cublas_handles_) {
    for (cublasHandle_t h : group_cublas_handles) {
      if (h) {
        CUBLAS_CHECK(cublasDestroy(h));
      }
    }
  }
  for_each(curand_generators_.begin(), curand_generators_.end(), [](curandGenerator_t h) {
    if (h) {
      CURAND_CHECK(curandDestroyGenerator(h));
    }
  });
}

CudaStream::CudaStream(bool high_priority = false) {
  if (high_priority) {
    int leastPriority, greatestPriority;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
    CUDA_CHECK(cudaStreamCreateWithPriority(&stream_, cudaStreamDefault, greatestPriority));

  } else {
    CUDA_CHECK(cudaStreamCreate(&stream_));
  }
  DLOG(INFO) << "New " << (high_priority ? "high priority " : "") << "stream "
      << stream_ << " on device " << current_device() << ", thread " << std::this_thread::get_id();
}

CudaStream::~CudaStream() {
  int current_device;  // Just to check CUDA status:
  cudaError_t status = cudaGetDevice(&current_device);
  // Preventing dead lock while Caffe shutting down.
  if (status != cudaErrorCudartUnloading) {
    CUDA_CHECK(cudaStreamDestroy(stream_));
  }
}

shared_ptr<CudaStream> Caffe::device_pstream(int group) {
  std::lock_guard<std::mutex> lock(mutex_);
  vector<shared_ptr<CudaStream>>& group_streams = device_streams_[current_device()];
  if (group + 1 > group_streams.size()) {
    group_streams.resize(group + 1);
  }
  if (!group_streams[group]) {
    group_streams[group] = CudaStream::create();
    all_streams_.push_back(group_streams[group]);
  }
  return group_streams[group];
}

shared_ptr<CudaStream> Caffe::device_pstream_aux(int id) {
  std::lock_guard<std::mutex> lock(mutex_);
  vector<shared_ptr<CudaStream>>& streams = device_streams_aux_[current_device()];
  if (id + 1 > streams.size()) {
    streams.resize(id + 1);
  }
  if (!streams[id]) {
    streams[id] = CudaStream::create();
    all_streams_.push_back(streams[id]);
  }
  return streams[id];
}

cublasHandle_t Caffe::device_cublas_handle(int group) {
  std::lock_guard<std::mutex> lock(mutex2_);
  vector<cublasHandle_t>& group_cublas_handles = cublas_handles_[current_device()];
  if (group + 1 > group_cublas_handles.size()) {
    group_cublas_handles.resize(group + 1);
  }
  cublasHandle_t& cublas_handle = group_cublas_handles[group];
  if (!cublas_handle) {
    // Try to create a cublas handler, and report an error if failed (but we will
    // keep the program running as one might just want to run CPU code).
    if (cublasCreate(&cublas_handle) != CUBLAS_STATUS_SUCCESS) {
      LOG(ERROR) << "Cannot create Cublas handle. Cublas won't be available.";
    }
    CUBLAS_CHECK(cublasSetStream(cublas_handle, device_pstream(group)->get()));
  }
  return cublas_handle;
}

curandGenerator_t Caffe::device_curand_generator() {
  curandGenerator_t& curand_generator = curand_generators_[current_device()];
  if (!curand_generator) {
    // Try to create a curand handler.
    if (curandCreateGenerator(&curand_generator, CURAND_RNG_PSEUDO_DEFAULT) !=
            CURAND_STATUS_SUCCESS ||
        curandSetPseudoRandomGeneratorSeed(curand_generator, cluster_seedgen()) !=
            CURAND_STATUS_SUCCESS) {
      LOG(ERROR) << "Cannot create Curand generator. Curand won't be available.";
    }
    curandSetStream(curand_generator, device_pstream()->get());
  }
  return curand_generator;
}

#ifdef USE_CUDNN
cudnnHandle_t Caffe::device_cudnn_handle(int group) {
  vector<shared_ptr<CuDNNHandle>>& group_cudnn_handles = cudnn_handles_[current_device()];
  if (group + 1 > group_cudnn_handles.size()) {
    group_cudnn_handles.resize(group + 1);
  }
  shared_ptr<CuDNNHandle>& cudnn_handle = group_cudnn_handles[group];
  if (!cudnn_handle) {
    cudnn_handle = make_shared<CuDNNHandle>(device_pstream(group)->get());
  }
  return cudnn_handle->get();
}
#endif


void Caffe::set_random_seed(const unsigned int seed) {
  // Curand seed
  static bool g_curand_availability_logged = false;
  curandGenerator_t curand_generator_handle = curand_generator();
  if (curand_generator_handle) {
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(curand_generator_handle, seed));
    CURAND_CHECK(curandSetGeneratorOffset(curand_generator_handle, 0));
  } else if (!g_curand_availability_logged) {
    LOG(ERROR) << "Curand not available. Skipping setting the curand seed.";
    g_curand_availability_logged = true;
  }
  // RNG seed
  Get().random_generator_.reset(new RNG(seed));
}

void Caffe::SetDevice(const int device_id) {
  root_device_ = device_id;
  CUDA_CHECK(cudaSetDevice(root_device_));
}

void Caffe::DeviceQuery() {
  cudaDeviceProp prop;
  int device;
  if (cudaSuccess != cudaGetDevice(&device)) {
    printf("No cuda device present.\n");
    return;
  }
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  LOG(INFO) << "Device id:                     " << device;
  LOG(INFO) << "Major revision number:         " << prop.major;
  LOG(INFO) << "Minor revision number:         " << prop.minor;
  LOG(INFO) << "Name:                          " << prop.name;
  LOG(INFO) << "Total global memory:           " << prop.totalGlobalMem;
  LOG(INFO) << "Total shared memory per block: " << prop.sharedMemPerBlock;
  LOG(INFO) << "Total registers per block:     " << prop.regsPerBlock;
  LOG(INFO) << "Warp size:                     " << prop.warpSize;
  LOG(INFO) << "Maximum memory pitch:          " << prop.memPitch;
  LOG(INFO) << "Maximum threads per block:     " << prop.maxThreadsPerBlock;
  LOG(INFO) << "Maximum dimension of block:    "
      << prop.maxThreadsDim[0] << ", " << prop.maxThreadsDim[1] << ", "
      << prop.maxThreadsDim[2];
  LOG(INFO) << "Maximum dimension of grid:     "
      << prop.maxGridSize[0] << ", " << prop.maxGridSize[1] << ", "
      << prop.maxGridSize[2];
  LOG(INFO) << "Clock rate:                    " << prop.clockRate;
  LOG(INFO) << "Total constant memory:         " << prop.totalConstMem;
  LOG(INFO) << "Texture alignment:             " << prop.textureAlignment;
  LOG(INFO) << "Concurrent copy and execution: "
      << (prop.deviceOverlap ? "Yes" : "No");
  LOG(INFO) << "Number of multiprocessors:     " << prop.multiProcessorCount;
  LOG(INFO) << "Kernel execution timeout:      "
      << (prop.kernelExecTimeoutEnabled ? "Yes" : "No");
  return;
}

bool Caffe::CheckDevice(const int device_id) {
  // This function checks the availability of GPU #device_id.
  // It attempts to create a context on the device by calling cudaFree(0).
  // cudaSetDevice() alone is not sufficient to check the availability.
  // It lazily records device_id, however, does not initialize a
  // context. So it does not know if the host thread has the permission to use
  // the device or not.
  //
  // In a shared environment where the devices are set to EXCLUSIVE_PROCESS
  // or EXCLUSIVE_THREAD mode, cudaSetDevice() returns cudaSuccess
  // even if the device is exclusively occupied by another process or thread.
  // Cuda operations that initialize the context are needed to check
  // the permission. cudaFree(0) is one of those with no side effect,
  // except the context initialization.
  bool r = ((cudaSuccess == cudaSetDevice(device_id)) &&
            (cudaSuccess == cudaFree(0)));
  // reset any error that may have occurred.
  cudaGetLastError();
  return r;
}

int Caffe::FindDevice(const int start_id) {
  // This function finds the first available device by checking devices with
  // ordinal from start_id to the highest available value. In the
  // EXCLUSIVE_PROCESS or EXCLUSIVE_THREAD mode, if it succeeds, it also
  // claims the device due to the initialization of the context.
  int count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&count));
  for (int i = start_id; i < count; i++) {
    if (CheckDevice(i)) return i;
  }
  return -1;
}

class Caffe::RNG::Generator {
 public:
  Generator() : rng_(new caffe::rng_t(cluster_seedgen())) {}
  explicit Generator(unsigned int seed) : rng_(new caffe::rng_t(seed)) {}
  caffe::rng_t* rng() { return rng_.get(); }
 private:
  shared_ptr<caffe::rng_t> rng_;
};

Caffe::RNG::RNG() : generator_(new Generator()) { }

Caffe::RNG::RNG(unsigned int seed) : generator_(new Generator(seed)) { }

Caffe::RNG& Caffe::RNG::operator=(const RNG& other) {
  generator_.reset(other.generator_.get());
  return *this;
}

void* Caffe::RNG::generator() {
  return static_cast<void*>(generator_->rng());
}

const char* cublasGetErrorString(cublasStatus_t error) {
  switch (error) {
  case CUBLAS_STATUS_SUCCESS:
    return "CUBLAS_STATUS_SUCCESS";
  case CUBLAS_STATUS_NOT_INITIALIZED:
    return "CUBLAS_STATUS_NOT_INITIALIZED";
  case CUBLAS_STATUS_ALLOC_FAILED:
    return "CUBLAS_STATUS_ALLOC_FAILED";
  case CUBLAS_STATUS_INVALID_VALUE:
    return "CUBLAS_STATUS_INVALID_VALUE";
  case CUBLAS_STATUS_ARCH_MISMATCH:
    return "CUBLAS_STATUS_ARCH_MISMATCH";
  case CUBLAS_STATUS_MAPPING_ERROR:
    return "CUBLAS_STATUS_MAPPING_ERROR";
  case CUBLAS_STATUS_EXECUTION_FAILED:
    return "CUBLAS_STATUS_EXECUTION_FAILED";
  case CUBLAS_STATUS_INTERNAL_ERROR:
    return "CUBLAS_STATUS_INTERNAL_ERROR";
#if CUDA_VERSION >= 6000
  case CUBLAS_STATUS_NOT_SUPPORTED:
    return "CUBLAS_STATUS_NOT_SUPPORTED";
#endif
#if CUDA_VERSION >= 6050
  case CUBLAS_STATUS_LICENSE_ERROR:
    return "CUBLAS_STATUS_LICENSE_ERROR";
#endif
  }
  return "Unknown cublas status";
}

const char* curandGetErrorString(curandStatus_t error) {
  switch (error) {
  case CURAND_STATUS_SUCCESS:
    return "CURAND_STATUS_SUCCESS";
  case CURAND_STATUS_VERSION_MISMATCH:
    return "CURAND_STATUS_VERSION_MISMATCH";
  case CURAND_STATUS_NOT_INITIALIZED:
    return "CURAND_STATUS_NOT_INITIALIZED";
  case CURAND_STATUS_ALLOCATION_FAILED:
    return "CURAND_STATUS_ALLOCATION_FAILED";
  case CURAND_STATUS_TYPE_ERROR:
    return "CURAND_STATUS_TYPE_ERROR";
  case CURAND_STATUS_OUT_OF_RANGE:
    return "CURAND_STATUS_OUT_OF_RANGE";
  case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
    return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";
  case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
    return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
  case CURAND_STATUS_LAUNCH_FAILURE:
    return "CURAND_STATUS_LAUNCH_FAILURE";
  case CURAND_STATUS_PREEXISTING_FAILURE:
    return "CURAND_STATUS_PREEXISTING_FAILURE";
  case CURAND_STATUS_INITIALIZATION_FAILED:
    return "CURAND_STATUS_INITIALIZATION_FAILED";
  case CURAND_STATUS_ARCH_MISMATCH:
    return "CURAND_STATUS_ARCH_MISMATCH";
  case CURAND_STATUS_INTERNAL_ERROR:
    return "CURAND_STATUS_INTERNAL_ERROR";
  }
  return "Unknown curand status";
}

#endif  // CPU_ONLY

const double TypedConsts<double>::zero = 0.0;
const double TypedConsts<double>::one = 1.0;

const float TypedConsts<float>::zero = 0.0f;
const float TypedConsts<float>::one = 1.0f;

#ifndef CPU_ONLY
const float16 TypedConsts<float16>::zero = 0.0f;
const float16 TypedConsts<float16>::one = 1.0f;
#endif

const int TypedConsts<int>::zero = 0;
const int TypedConsts<int>::one = 1;

#ifdef USE_CUDNN
CuDNNHandle::CuDNNHandle(cudaStream_t stream) {
  if (cudnnCreate(&handle_) != CUDNN_STATUS_SUCCESS) {
    LOG(ERROR) << "Cannot create cuDNN handle. cuDNN won't be available.";
  }
  CUDNN_CHECK(cudnnSetStream(handle_, stream));
}

CuDNNHandle::~CuDNNHandle() {
  CUDNN_CHECK(cudnnDestroy(handle_));
}
#endif


Caffe::Properties::Properties() :
      init_time_(std::time(nullptr)),
      main_thread_id_(std::this_thread::get_id()),
      caffe_version_(AS_STRING(CAFFE_VERSION)) {
// gcc 4.8.4
//  atomic_init(&counter1_, 0);
//  atomic_init(&counter2_, 0);
#ifndef CPU_ONLY
  int count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&count));
  compute_capabilities_.resize(count);
  cudaDeviceProp device_prop;
  for (int gpu = 0; gpu < compute_capabilities_.size(); ++gpu) {
    CUDA_CHECK(cudaGetDeviceProperties(&device_prop, gpu));
    compute_capabilities_[gpu] = device_prop.major * 100 + device_prop.minor;
    DLOG(INFO) << "GPU " << gpu << " '" << device_prop.name << "' has compute capability "
        << device_prop.major << "." << device_prop.minor;
  }
#ifdef USE_CUDNN
  cudnn_version_ =
      AS_STRING(CUDNN_MAJOR) "." AS_STRING(CUDNN_MINOR) "." AS_STRING(CUDNN_PATCHLEVEL);
#else
  cudnn_version_ = "USE_CUDNN is not defined";
#endif
  int cublas_version = 0;
  CUBLAS_CHECK(cublasGetVersion(Caffe::cublas_handle(), &cublas_version));
  cublas_version_ = std::to_string(cublas_version);

  int cuda_version = 0;
  CUDA_CHECK(cudaRuntimeGetVersion(&cuda_version));
  cuda_version_ = std::to_string(cuda_version);

  int cuda_driver_version = 0;
  CUDA_CHECK(cudaDriverGetVersion(&cuda_driver_version));
  cuda_driver_version_ = std::to_string(cuda_driver_version);
#endif
}

Caffe::Properties::~Properties() {
}

std::string Caffe::time_from_init() {
  std::ostringstream os;
  os.unsetf(std::ios_base::floatfield);
  os.precision(4);
  double span = std::difftime(std::time(NULL), init_time());
  const double mn = 60.;
  const double hr = 3600.;
  if (span < mn) {
    os << span << "s";
  } else if (span < hr) {
    int m = static_cast<int>(span / mn);
    double s = span - m * mn;
    os << m << "m " << s << "s";
  } else {
    int h = static_cast<int>(span / hr);
    int m = static_cast<int>((span - h * hr) / mn);
    double s = span - h * hr - m * mn;
    os << h << "h " << m << "m " << s << "s";
  }
  return os.str();
}

#ifndef CPU_ONLY
#ifndef NO_NVML
namespace nvml {

std::mutex NVMLInit::m_;

// set the CPU affinity for this GPU
void setCpuAffinity(unsigned int rank) {
  std::lock_guard<std::mutex> lock(NVMLInit::m_);
  static thread_local NVMLInit nvml_init_;
  bool result = false;
  unsigned int deviceCount = 0U;
  const std::vector<int>& gpus = Caffe::gpus();
  if (nvmlDeviceGetCount(&deviceCount) == NVML_SUCCESS) {
    CHECK_LT(rank, deviceCount);
    if (rank < deviceCount && rank < gpus.size() &&
        nvmlDeviceGetHandleByIndex(gpus[rank], &nvml_init_.device_) == NVML_SUCCESS) {
      if (nvmlDeviceSetCpuAffinity(nvml_init_.device_) == NVML_SUCCESS) {
        LOG(INFO) << "NVML succeeded to set CPU affinity on device " << gpus[rank];
        result = true;
      }
    }
  }
  if (!result && rank < gpus.size()) {
    LOG(ERROR) << "NVML failed to set CPU affinity on device " << gpus[rank];
  }
}

}  // namespace nvml
#endif  // NO_NVML
#endif

}  // namespace caffe
