#ifndef CPU_ONLY
#include <cuda_runtime.h>
#include "caffe/util/coll.h"
#endif
#include <glog/logging.h>
#include <stdio.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

#include "boost/thread.hpp"
#include "boost/thread/latch.hpp"
#include "caffe/caffe.hpp"
#include "caffe/parallel.hpp"
#include "caffe/util/gpu_memory.hpp"

#define GRID_DIM 8

namespace caffe {

boost::barrier *bar;

enum Op {
  copy,
  replace_cpu,
  replace_gpu,
  replace_cpu_diff,
  replace_gpu_diff
};

template<typename Dtype>
static void apply_buffers(const vector<Blob<Dtype>*>& blobs,
                          Dtype* buffer, size_t total_size, Op op) {
  Dtype* ptr = buffer;
  for (int i = 0; i < blobs.size(); ++i) {
    int size = blobs[i]->count();
    switch (op) {
      case copy: {
        // Init buffer to current values of blobs
        caffe_copy(size,
                   reinterpret_cast<const Dtype*>(blobs[i]->data()->cpu_data()),
                   ptr);
        break;
      }
      case replace_cpu:
        blobs[i]->data()->set_cpu_data(ptr);
        break;
      case replace_gpu:
        blobs[i]->data()->set_gpu_data(ptr);
        break;
      case replace_cpu_diff:
        blobs[i]->diff()->set_cpu_data(ptr);
        break;
      case replace_gpu_diff:
        blobs[i]->diff()->set_gpu_data(ptr);
        break;
    }
    ptr += size;
  }
  // total_size is at least one byte
  CHECK_EQ(total_size, (ptr == buffer ? 1 : ptr - buffer));
}

// Buffer size necessary to store given blobs
template<typename Dtype>
static size_t total_size(const vector<Blob<Dtype>*>& params) {
  size_t size = 0;
  for (int i = 0; i < params.size(); ++i)
    size += params[i]->count();
  // Size have at least one byte, otherwise cudaMalloc fails if net has no
  // learnable parameters.
  return (size > 0) ? size : 1;
}

template<typename Dtype>
Params<Dtype>::Params(shared_ptr<Solver<Dtype> > root_solver)
    : size_(total_size<Dtype>(root_solver->net()->learnable_params())),
      data_(NULL),
      diff_(NULL) {
}

template<typename Dtype>
GPUParams<Dtype>::GPUParams(shared_ptr<Solver<Dtype> > root_solver, int device)
    : Params<Dtype>(root_solver) {
#ifndef CPU_ONLY
  int initial_device;
  CUDA_CHECK(cudaGetDevice(&initial_device));

  // Allocate device buffers
  CUDA_CHECK(cudaSetDevice(device));
  buffer_device_ = device;
  // CUDA_CHECK(cudaMalloc(&data_, size_ * sizeof(Dtype)));
  gpu_memory::allocate(reinterpret_cast<void **>(&data_),
                           size_ * sizeof(Dtype));

  // Copy blob values
  const vector<Blob<Dtype>*>& net =
      root_solver->net()->learnable_params();
  apply_buffers(net, data_, size_, copy);

  // CUDA_CHECK(cudaMalloc(&diff_, size_ * sizeof(Dtype)));
  gpu_memory::allocate(reinterpret_cast<void **>(&diff_),
                           size_ * sizeof(Dtype));
  caffe_gpu_set(size_, Dtype(0), diff_);

  CUDA_CHECK(cudaSetDevice(initial_device));
#else
  NO_GPU;
#endif
}

template<typename Dtype>
GPUParams<Dtype>::~GPUParams() {
#ifndef CPU_ONLY
  int initial_device;
  CUDA_CHECK(cudaGetDevice(&initial_device));
  CUDA_CHECK(cudaSetDevice(buffer_device_));
  gpu_memory::deallocate(data_);
  gpu_memory::deallocate(diff_);
  data_ = NULL;
  diff_ = NULL;
  CUDA_CHECK(cudaSetDevice(initial_device));
#endif
}

template<typename Dtype>
void GPUParams<Dtype>::configure(Solver<Dtype>* solver) const {
  const vector<Blob<Dtype>*>& net =
      solver->net()->learnable_params();
  apply_buffers(net, data_, size_, replace_gpu);
  apply_buffers(net, diff_, size_, replace_gpu_diff);
}

void DevicePair::compute(const vector<int> devices, vector<DevicePair>* pairs) {
#ifndef CPU_ONLY
  pairs->push_back(DevicePair(-1, devices[0]));
  for (int i = 0; i < devices.size()-1; i++) {
    pairs->push_back(DevicePair(devices[i], devices[i+1]));
  }
#else
  NO_GPU;
#endif
}

//

template<typename Dtype>
P2PSync<Dtype>::P2PSync(shared_ptr<Solver<Dtype> > root_solver,
                        int rank, int nranks, const SolverParameter& param)
    : GPUParams<Dtype>(root_solver, param.device_id()),
      rank_(rank),
      nranks_(nranks),
      parent_(),
      child_(),
      queue_(),
      initial_iter_(root_solver->iter()),
      solver_(),
      params_(param) {
#ifndef CPU_ONLY
  int initial_device;
  CUDA_CHECK(cudaGetDevice(&initial_device));
  const int self = param.device_id();
  CUDA_CHECK(cudaSetDevice(self));

  if (rank == 0) {
    solver_ = root_solver;
  } else {
    Caffe::set_root_solver(false);
    solver_.reset(GetSolver<Dtype>(param, root_solver.get()));
    Caffe::set_root_solver(true);
  }
  this->configure(solver_.get());
  solver_->add_callback(this);

  CUDA_CHECK(cudaSetDevice(initial_device));
#else
  NO_GPU;
#endif
}

template<typename Dtype>
void P2PSync<Dtype>::SetupP2PAccess() {
#ifndef CPU_ONLY
  int initial_device;
  CUDA_CHECK(cudaGetDevice(&initial_device));
  const int self = solver_->param().device_id();
  CUDA_CHECK(cudaSetDevice(self));

  cudaStreamCreateWithFlags(&cuda_stream_, cudaStreamNonBlocking);

  gpu_memory::allocate(reinterpret_cast<void **>(&parent_grads_),
                       size_ * sizeof(Dtype));
  gpu_memory::allocate(reinterpret_cast<void **>(&offset_),
                       GRID_DIM*sizeof(int));
  caffe_gpu_memset(GRID_DIM*sizeof(int), -1, offset_ );

  if (parent_) {
    // Enable p2p access between devices
    const int peer = parent_->solver_->param().device_id();
    CUDA_CHECK(cudaDeviceCanAccessPeer(&parent_peer_access_, self, peer));
    if (parent_peer_access_) {
      CUDA_CHECK(cudaDeviceEnablePeerAccess(peer, 0));
    } else {
      LOG(INFO)<< "GPU " << self << " does not have p2p access to GPU " << peer;
    }
  }

  if (child_) {
    const int peer = child_->solver_->param().device_id();
    CUDA_CHECK(cudaDeviceCanAccessPeer(&child_peer_access_, self, peer));
    if (child_peer_access_) {
      if (child_ != parent_) CUDA_CHECK(cudaDeviceEnablePeerAccess(peer, 0));
    } else {
      LOG(INFO)<< "GPU " << self << " does not have p2p access to GPU " << peer;
    }
  }
  CUDA_CHECK(cudaSetDevice(initial_device));
#else
  NO_GPU;
#endif
}

template<typename Dtype>
P2PSync<Dtype>::~P2PSync() {
#ifndef CPU_ONLY
  int initial_device;
  CUDA_CHECK(cudaGetDevice(&initial_device));
  const int self = solver_->param().device_id();
  CUDA_CHECK(cudaSetDevice(self));

  cudaStreamDestroy(cuda_stream_);

  gpu_memory::deallocate(parent_grads_);
  gpu_memory::deallocate(offset_);

  if (child_) {
    const int peer = child_->solver_->param().device_id();
    if (child_peer_access_) {
      CUDA_CHECK(cudaDeviceDisablePeerAccess(peer));
    }
  }
  if (parent_) {
    const int peer = parent_->solver_->param().device_id();
    if (parent_peer_access_) {
      if (child_ != parent_) CUDA_CHECK(cudaDeviceDisablePeerAccess(peer));
    }
  }

  CUDA_CHECK(cudaSetDevice(initial_device));
#endif
}

template<typename Dtype>
void P2PSync<Dtype>::InternalThreadEntry() {
  Caffe::SetDevice(solver_->param().device_id());
  CHECK(Caffe::root_solver());
  Caffe::set_root_solver(false);
  // See if there is a defined seed and reset random state if so
  if (solver_->param().random_seed() >= 0) {
    // Fetch random seed and modulate by device ID to make sure
    // everyone doesn't have the same seed.  We seem to have some
    // solver instability if we have everyone with the same seed
    Caffe::set_random_seed(
        solver_->param().random_seed() + solver_->param().device_id());
  }
  solver_->Step(solver_->param().max_iter() - initial_iter_);
}

template<typename Dtype>
void P2PSync<Dtype>::soft_barrier() {
#ifndef CPU_ONLY
  // CPU barrier to avoid busy-polling on the GPU.
  CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
  if (rank_ != nranks_ - 1) queue_.pop();
  if (rank_) parent_->queue_.push(this);
  if (rank_) queue_.pop();
  if (rank_ != nranks_ - 1) child_->queue_.push(this);
#endif
}

template<typename Dtype>
void P2PSync<Dtype>::on_start() {
#ifndef CPU_ONLY
  CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
  multi_gpu_pipeline_bcast(
    rank_ ? offset_ : NULL,
    data_,
    ((rank_ != nranks_ - 1) && child_peer_access_) ? child_->offset_ : NULL,
    ((rank_ != nranks_ - 1) && child_peer_access_) ? child_->data_ : NULL,
    size_,
    GRID_DIM,
    cuda_stream_);
  if ((rank_ != nranks_ - 1) && !child_peer_access_) {
    CUDA_CHECK(cudaMemcpyAsync(child_->data_, data_,
          size_ * sizeof(Dtype), cudaMemcpyDeviceToDevice, cuda_stream_));
    for (int i = 0; i< GRID_DIM; i++) {
      CUDA_CHECK(cudaMemcpyAsync(child_->offset_+i, &size_,
            sizeof(int), cudaMemcpyHostToDevice, cuda_stream_));
    }
  }
  CUDA_CHECK(cudaStreamSynchronize(cuda_stream_));
  caffe_gpu_memset(GRID_DIM*sizeof(int), -1, offset_);
#endif
}

template<typename Dtype>
void P2PSync<Dtype>::allreduce() {
  bar->wait();
#ifndef CPU_ONLY
  CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
  multi_gpu_ring_sum(
    rank_,
    nranks_,
    offset_,
    diff_,
    parent_grads_,
    parent_->offset_,
    parent_->diff_,
    parent_->parent_grads_,
    (Dtype)1.0 / Caffe::solver_count(),  // Mult factor
    size_,
    GRID_DIM,
    cuda_stream_);
  CUDA_CHECK(cudaStreamSynchronize(cuda_stream_));
#endif
}

template<typename Dtype>
void P2PSync<Dtype>::run(shared_ptr<Solver<Dtype> > root,
                         const vector<int>& gpus) {
  int nranks = gpus.size();
  bar = new boost::barrier(nranks);
  SolverParameter param(this->params_);
  vector<shared_ptr<P2PSync<Dtype> > > syncs(nranks);

  for (int i = 1; i < nranks; i++) {
    param.set_device_id(gpus[i]);
    // Set parent_/child_
    syncs[i].reset(new P2PSync<Dtype>(root, i, nranks, param));
  }

  syncs[1].get()->parent_ = this;
  this->child_ = syncs[1].get();
  for (int i = 1; i < syncs.size()-1; ++i) {
    syncs[(i+1)%nranks].get()->parent_ = syncs[i].get();
    syncs[i].get()->child_ = syncs[(i+1)%nranks].get();
  }
  this->parent_ = syncs[syncs.size()-1].get();
  syncs[syncs.size()-1]->child_ = this;

  this->SetupP2PAccess();
  for (int i = 1; i < syncs.size(); ++i) {
    syncs[i]->SetupP2PAccess();
  }

  LOG(INFO)<< "Starting Optimization";

  for (int i = 1; i < syncs.size(); ++i) {
    syncs[i]->StartInternalThread();
  }

  // Run root solver on current thread
  this->solver_->Solve();

  for (int i = 1; i < syncs.size(); ++i) {
    syncs[i]->StopInternalThread();
  }
}

template<typename Dtype>
void P2PSync<Dtype>::divide_batch_size(NetParameter* net) {
  int solver_count = Caffe::solver_count();
  for (int i = 0; i < net->layer_size(); ++i) {
    string m = "Batch size must be divisible by the number of solvers (GPUs)";
    if (net->layer(i).has_data_param()) {
      if (net->layer(i).data_param().has_batch_size()) {
        uint32_t total = net->layer(i).data_param().batch_size();
        uint32_t batch = total / solver_count;
        CHECK(batch * solver_count == total) << m;
        net->mutable_layer(i)->mutable_data_param()->set_batch_size(batch);
      }
    }
    if (net->layer(i).has_hdf5_data_param()) {
      if (net->layer(i).hdf5_data_param().has_batch_size()) {
        uint32_t total = net->layer(i).hdf5_data_param().batch_size();
        uint32_t batch = total / solver_count;
        CHECK(batch * solver_count == total) << m;
        net->mutable_layer(i)->mutable_hdf5_data_param()->set_batch_size(batch);
      }
    }
    if (net->layer(i).has_image_data_param()) {
      if (net->layer(i).image_data_param().has_batch_size()) {
        uint32_t total = net->layer(i).image_data_param().batch_size();
        uint32_t batch = total / solver_count;
        CHECK(batch * solver_count == total) << m;
        net->mutable_layer(i)->mutable_image_data_param()->set_batch_size(
            batch);
      }
    }
    if (net->layer(i).has_memory_data_param()) {
      if (net->layer(i).memory_data_param().has_batch_size()) {
        uint32_t total = net->layer(i).memory_data_param().batch_size();
        uint32_t batch = total / solver_count;
        CHECK(batch * solver_count == total) << m;
        net->mutable_layer(i)->mutable_memory_data_param()->set_batch_size(
            batch);
      }
    }
    if (net->layer(i).has_window_data_param()) {
      if (net->layer(i).window_data_param().has_batch_size()) {
        uint32_t total = net->layer(i).window_data_param().batch_size();
        uint32_t batch = total / solver_count;
        CHECK(batch * solver_count == total) << m;
        net->mutable_layer(i)->mutable_window_data_param()->set_batch_size(
            batch);
      }
    }
  }
}

INSTANTIATE_CLASS(Params);
INSTANTIATE_CLASS(GPUParams);
INSTANTIATE_CLASS(P2PSync);

}  // namespace caffe
