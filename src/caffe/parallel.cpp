#include <cuda_runtime.h>
#include <glog/logging.h>
#include <boost/thread.hpp>
#include <boost/thread/latch.hpp>

#include "caffe/caffe.hpp"
#include "caffe/parallel.hpp"

#ifdef USE_NCCL
#include "caffe/util/nccl.hpp"
#endif

#ifdef USE_MPI
#include "caffe/clusters.hpp"
#endif

namespace caffe {

unique_ptr<boost::barrier> P2PManager::dl_bar(new boost::barrier(1));
unique_ptr<boost::barrier> P2PManager::bar;
unique_ptr<boost::barrier> P2PManager::rbar0;
unique_ptr<boost::barrier> P2PManager::rbar1;

P2PManager::P2PManager(shared_ptr<Solver> root_solver,
    int nranks, const SolverParameter& solver_param) :
      nranks_(nranks),
      syncs_(nranks),
      root_solver_(root_solver) {
#ifndef USE_NCCL
  LOG(FATAL) << "USE_NCCL must be specified for multi-GPU mode";
#endif
  dl_bar.reset(new boost::barrier(nranks_));
  bar.reset(new boost::barrier(nranks_));
  rbar0.reset(new boost::barrier(nranks_));
  rbar1.reset(new boost::barrier(nranks_));
}

void P2PManager::Run(const vector<int>& gpus) {
#ifdef USE_NCCL
  CHECK_EQ(nranks_, gpus.size());
  CHECK_EQ(nranks_, Caffe::solver_count());
#ifdef USE_MPI
  if(Clusters::node_rank() == 0) {
    NCCL_CHECK(ncclGetUniqueId(&nccl_id_));
  }  
  MPI_Bcast((void*) &nccl_id_, sizeof(nccl_id_), MPI_BYTE, 0, MPI_COMM_WORLD);
#else
  NCCL_CHECK(ncclGetUniqueId(&nccl_id_));
#endif
#else
  LOG(FATAL) << "Multi-GPU execution not available - rebuild with USE_NCCL";
#endif  // USE_NCCL
  SolverParameter param = root_solver_->param();
  this->shared_ = make_shared<SharedScores<float>>(nranks_);
  for (int i = 0; i < gpus.size(); ++i) {
    param.set_device_id(gpus[i]);
    syncs_[i].reset(new P2PSync(this, root_solver_, i, gpus.size(), param));
#ifdef USE_NCCL
    syncs_[i]->aux_ = &nccl_id_;
#else
    LOG(FATAL) << "Multi-GPU execution not available - rebuild with USE_NCCL";
#endif  // USE_NCCL
    syncs_[i]->shared_ = this->shared_;
  }

  LOG(INFO)<< "Starting Optimization";

  for (int i = 0; i < syncs_.size(); ++i) {
    syncs_[i]->StartInternalThread(true, static_cast<uint64_t>(param.random_seed()));
  }
  for (int i = 0; i < syncs_.size(); ++i) {
    syncs_[i]->WaitAll();
  }

  std::ostringstream os;
  os.precision(4);
  float total_perf = this->root_solver_->perf_report(os, syncs_[0]->target_device_);
  LOG(INFO) << "Root " << os.str();
  for (int i = 1; i < syncs_.size(); ++i) {
    std::ostringstream os;
    os.precision(4);
    total_perf += syncs_[i]->solver_->perf_report(os, syncs_[i]->target_device_, 5 /* "Root " */);
    LOG(INFO) << os.str();
  }
  if (syncs_.size() > 1) {
    LOG(INFO) << "Overall multi-GPU performance: " << total_perf << " img/sec";
  }
}

void P2PManager::EarlyCancel(P2PSync* killed) {
  for (int i = 0; i < syncs_.size(); ++i) {
    syncs_[i]->solver_->request_early_exit();
    syncs_[i]->StopInternalThread(false);
  }
}

P2PSync::P2PSync(P2PManager* mgr, shared_ptr<Solver> root_solver,
    int rank, int nranks, const SolverParameter& solver_param)
    : InternalThread(solver_param.device_id(), rank, 1, false),
      mgr_(mgr),
      rank_(rank),
      nranks_(nranks),
      initial_iter_(root_solver->iter()),
      solver_(),
      root_solver_(root_solver),
      solver_param_(solver_param) {
#ifndef USE_NCCL
  LOG(FATAL) << "USE_NCCL := 1 must be specified for multi-GPU";
#endif
  CHECK_EQ(target_device_, solver_param_.device_id());
  LOG(INFO) << "[" << rank << " - " << this->target_device_ << "] P2pSync adding callback";
}

P2PSync::~P2PSync() {
#ifdef USE_NCCL
  ncclCommDestroy(nccl_comm_);
#endif
}

void P2PSync::InternalThreadEntry() {
  if (rank_ == 0) {
    Caffe::set_root_solver(true);
    solver_ = root_solver_;
    solver_->root_add_callback(this);
  } else {
    Caffe::set_root_solver(false);
    solver_.reset(caffe::SolverRegistry::CreateSolver(solver_param_, root_solver_.get(), rank_));
  }
  solver_->set_callback(this);

  CHECK_EQ(nranks_, Caffe::solver_count());
  CHECK_EQ(target_device_, Caffe::current_device());

#ifdef USE_NCCL
  ncclUniqueId* nccl_id = reinterpret_cast<ncclUniqueId*>(this->aux_);
  soft_barrier();
#ifdef USE_MPI
  if(Caffe::root_solver()) {
    MPI_Barrier(MPI_COMM_WORLD);
  }
  
  auto total = nranks_ * Clusters::node_count();
  auto nrank = Clusters::node_rank() * nranks_ + rank_;
  LOG(INFO) << "MPI world: ("<< Clusters::node_rank() << "/" << Clusters::node_count() << ") local:(" << Clusters::node_local_rank() <<"/" << Clusters::node_local_count() << ")";
  LOG(INFO) << "[" << nrank << " / " << total << "] P2pSync";
  NCCL_CHECK(ncclCommInitRank(&nccl_comm_, 
                              total,
                              *nccl_id, 
                              nrank));
  LOG(INFO) << "MPI barrier: ("<< Clusters::node_rank() << "/" << Clusters::node_count() << ") local:(" << Clusters::node_local_rank() <<"/" << Clusters::node_local_count() << ")";
  if(Caffe::root_solver()) {
    MPI_Barrier(MPI_COMM_WORLD);
  }  
#else
  NCCL_CHECK(ncclCommInitRank(&nccl_comm_, nranks_, *nccl_id, rank_));
#endif
  soft_barrier();
#endif
  comm_stream_[0] = CudaStream::create(true);
  comm_stream_[1] = CudaStream::create(true);

  LOG(INFO) << "[" << rank_ << " - " << target_device_ << "] P2pSync adding callback";
  // See if there is a defined seed and reset random state if so
  if (solver_->param().random_seed() >= 0) {
    // Fetch random seed and modulate by device ID to make sure
    // everyone doesn't have the same seed.  We seem to have some
    // solver instability if we have everyone with the same seed
    Caffe::set_random_seed(solver_->param().random_seed() + static_cast<uint64_t>(rank_));
  } else {
    // Or system generated one
    Caffe::set_random_seed(Caffe::SEED_NOT_SET);
  }

  if (solver_->Solve()) {
    mgr_->EarlyCancel(this);
  }
}

void P2PSync::soft_barrier() {
  // CPU barrier to avoid busy-polling on the GPU.
  P2PManager::bar_wait();
}

void P2PSync::reduce_barrier(int type_id) {
  P2PManager::rbar_wait(type_id);
}

void P2PSync::on_start(const vector<shared_ptr<Blob>>& net) {
#ifdef USE_NCCL
  int count = 0;
  NCCL_CHECK(ncclCommCount(nccl_comm_, &count));
#ifdef USE_MPI
  CHECK_EQ(count, nranks_ * Clusters::node_count());
#else
  CHECK_EQ(count, nranks_);
#endif

  for (int i = 0; i < net.size(); ++i) {
    Blob* param = net[i].get();
    const Type param_type = param->data_type();
    const int type_id = solver_->net()->learnable_types()[0] == param_type ? 0 : 1;
    reduce_barrier(type_id);
    NCCL_CHECK(ncclBcast(param->current_mutable_data_memory(true),
        even(param->count()),
        nccl::nccl_type(param_type),
        0,
        nccl_comm_,
        comm_stream(type_id)));
    CUDA_CHECK(cudaStreamSynchronize(comm_stream(type_id)));
    reduce_barrier(type_id);
  }
#endif  // USE_NCCL
}

void P2PSync::allreduce(int type_id, int param_id) {
#ifdef USE_NCCL
  const shared_ptr<Blob>& param = solver_->net()->learnable_params()[param_id];
  NCCL_CHECK(ncclAllReduce(param->current_diff_memory(true),
      param->current_mutable_diff_memory(true),
      even(param->count()),
      nccl::nccl_type(param->diff_type()),
      ncclSum,
      nccl_comm_,
      comm_stream(type_id)));
  CUDA_CHECK(cudaStreamSynchronize(comm_stream(type_id)));
#endif  // USE_NCCL
}

void P2PSync::allreduce_bucket(int type_id, size_t count, void* bucket, Type type) {
#ifdef USE_NCCL
  CHECK(bucket);
  NCCL_CHECK(ncclAllReduce(bucket,
      bucket,
      count,
      nccl::nccl_type(type),
      ncclSum,
      nccl_comm_,
      comm_stream(type_id)));
  CUDA_CHECK(cudaStreamSynchronize(comm_stream(type_id)));
#endif  // USE_NCCL
}

// master thread gets aggregate of results for output
void P2PSync::aggregateTestResults(float* loss, vector<float>* scores) {
  // only run on master thread
  if (this->rank_ == 0) {
    // initialize results
    *loss = 0.F;
    for (size_t i = 0; i < scores->size(); ++i) {
      (*scores)[i] = 0.F;
    }
    // all test threads
    for (size_t i = 0; i < nranks_; ++i) {
      vector<float>& shared_scr = shared_->rank_scores(this->rank_);
      *loss += shared_scr[0];
      // all scores within each test thread
      for (size_t j = 0; j < scores->size(); ++j) {
        (*scores)[j] += shared_scr[j+1];
      }
    }
  }
}

void P2PSync::saveTestResults(float loss, const vector<float>& scores) {
  vector<float>& shared_scr = shared_->rank_scores(this->rank_);
  CHECK_GE(shared_scr.size(), scores.size() + 1);
  shared_scr[0] = loss;
  for (size_t i = 0; i < scores.size(); ++i) {
    shared_scr[i+1] = scores[i];
  }
}

uint32_t batch_per_gpu(uint32_t total) {
  int solver_count = Caffe::solver_count();
  if (total == 0 || total % solver_count != 0) {
    uint32_t new_total = total + (solver_count - (total % solver_count));
    LOG(WARNING) << "Batch size must be divisible by the number of solvers (GPUs): "
        << "it's been adjusted from " << total << " to " << new_total;
    total = new_total;
  }
  return total / solver_count;
}

unsigned int P2PSync::divide_batch_size(NetParameter* net) {
  unsigned int ret = 0U;
  for (int i = 0; i < net->layer_size(); ++i) {
    if (net->layer(i).has_data_param()) {
      if (net->layer(i).data_param().has_batch_size()) {
        uint32_t total = net->layer(i).data_param().batch_size();
        uint32_t batch = batch_per_gpu(total);
        net->mutable_layer(i)->mutable_data_param()->set_batch_size(batch);
        ret = batch;
      }
    }
    if (net->layer(i).has_hdf5_data_param()) {
      if (net->layer(i).hdf5_data_param().has_batch_size()) {
        uint32_t total = net->layer(i).hdf5_data_param().batch_size();
        uint32_t batch = batch_per_gpu(total);
        net->mutable_layer(i)->mutable_hdf5_data_param()->set_batch_size(batch);
        if (ret == 0U) {
          ret = batch;
        }
      }
    }
    if (net->layer(i).has_image_data_param()) {
      if (net->layer(i).image_data_param().has_batch_size()) {
        uint32_t total = net->layer(i).image_data_param().batch_size();
        uint32_t batch = batch_per_gpu(total);
        net->mutable_layer(i)->mutable_image_data_param()->set_batch_size(batch);
        if (ret == 0U) {
          ret = batch;
        }
      }
    }
    if (net->layer(i).has_memory_data_param()) {
      if (net->layer(i).memory_data_param().has_batch_size()) {
        uint32_t total = net->layer(i).memory_data_param().batch_size();
        uint32_t batch = batch_per_gpu(total);
        net->mutable_layer(i)->mutable_memory_data_param()->set_batch_size(batch);
        if (ret == 0U) {
          ret = batch;
        }
      }
    }
    if (net->layer(i).has_window_data_param()) {
      if (net->layer(i).window_data_param().has_batch_size()) {
        uint32_t total = net->layer(i).window_data_param().batch_size();
        uint32_t batch = batch_per_gpu(total);
        net->mutable_layer(i)->mutable_window_data_param()->set_batch_size(batch);
        if (ret == 0U) {
          ret = batch;
        }
      }
    }
  }
  return ret;
}

}  // namespace caffe
