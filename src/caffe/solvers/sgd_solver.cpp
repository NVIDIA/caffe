#include "caffe/sgd_solvers.hpp"
#include "caffe/util/hdf5.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/upgrade_proto.hpp"

namespace caffe {

// Return the current learning rate. The currently implemented learning rate
// policies are as follows:
//    - fixed: always return base_lr.
//    - step: return base_lr * gamma ^ (floor(iter / step))
//    - exp: return base_lr * gamma ^ iter
//    - inv: return base_lr * (1 + gamma * iter) ^ (- power)
//    - multistep: similar to step but it allows non uniform steps defined by
//      stepvalue
//    - poly: the effective learning rate follows a polynomial decay, to be
//      zero by the max_iter. return base_lr (1 - iter/max_iter) ^ (power)
//    - sigmoid: the effective learning rate follows a sigmod decay
//      return base_lr ( 1/(1 + exp(-gamma * (iter - stepsize))))
//
// where base_lr, max_iter, gamma, step, stepvalue and power are defined
// in the solver parameter protocol buffer, and iter is the current iteration.
template<typename Dtype>
float SGDSolver<Dtype>::GetLearningRate() {
  float rate;
  const string& lr_policy = this->param_.lr_policy();
  if (this->iter_ < this->param_.rampup_interval()) {
    if (this->param_.has_rampup_lr()) {
      rate = this->param_.rampup_lr();
    } else {
      rate = this->param_.base_lr() * float(this->iter_) / float(this->param_.rampup_interval());
    }
  } else if (lr_policy == "fixed") {
    rate = this->param_.base_lr();
  } else if (lr_policy == "step") {
    this->current_step_ = this->iter_ / this->param_.stepsize();
    rate = this->param_.base_lr() * pow(this->param_.gamma(), this->current_step_);
  } else if (lr_policy == "exp") {
    rate = this->param_.base_lr() * pow(this->param_.gamma(), this->iter_);
  } else if (lr_policy == "inv") {
    rate = this->param_.base_lr() *
           pow(1.F + this->param_.gamma() * float(this->iter_), -this->param_.power());
  } else if (lr_policy == "multistep") {
    if (this->current_step_ < this->param_.stepvalue_size() &&
        this->iter_ >= this->param_.stepvalue(this->current_step_)) {
      this->current_step_++;
      LOG(INFO) << "MultiStep Status: Iteration " << this->iter_ << ", step = "
                << this->current_step_;
    }
    rate = this->param_.base_lr() * pow(this->param_.gamma(), this->current_step_);
  } else if (lr_policy == "poly") {
  float min_lr  = this->param_.min_lr();
  float base_lr = this->param_.base_lr();
  float power = this->param_.power();
  rate = min_lr + (base_lr - min_lr) *
      pow(1.F - (float(this->iter_) / float(this->param_.max_iter())), power);
  } else if (lr_policy == "sigmoid") {
    rate = this->param_.base_lr() / (1.F +
        exp(-this->param_.gamma() * (double(this->iter_ - this->param_.stepsize()))));
  } else {
    LOG(FATAL) << "Unknown learning rate policy: " << lr_policy;
  }
  return rate;
}

template<typename Dtype>
void SGDSolver<Dtype>::PreSolve() {
  // Initialize the history
  const vector<shared_ptr<Blob>>& net_params = this->net_->learnable_params();
  history_.clear();
  update_.clear();
  temp_.clear();

  for (int i = 0; i < net_params.size(); ++i) {
    const vector<int>& shape = net_params[i]->shape();
    history_.emplace_back(boost::make_shared<TBlob<Dtype>>(shape));
    update_.emplace_back(boost::make_shared<TBlob<Dtype>>(shape));
    temp_.emplace_back(boost::make_shared<TBlob<Dtype>>(shape));
  }
}

template<typename Dtype>
void SGDSolver<Dtype>::ClipGradients(void* handle) {
  const float clip_gradients = this->param_.clip_gradients();
  if (clip_gradients < 0) { return; }
  const vector<shared_ptr<Blob>>& net_params = this->net_->learnable_params();
  float sumsq_diff = 0.F;
  for (int i = 0; i < net_params.size(); ++i) {
    sumsq_diff += net_params[i]->sumsq_diff();
  }
  const float l2norm_diff = std::sqrt(sumsq_diff);
  if (l2norm_diff > clip_gradients) {
    float scale_factor = clip_gradients / l2norm_diff;
    LOG(INFO) << "Gradient clipping: scaling down gradients (L2 norm " << l2norm_diff << " > "
              << clip_gradients << ") " << "by scale factor " << scale_factor;
    for (int i = 0; i < net_params.size(); ++i) {
      net_params[i]->scale_diff(scale_factor, handle, false);
    }
  }
}

template<typename Dtype>
void SGDSolver<Dtype>::PrintRate(float rate) {
  if (Caffe::root_solver() && this->param_.display() && this->iter_ % this->param_.display() == 0) {
    if (rate == 0.F) {
      rate = GetLearningRate();
    }
    LOG(INFO) << "Iteration " << this->iter_ << ", lr = " << rate;
  }
}

// Note: this is asynchronous call
template<typename Dtype>
void SGDSolver<Dtype>::ApplyUpdate(int param_id, void* handle, bool clear_grads) {
  float rate = GetLearningRate();  // TODO take it out
  ClipGradients(handle);
  Normalize(param_id, handle);
  Regularize(param_id, handle);
  ComputeUpdateValue(param_id, handle, rate, clear_grads, false);
}

template<typename Dtype>
void SGDSolver<Dtype>::Normalize(int param_id, void* handle) {
  if (this->param_.iter_size() == 1) { return; }
  // Scale gradient to counterbalance accumulation.
  const vector<shared_ptr<Blob>>& net_params = this->net_->learnable_params();
  const float accum_normalization = 1.F / this->param_.iter_size();
  net_params[param_id]->scale_diff(accum_normalization, handle, false);
}

template<typename Dtype>
void SGDSolver<Dtype>::Regularize(int param_id, void* handle) {
  if (Caffe::mode() == Caffe::CPU) {
    const vector<shared_ptr<Blob>>& net_params = this->net_->learnable_params();
    const vector<float>& net_params_weight_decay = this->net_->params_weight_decay();
    float weight_decay = this->param_.weight_decay();
    string regularization_type = this->param_.regularization_type();
    float local_decay = weight_decay * net_params_weight_decay[param_id];
    if (local_decay) {
      if (regularization_type == "L2") {
        // add weight decay
        caffe_axpy<Dtype>(net_params[param_id]->count(), local_decay,
            net_params[param_id]->cpu_data<Dtype>(),
            net_params[param_id]->mutable_cpu_diff<Dtype>());
      } else if (regularization_type == "L1") {
        caffe_cpu_sign<Dtype>(net_params[param_id]->count(),
            net_params[param_id]->cpu_data<Dtype>(), temp_[param_id]->mutable_cpu_data());
        caffe_axpy<Dtype>(net_params[param_id]->count(), local_decay, temp_[param_id]->cpu_data(),
            net_params[param_id]->mutable_cpu_diff<Dtype>());
      } else {
        LOG(FATAL) << "Unknown regularization type: " << regularization_type;
      }
    }
  } else if (Caffe::mode() == Caffe::GPU) {
#ifndef CPU_ONLY
    //Fused with ComputeUpdateValue
#else
    NO_GPU;
#endif
  } else {
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
}

#ifndef CPU_ONLY
template<typename Gtype, typename Wtype>
void sgd_reg_update_all_and_clear_gpu(int N,
    Gtype* g, Wtype* w, Wtype* h,
    float momentum, float local_rate, const std::string& regularization_type, float local_decay,
    void* handle, bool clear_grads, bool synced = true);
#endif


template<typename Dtype>
void
SGDSolver<Dtype>::ComputeUpdateValue(int param_id, void* handle, float rate, bool clear_grads,
    bool synced) {
  shared_ptr<Blob> param = this->net_->learnable_params()[param_id];
  shared_ptr<TBlob<Dtype>> history = history_[param_id];
  const vector<float>& net_params_lr = this->net_->params_lr();
  float momentum = this->param_.momentum();
  float local_rate = rate * net_params_lr[param_id];
  // Compute the update to history, then copy it to the parameter diff.
  if (Caffe::mode() == Caffe::CPU) {
    caffe_cpu_axpby<Dtype>(param->count(), local_rate, param->cpu_diff<Dtype>(), momentum,
        history->mutable_cpu_data());
    caffe_copy<Dtype>(param->count(), history->cpu_data(), param->mutable_cpu_diff<Dtype>());
    param->Update();
    if (clear_grads) {
      param->set_diff(0.F);
    }
  } else if (Caffe::mode() == Caffe::GPU) {
#ifndef CPU_ONLY
    const std::string& regularization_type = this->param_.regularization_type();
    const float decay = local_decay(param_id);
    const Type gtype = param->diff_type();
    if (gtype == tp<float16>()) {
      sgd_reg_update_all_and_clear_gpu<float16, Dtype>(param->count(),
          param->mutable_gpu_diff<float16>(),
          param->mutable_gpu_data<Dtype>(),
          history->mutable_gpu_data(),
          momentum, local_rate, regularization_type, decay,  handle, clear_grads, synced);
    } else if (gtype == tp<float>()) {
      sgd_reg_update_all_and_clear_gpu<float, Dtype>(param->count(),
          param->mutable_gpu_diff<float>(),
          param->mutable_gpu_data<Dtype>(),
          history->mutable_gpu_data(),
          momentum, local_rate, regularization_type, decay,  handle, clear_grads, synced);
    } else if (gtype == tp<double>()) {
      sgd_reg_update_all_and_clear_gpu<double, Dtype>(param->count(),
          param->mutable_gpu_diff<double>(),
          param->mutable_gpu_data<Dtype>(),
          history->mutable_gpu_data(),
          momentum, local_rate, regularization_type, decay,  handle, clear_grads, synced);
    } else {
      LOG(FATAL) << "Gradient type " << Type_Name(gtype) << " is not supported";
    }
#else
    NO_GPU;
#endif
  } else {
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
}

template<typename Dtype>
float SGDSolver<Dtype>::local_decay(int param_id) const {
  const vector<float>& net_params_weight_decay = this->net_->params_weight_decay();
  float weight_decay = this->param_.weight_decay();
  return weight_decay * net_params_weight_decay[param_id];
}

template<typename Dtype>
void SGDSolver<Dtype>::SnapshotSolverState(const string& model_filename) {
  switch (this->param_.snapshot_format()) {
    case caffe::SolverParameter_SnapshotFormat_BINARYPROTO:
      SnapshotSolverStateToBinaryProto(model_filename);
      break;
    case caffe::SolverParameter_SnapshotFormat_HDF5:
      SnapshotSolverStateToHDF5(model_filename);
      break;
    default:
      LOG(FATAL) << "Unsupported snapshot format.";
  }
}

template<typename Dtype>
void SGDSolver<Dtype>::SnapshotSolverStateToBinaryProto(const string& model_filename) {
  SolverState state;
  state.set_iter(this->iter_);
  state.set_learned_net(model_filename);
  state.set_current_step(this->current_step_);
  state.clear_history();
  for (int i = 0; i < history_.size(); ++i) {
    // Add history
    BlobProto* history_blob = state.add_history();
    history_[i]->template ToProto<Dtype>(history_blob);
  }
  string snapshot_filename = Solver::SnapshotFilename(".solverstate");
  LOG(INFO) << "Snapshotting solver state to binary proto file " << snapshot_filename;
  WriteProtoToBinaryFile(state, snapshot_filename.c_str());
}

template<typename Dtype>
void SGDSolver<Dtype>::SnapshotSolverStateToHDF5(const string& model_filename) {
  string snapshot_filename = Solver::SnapshotFilename(".solverstate.h5");
  LOG(INFO) << "Snapshotting solver state to HDF5 file " << snapshot_filename;
  hid_t file_hid = H5Fcreate(snapshot_filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  CHECK_GE(file_hid, 0) << "Couldn't open " << snapshot_filename << " to save solver state.";
  hdf5_save_int(file_hid, "iter", this->iter_);
  hdf5_save_string(file_hid, "learned_net", model_filename);
  hdf5_save_int(file_hid, "current_step", this->current_step_);
  hid_t history_hid = H5Gcreate2(file_hid, "history", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  CHECK_GE(history_hid, 0) << "Error saving solver state to " << snapshot_filename << ".";
  for (int i = 0; i < history_.size(); ++i) {
    ostringstream oss;
    oss << i;
    hdf5_save_nd_dataset(history_hid, oss.str(), *history_[i]);
  }
  H5Gclose(history_hid);
  H5Fclose(file_hid);
}

template<typename Dtype>
void SGDSolver<Dtype>::RestoreSolverStateFromBinaryProto(const string& state_file) {
  SolverState state;
  ReadProtoFromBinaryFile(state_file, &state);
  this->iter_ = state.iter();
  LOG(INFO) << "restore iter is " << this->iter_;
  Caffe::set_restored_iter(state.iter());
  this->iterations_restored_ = this->iter_;
  this->iterations_last_ = -1;  // for correct benchmarking
  if (state.has_learned_net()) {
    NetParameter net_param;
    ReadNetParamsFromBinaryFileOrDie(state.learned_net().c_str(), &net_param);
    this->net_->CopyTrainedLayersFrom(net_param);
  }
  this->current_step_ = state.current_step();
  CHECK_EQ(state.history_size(), history_.size()) << "Incorrect length of history blobs.";
  LOG(INFO) << "SGDSolver: restoring history";
  for (int i = 0; i < history_.size(); ++i) {
    history_[i]->FromProto(state.history(i));
  }
}

template<typename Dtype>
void SGDSolver<Dtype>::RestoreSolverStateFromHDF5(const string& state_file) {
  hid_t file_hid = H5Fopen(state_file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  CHECK_GE(file_hid, 0) << "Couldn't open solver state file " << state_file;
  this->iter_ = hdf5_load_int(file_hid, "iter");
  this->iterations_restored_ = this->iter_;
  this->iterations_last_ = -1;  // for correct benchmarking
  if (H5LTfind_dataset(file_hid, "learned_net")) {
    string learned_net = hdf5_load_string(file_hid, "learned_net");
    this->net_->CopyTrainedLayersFrom(learned_net);
  }
  this->current_step_ = hdf5_load_int(file_hid, "current_step");
  hid_t history_hid = H5Gopen2(file_hid, "history", H5P_DEFAULT);
  CHECK_GE(history_hid, 0) << "Error reading history from " << state_file;
  int state_history_size = hdf5_get_num_links(history_hid);
  CHECK_EQ(state_history_size, history_.size()) << "Incorrect length of history blobs.";
  for (int i = 0; i < history_.size(); ++i) {
    ostringstream oss;
    oss << i;
    hdf5_load_nd_dataset(history_hid, oss.str().c_str(), 0, kMaxBlobAxes, history_[i].get());
  }
  H5Gclose(history_hid);
  H5Fclose(file_hid);
}

INSTANTIATE_CLASS(SGDSolver);

REGISTER_SOLVER_CLASS(SGD);

}  // namespace caffe
