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
  const float min_lr = this->param_.min_lr();
  CHECK_GE(min_lr, 0.F);
  if (this->iter_ < this->param_.rampup_interval()) {
    float alpha = float(this->iter_) / this->param_.rampup_interval();
    float rampup_lr = 0.;
    if (this->param_.has_rampup_lr()) {
      rampup_lr = this->param_.rampup_lr();
    }
    rate = rampup_lr + (this->param_.base_lr() - rampup_lr) * alpha * alpha;
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
    float base_lr = this->param_.base_lr();
    CHECK_GE(base_lr, min_lr);
    float power = this->param_.power();
    float maxiter = this->param_.max_iter() > 0 ? float(this->param_.max_iter()) : 1.F;
    rate = (base_lr - min_lr) * std::pow(1.F - (float(this->iter_) / maxiter), power) + min_lr;
  } else if (lr_policy == "sigmoid") {
    rate = this->param_.base_lr() / (1.F +
        std::exp(-this->param_.gamma() * (double(this->iter_ - this->param_.stepsize()))));
  } else {
    LOG(FATAL) << "Unknown learning rate policy: " << lr_policy;
  }
  if (rate < min_lr) {
    rate = min_lr;
  }
  return rate;
}

template<typename Dtype>
float SGDSolver<Dtype>::GetMomentum() {
  float moment;
  float base_momentum = this->param_.momentum();
  const string& momentum_policy = this->param_.momentum_policy();

  if (momentum_policy == "fixed") {
     moment = base_momentum;
  } else if (momentum_policy == "poly") {
    float max_momentum  = this->param_.max_momentum();
    float power = this->param_.momentum_power();
    moment = base_momentum + (max_momentum - base_momentum) *
           pow((float(this->iter_) / float(this->param_.max_iter())), power);
  } else if (momentum_policy == "opt") {
    float lr = GetLearningRate();
    moment = (1. - 0.5*std::sqrt(lr)) * (1. - 0.5*std::sqrt(lr));
    if (this->param_.has_max_momentum()) {
      float max_momentum  = this->param_.max_momentum();
      moment = std::min(max_momentum, moment);
    }
  } else {
    LOG(FATAL) << "Unknown momentum policy: " << momentum_policy;
  }
  return moment;
}

template<typename Dtype>
float SGDSolver<Dtype>::GetWeightDecay() const {
  float wd = this->param_.weight_decay();
  const string& wd_policy = this->param_.weight_decay_policy();
  float weight_decay = wd;
  if (wd_policy == "poly") {
    float power = this->param_.weight_decay_power();
    weight_decay = wd * pow(1.f - float(this->iter_)/this->param_.max_iter(), power);
  }
  return weight_decay;
}

template<typename Dtype>
float SGDSolver<Dtype>::local_decay(int param_id) const {
  const vector<float>& net_params_weight_decay = this->net_->params_weight_decay();
  float weight_decay = GetWeightDecay();
  weight_decay *= net_params_weight_decay[param_id];
  return weight_decay;
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
void SGDSolver<Dtype>::ClipGradientsAndNormalize(void* handle, int type_id,
    const std::set<int>& param_ids) {
  const float clip_gradients = this->param_.clip_gradients();
  if (clip_gradients < 0 && this->param_.iter_size() == 1) {
    return;
  }
  const float accum_normalization = 1.F / this->param_.iter_size();
  const vector<shared_ptr<Blob>>& net_params = this->net_->learnable_params();
  float sumsq_diff = 0.F;
  for (int param_id : param_ids) {
    sumsq_diff += net_params[param_id]->sumsq_diff(type_id);
  }

  const float l2norm_diff = std::sqrt(sumsq_diff);
  if (l2norm_diff > clip_gradients) {
    float scale_factor = clip_gradients / l2norm_diff;
    LOG(INFO) << "Gradient clipping: scaling down gradients (L2 norm " << l2norm_diff << " > "
              << clip_gradients << ") " << "by scale factor " << scale_factor;
    for (int param_id : param_ids) {
      net_params[param_id]->scale_diff(scale_factor * accum_normalization, handle);
    }
  } else if (this->param_.iter_size() != 1) {
    for (int param_id : param_ids) {
      net_params[param_id]->scale_diff(accum_normalization, handle);
    }
  }
}

template<typename Dtype>
void SGDSolver<Dtype>::PrintRate(float rate) {
  if (Caffe::root_solver() && this->param_.display() && this->iter_ % this->param_.display() == 0) {
    if (rate == 0.F) {
      rate = GetLearningRate();
    }
    float moment = GetMomentum();
    float wd = GetWeightDecay();
    LOG(INFO) << "Iteration " << this->iter_
    		  << ", lr = " << rate << ", m = " << moment
    		  << ", lrm = " << rate / (1.F- moment)
              << ", wd = " << wd  << ", gs = " << f_round2(net_->global_grad_scale());
  }
}

// Note: this is asynchronous call
template<typename Dtype>
float SGDSolver<Dtype>::ApplyUpdate(int param_id, void* handle, float rate, bool normalize,
    bool clear_grads) {
  if (normalize) {
    Normalize(param_id, handle);
  }
  Regularize(param_id);
  return ComputeUpdateValue(param_id, handle, rate, clear_grads);
}

template<typename Dtype>
void SGDSolver<Dtype>::Normalize(int param_id, void* handle) {
  if (this->param_.iter_size() == 1) { return; }
  // Scale gradient to counterbalance accumulation.
  const vector<shared_ptr<Blob>>& net_params = this->net_->learnable_params();
  const float accum_normalization = 1.F / this->param_.iter_size();
  net_params[param_id]->scale_diff(accum_normalization, handle);
}

template<typename Dtype>
void SGDSolver<Dtype>::Regularize(int param_id) {
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
    //Fused with ComputeUpdateValue
  } else {
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
}

template<typename Gtype, typename Wtype, typename Htype>
void sgd_reg_update_all_and_clear_gpu(int N,
    Gtype* g, Wtype* w, Htype* h,
    float momentum, float local_rate, const std::string& regularization_type, float local_decay,
    void* handle, bool clear_grads);

template<typename Dtype>
float SGDSolver<Dtype>::ComputeUpdateValue(int param_id, void* handle, float rate,
    bool clear_grads) {
  if (this->param_.debug_info()) {
    PrintParams(param_id);
  }
  Blob* param = this->net_->learnable_params()[param_id].get();
  TBlob<Dtype>* history = history_[param_id].get();
  float momentum = GetMomentum();
  float wgrad_sq = 0.F;

  const bool larc = this->param_.larc();
  const string& larc_policy = this->param_.larc_policy();
  float local_rate = GetLocalRate(param_id, wgrad_sq);
  if (larc) {
    if (larc_policy == "scale") {
      local_rate = rate * local_rate;
    } else if (larc_policy == "clip") {
      local_rate = std::min(rate, local_rate);
    } else {
      LOG(FATAL) << "Unknown larc policy: " << larc_policy;
    }
  } else {
    local_rate = rate * local_rate;
  }

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
    const std::string& regularization_type = this->param_.regularization_type();
    float decay = local_decay(param_id);
    const Type wtype = param->data_type();
    const Type gtype = param->diff_type();
    if (gtype == tp<float16>()) {
      sgd_reg_update_all_and_clear_gpu<float16, Dtype, Dtype>(param->count(),
          param->mutable_gpu_diff<float16>(),
          param->mutable_gpu_data<Dtype>(),
          history->mutable_gpu_data(),
          momentum, local_rate, regularization_type, decay,  handle, clear_grads);
    } else if (gtype == tp<float>()) {
      if (wtype == tp<float>()) {
        sgd_reg_update_all_and_clear_gpu<float, float, Dtype>(param->count(),
            param->mutable_gpu_diff<float>(),
            param->mutable_gpu_data<float>(),
            history->mutable_gpu_data(),
            momentum, local_rate, regularization_type, decay, handle, clear_grads);
      } else {
        sgd_reg_update_all_and_clear_gpu<float, Dtype, Dtype>(param->count(),
            param->mutable_gpu_diff<float>(),
            param->mutable_gpu_data<Dtype>(),
            history->mutable_gpu_data(),
            momentum, local_rate, regularization_type, decay, handle, clear_grads);
      }
    } else if (gtype == tp<double>()) {
      sgd_reg_update_all_and_clear_gpu<double, Dtype, Dtype>(param->count(),
          param->mutable_gpu_diff<double>(),
          param->mutable_gpu_data<Dtype>(),
          history->mutable_gpu_data(),
          momentum, local_rate, regularization_type, decay,  handle, clear_grads);
    } else {
      LOG(FATAL) << "Gradient type " << Type_Name(gtype) << " is not supported";
    }
  } else {
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
  return wgrad_sq;
}

template<typename Dtype>
float SGDSolver<Dtype>::GetLocalRate(int param_id, float& wgrad_sq) {
  const vector<float>& net_params_lr = this->net_->params_lr();
  float local_lr = net_params_lr[param_id];
  if (this->net_->global_grad_scale_enabled() || this->param_.larc()) {
    shared_ptr<Blob> param = this->net_->learnable_params()[param_id];
    const int type_id = net_->learnable_types()[0] == param->diff_type() ? 0 : 1;
    wgrad_sq = param->sumsq_diff(type_id);
    if (std::isnan(wgrad_sq)) {
      wgrad_sq = 0.F;  // skip this
    }
    if (this->param_.larc()) {
      const float wgrad_norm = std::sqrt(wgrad_sq);
      const float w_norm = std::sqrt(param->sumsq_data(type_id));
      const float gw_ratio = this->param_.larc_eta();
      float rate = 1.F;
      float momentum = this->GetMomentum();
      if (w_norm > 0.F && wgrad_norm > 0.F) {
        //float weight_decay = this->param_.weight_decay();
        //rate = gw_ratio * w_norm / (wgrad_norm + weight_decay * w_norm);
        rate = (1.F - momentum) * gw_ratio * w_norm / wgrad_norm;
      }
      if (local_lr > 0.) {
        local_lr = rate;
      }
#ifdef DEBUG
      if (Caffe::root_solver()
          && this->param_.display()
          && (this->iter_ % this->param_.display() == 0)) {
        const int layer_id = this->net_->param_layer_indices(param_id).first;
        const string &layer_name = this->net_->layer_names()[layer_id];
        const int blob_id = this->net_->param_layer_indices(param_id).second;
        LOG(INFO) << layer_name << "." << blob_id << " lr=" << local_lr
                  << " " << "\t  w=" << w_norm << "\t dw=" << wgrad_norm;
      }
#endif
    }
  }
  return local_lr;
}


template<typename Dtype>
void SGDSolver<Dtype>::PrintParams(int param_id) {
  if (Caffe::root_solver()
      && this->param_.display()
      && (this->iter_ % this->param_.display() == 0)) {
    const int layer_id = this->net_->param_layer_indices(param_id).first;
    const int blob_id  = this->net_->param_layer_indices(param_id).second;
    const string& layer_name = this->net_->layer_names()[layer_id];
    const string& layer_type = this->net_->layers()[layer_id]->type();
    shared_ptr<Blob> param = this->net_->learnable_params()[param_id];
    shared_ptr<TBlob<Dtype>> history = history_[param_id];

    if ((layer_type == "Convolution") || (layer_type == "InnerProduct")) {
      float w_norm = std::sqrt(param->sumsq_data());
      float wgrad_norm = std::sqrt(param->sumsq_diff());
      float h_norm = std::sqrt(history->sumsq_data());
      DLOG(INFO) << "SGD_update " << layer_name << "." <<  blob_id
          << " W=" << w_norm << " \tdW=" << wgrad_norm << " \tH="<< h_norm;
    }
  }
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
    history_[i]->ToProto(state.add_history(), param().store_blobs_in_old_format());
  }
  string snapshot_filename = Solver::SnapshotFilename(".solverstate", vector<float>());
  LOG(INFO) << "Snapshotting solver state to binary proto file " << snapshot_filename;
  WriteProtoToBinaryFile(state, snapshot_filename.c_str());
}

template<typename Dtype>
void SGDSolver<Dtype>::SnapshotSolverStateToHDF5(const string& model_filename) {
  string snapshot_filename = Solver::SnapshotFilename(".solverstate.h5", vector<float>());
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
  Caffe::set_restored_iter(this->iter_);
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
  Caffe::set_restored_iter(this->iter_);
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
