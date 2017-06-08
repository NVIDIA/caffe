#include <cstdio>

#include <string>
#include <vector>

#include <boost/thread.hpp>
#include "caffe/solver.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/gpu_memory.hpp"
#include "caffe/util/hdf5.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/upgrade_proto.hpp"

namespace caffe {

void Solver::SetActionFunction(ActionCallback func) {
  action_request_function_ = func;
}

SolverAction::Enum Solver::GetRequestedAction() {
  if (action_request_function_) {
    // If the external request function has been set, call it.
    return action_request_function_();
  }
  return SolverAction::NONE;
}

Solver::Solver(const SolverParameter& param, size_t rank, const Solver* root_solver)
    : param_(param), data_type_(param_.solver_data_type()), iter_(0), id_(0), net_(),
      callback_(nullptr), root_solver_(root_solver), rank_(rank), requested_early_exit_(false),
      iteration_timer_(), test_timer_(), iterations_last_(0), iterations_restored_(0),
      iter_size_complete_(false) {
  Init();
}

Solver::Solver(const string& param_file, size_t rank, const Solver* root_solver)
    : Solver(ReadSolverParamsFromTextFileOrDie(param_file), rank, root_solver) {}

Solver::~Solver() {
}

void Solver::Init() {
  LOG(INFO) << "Solver data type: " << Type_Name(data_type_);
  CHECK(Caffe::root_solver() || root_solver_)
      << "root_solver_ needs to be set for all non-root solvers";
  LOG_IF(INFO, Caffe::root_solver()) << "Initializing solver from parameters: "
    << std::endl << param_.DebugString();

  CHECK_GE(param_.average_loss(), 1) << "average_loss should be non-negative.";
  CheckSnapshotWritePermissions();
  if (Caffe::root_solver() && param_.random_seed() >= 0) {
    Caffe::set_random_seed(param_.random_seed());
  }
  // Scaffolding code
  InitTrainNet();
  InitTestNets();
  LOG(INFO) << "Solver scaffolding done.";
  iter_ = 0;
  total_lapse_ = 0.F;
  current_step_ = 0;
}

void Solver::InitTrainNet() {
  const int num_train_nets = param_.has_net() + param_.has_net_param() +
      param_.has_train_net() + param_.has_train_net_param();
  const string& field_names = "net, net_param, train_net, train_net_param";
  CHECK_GE(num_train_nets, 1) << "SolverParameter must specify a train net "
      << "using one of these fields: " << field_names;
  CHECK_LE(num_train_nets, 1) << "SolverParameter must not contain more than "
      << "one of these fields specifying a train_net: " << field_names;
  NetParameter net_param;
  if (param_.has_train_net_param()) {
    LOG_IF(INFO, Caffe::root_solver())
        << "Creating training net specified in train_net_param.";
    net_param.CopyFrom(param_.train_net_param());
  } else if (param_.has_train_net()) {
    LOG_IF(INFO, Caffe::root_solver())
        << "Creating training net from train_net file: " << param_.train_net();
    ReadNetParamsFromTextFileOrDie(param_.train_net(), &net_param);
  }
  if (param_.has_net_param()) {
    LOG_IF(INFO, Caffe::root_solver())
        << "Creating training net specified in net_param.";
    net_param.CopyFrom(param_.net_param());
  }
  if (param_.has_net()) {
    LOG_IF(INFO, Caffe::root_solver())
        << "Creating training net from net file: " << param_.net();
    ReadNetParamsFromTextFileOrDie(param_.net(), &net_param);
  }
  // Set the correct NetState.  We start with the solver defaults (lowest
  // precedence); then, merge in any NetState specified by the net_param itself;
  // finally, merge in any NetState specified by the train_state (highest
  // precedence).
  NetState net_state;
  net_state.set_phase(TRAIN);
  net_state.MergeFrom(net_param.state());
  net_state.MergeFrom(param_.train_state());
  net_param.mutable_state()->CopyFrom(net_state);
  if (Caffe::root_solver()) {
    net_.reset(new Net(net_param, rank_, &init_flag_, &iter0_flag_));
  } else {
    net_.reset(new Net(net_param, rank_, &init_flag_, &iter0_flag_,
        root_solver_->net_.get()));
  }
}

void Solver::InitTestNets() {
  const bool has_net_param = param_.has_net_param();
  const bool has_net_file = param_.has_net();
  const int num_generic_nets = has_net_param + has_net_file;
  CHECK_LE(num_generic_nets, 1)
      << "Both net_param and net_file may not be specified.";
  const int num_test_net_params = param_.test_net_param_size();
  const int num_test_net_files = param_.test_net_size();
  const int num_test_nets = num_test_net_params + num_test_net_files;
  if (num_generic_nets) {
      CHECK_GE(param_.test_iter_size(), num_test_nets)
          << "test_iter must be specified for each test network.";
  } else {
      CHECK_EQ(param_.test_iter_size(), num_test_nets)
          << "test_iter must be specified for each test network.";
  }
  // If we have a generic net (specified by net or net_param, rather than
  // test_net or test_net_param), we may have an unlimited number of actual
  // test networks -- the actual number is given by the number of remaining
  // test_iters after any test nets specified by test_net_param and/or test_net
  // are evaluated.
  const int num_generic_net_instances = param_.test_iter_size() - num_test_nets;
  const int num_test_net_instances = num_test_nets + num_generic_net_instances;
  if (param_.test_state_size()) {
    CHECK_EQ(param_.test_state_size(), num_test_net_instances)
        << "test_state must be unspecified or specified once per test net.";
  }
  if (num_test_net_instances) {
    CHECK_GT(param_.test_interval(), 0);
  }
  int test_net_id = 0;
  vector<string> sources(num_test_net_instances);
  vector<NetParameter> net_params(num_test_net_instances);
  for (int i = 0; i < num_test_net_params; ++i, ++test_net_id) {
      sources[test_net_id] = "test_net_param";
      net_params[test_net_id].CopyFrom(param_.test_net_param(i));
  }
  for (int i = 0; i < num_test_net_files; ++i, ++test_net_id) {
      sources[test_net_id] = "test_net file: " + param_.test_net(i);
      ReadNetParamsFromTextFileOrDie(param_.test_net(i),
          &net_params[test_net_id]);
  }
  const int remaining_test_nets = param_.test_iter_size() - test_net_id;
  if (has_net_param) {
    for (int i = 0; i < remaining_test_nets; ++i, ++test_net_id) {
      sources[test_net_id] = "net_param";
      net_params[test_net_id].CopyFrom(param_.net_param());
    }
  }
  if (has_net_file) {
    for (int i = 0; i < remaining_test_nets; ++i, ++test_net_id) {
      sources[test_net_id] = "net file: " + param_.net();
      ReadNetParamsFromTextFileOrDie(param_.net(), &net_params[test_net_id]);
    }
  }
  test_nets_.resize(num_test_net_instances);
  for (int i = 0; i < num_test_net_instances; ++i) {
    // Set the correct NetState.  We start with the solver defaults (lowest
    // precedence); then, merge in any NetState specified by the net_param
    // itself; finally, merge in any NetState specified by the test_state
    // (highest precedence).
    NetState net_state;
    net_state.set_phase(TEST);
    net_state.MergeFrom(net_params[i].state());
    if (param_.test_state_size()) {
      net_state.MergeFrom(param_.test_state(i));
    }
    net_params[i].mutable_state()->CopyFrom(net_state);
    LOG(INFO)
        << "Creating test net (#" << i << ") specified by " << sources[i];
    if (Caffe::root_solver()) {
      test_nets_[i].reset(new Net(net_params[i], rank_, &init_flag_, &iter0_flag_));
    } else {
      test_nets_[i].reset(new Net(net_params[i], rank_, &init_flag_, &iter0_flag_,
          root_solver_->test_nets_[i].get()));
    }
    test_nets_[i]->set_debug_info(param_.debug_info());
  }
}

void Solver::Step(int iters) {
  const int start_iter = iter_;
  const int stop_iter = iter_ + iters;
  int average_loss = this->param_.average_loss();
  losses_.clear();
  smoothed_loss_ = 0;
  const Caffe::Brew mode = Caffe::mode();
  const int solver_count = Caffe::solver_count();
  const bool root_solver = this->is_root();

  net_->set_solver(this);

  for (const shared_ptr<Blob>& param : net_->learnable_params()) {
    // To prevent allocations inside on_start call:
#ifndef CPU_ONLY
    param->current_mutable_data_memory(true);
#else
    param->current_mutable_data_memory(false);
#endif
  }

#ifndef CPU_ONLY
  net_->InitializeLearnableDiffSpace();

  if (solver_count > 1) {
    // we need to sync all threads before starting, otherwise some cuda init,
    // malloc or other cuda stuff could interlock with in-loop cuda GPU sync
    // called in on_start.
    callback_soft_barrier();
    {
      unique_ptr<unique_lock<shared_mutex>> lock;
      if (root_solver) {
        lock.reset(new unique_lock<shared_mutex>(GPUMemory::read_write_mutex()));
      }
      callback_soft_barrier();
      callback_->on_start(net_->learnable_params());
      callback_->syncCommStream();
    }
    callback_soft_barrier();
    LOG(INFO) << "Starting Optimization on GPU " << Caffe::current_device();
  }
  const bool use_multi_gpu_testing = Caffe::solver_count() > 1;
  const string mgpu_str = use_multi_gpu_testing ? "[MultiGPU] " : "";
#else
  const bool use_multi_gpu_testing = false;
  const string mgpu_str;
#endif

  reduce_thread_.reset(new boost::thread(&Solver::Reduce, this,
      Caffe::current_device(), mode, caffe_rng_rand(), solver_count, root_solver));

  while (iter_ < stop_iter) {
    if (param_.snapshot_diff()) {
      net_->ClearParamDiffs();
    }  // we clean them in ApplyUpdate otherwise

    // Just started or restored?
    const bool first_loop = iter_ == 0 || iterations_last_ < 0;
    if (first_loop) {
      if (TestAll(1, use_multi_gpu_testing)) {
        break;
      }
      callback_soft_barrier();
      LOG_IF(INFO, Caffe::root_solver()) << mgpu_str << "Initial Test completed";
    } else if (param_.test_interval() && iter_ % param_.test_interval() == 0) {
      test_timer_.Start();
      if (TestAll(0, use_multi_gpu_testing)) {
        break;
      }
      callback_soft_barrier();
      float lapse = test_timer_.Seconds();
      LOG_IF(INFO, Caffe::root_solver()) << mgpu_str << "Tests completed in "
                                         << lapse << "s";
    }
    if (requested_early_exit_) {
      // Break out of the while loop because stop was requested while testing.
      break;
    }

    const bool display = this->display();
    net_->set_debug_info(display && param_.debug_info());
    // accumulate the loss and gradient
    float loss = 0.F;
    if (first_loop) {
      iterations_last_ = iter_;
      iteration_timer_.Start();
      init_flag_.set();
    }

    iteration_start_signal();
    for (int i = 0; i < param_.iter_size(); ++i) {
      loss += net_->ForwardBackward(i + 1 == param_.iter_size());
      if (first_loop && i == 0) {
        iter0_flag_.set();
        net_->wait_layers_init();
      }
    }
    iter_size_complete_ = true;
    loss /= param_.iter_size();
    iteration_wait();
    if (requested_early_exit_) {
      total_lapse_ += iteration_timer_.Seconds();
      break;
    }

    // average the loss across iterations for smoothed reporting
    UpdateSmoothedLoss(loss, start_iter, average_loss);
    if (display || iter_ <= 2 || iter_ + 1 >= stop_iter) {
      float lapse = iteration_timer_.Seconds();
      if (iter_ >= 2) {  // we skip 0th and 1st for correct benchmarking
        total_lapse_ += lapse;
        float per_s = (iter_ - iterations_last_) / (lapse > 0.F ? lapse : 1.F);
        LOG_IF(INFO, Caffe::root_solver()) << "Iteration " << iter_
                                           << " (" << per_s << " iter/s, " << lapse << "s/"
                                           << param_.display() << " iter), loss = "
                                           << smoothed_loss_;
      } else {
        LOG_IF(INFO, Caffe::root_solver()) << "Iteration " << iter_
                                           << " (" << lapse << " s), loss = " << smoothed_loss_;
      }
      const vector<Blob*>& result = net_->output_blobs();
      int score_index = 0;
      for (int j = 0; j < result.size(); ++j) {
        const float* result_vec = result[j]->cpu_data<float>();
        const string& output_name =
            net_->blob_names()[net_->output_blob_indices()[j]];
        const float loss_weight =
            net_->blob_loss_weights()[net_->output_blob_indices()[j]];
        for (int k = 0; k < result[j]->count(); ++k) {
          ostringstream loss_msg_stream;
          if (loss_weight) {
            loss_msg_stream << " (* " << loss_weight
                << " = " << (loss_weight * result_vec[k]) << " loss)";
          }
          LOG_IF(INFO, Caffe::root_solver()) << "    Train net output #"
              << score_index++ << ": " << output_name << " = "
              << result_vec[k] << loss_msg_stream.str();
        }
      }
      PrintRate();
      iterations_last_ = iter_;
      iteration_timer_.Start();
    }
    // Increment the internal iter_ counter -- its value should always indicate
    // the number of times the weights have been updated.
    ++iter_;

    SolverAction::Enum request = GetRequestedAction();
    // Save a snapshot if needed.
    if ((param_.snapshot()
         && iter_ % param_.snapshot() == 0
         && Caffe::root_solver()) ||
         (request == SolverAction::SNAPSHOT)) {
      Snapshot();
    }
    if (SolverAction::STOP == request) {
      requested_early_exit_ = true;
      total_lapse_ += iteration_timer_.Seconds();
      // Break out of training loop.
      break;
    }
  }
  Finalize();
}

void Solver::Finalize() {
  if (reduce_thread_) {
    net_->Finalize();
    reduce_thread_->join();
  }
}

void Solver::Reduce(int device, Caffe::Brew mode, int rand_seed,
    int solver_count, bool root_solver) {
  Caffe::set_mode(mode);
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaSetDevice(device));
#ifndef NO_NVML
    nvml::setCpuAffinity(rank_);
#endif
  }
#endif
  Caffe::set_random_seed(rand_seed);
  Caffe::set_solver_count(solver_count);
  Caffe::set_root_solver(root_solver);
  net_->ReduceAndUpdate();
}

bool Solver::Solve(const char* resume_file) {
  callback_soft_barrier();
  LOG(INFO) << "Solving " << net_->name();
  LOG(INFO) << "Learning Rate Policy: " << param_.lr_policy();
  // Initialize to false every time we start solving.
  requested_early_exit_ = false;

  if (resume_file) {
    LOG(INFO) << "Restoring previous solver status from " << resume_file;
    Restore(resume_file);
  }

  callback_soft_barrier();
  if (Caffe::restored_iter() != -1) {
    //set correct state for Rank > 0
    iter_ = Caffe::restored_iter();
    iterations_last_ = -1;
  }

  // For a network that is trained by the solver, no bottom or top vecs
  // should be given, and we will just provide dummy vecs.
  int start_iter = iter_;
  Step(param_.max_iter() - iter_);
  // If we haven't already, save a snapshot after optimization, unless
  // overridden by setting snapshot_after_train := false
  if (param_.snapshot_after_train()
      && (!param_.snapshot() || iter_ % param_.snapshot() != 0)) {
    if (Caffe::root_solver()) {
      Snapshot();
    }
  }
  if (requested_early_exit_) {
    LOG(INFO) << "Optimization stopped early.";
    return true;
  }
  // After the optimization is done, run an additional train and test pass to
  // display the train and test loss/outputs if appropriate (based on the
  // display and test_interval settings, respectively).  Unlike in the rest of
  // training, for the train net we only run a forward pass as we've already
  // updated the parameters "max_iter" times -- this final pass is only done to
  // display the loss, which is computed in the forward pass.
  if (this->display()) {
    int average_loss = this->param_.average_loss();
    float loss;
    net_->Forward(&loss);

    UpdateSmoothedLoss(loss, start_iter, average_loss);

    LOG_IF(INFO, Caffe::root_solver()) << "Iteration "
        << iter_ << ", loss = " << smoothed_loss_;
  }
  if (param_.test_interval() && iter_ % param_.test_interval() == 0) {
    bool use_multi_gpu_testing = Caffe::solver_count() > 1;
    TestAll(0, use_multi_gpu_testing);
    callback_soft_barrier();
  }
  return false;
}

bool Solver::TestAll(const int iters, bool use_multi_gpu) {
  for (int test_net_id = 0;
       test_net_id < test_nets_.size() && !requested_early_exit_;
       ++test_net_id) {
    if (Test(test_net_id, iters, use_multi_gpu)) {
      return true;
    }
  }
  return false;
}

bool Solver::Test(const int test_net_id, const int iters, bool use_multi_gpu) {
  LOG_IF(INFO, Caffe::root_solver()) << "Iteration " << iter_
            << ", Testing net (#" << test_net_id << ")";
  if (!test_nets_[test_net_id]->trained_layers_shared()) {
    CHECK_NOTNULL(test_nets_[test_net_id].get())->ShareTrainedLayersWith(net_.get());
  }
  vector<float> test_score;
  vector<int> test_score_output_id;
  const shared_ptr<Net>& test_net = test_nets_[test_net_id];
  test_net->set_solver(this);
  float loss = 0.F;
  const int test_iterations = iters > 0 ? iters : param_.test_iter(test_net_id);
  for (int i = 0; i < test_iterations; ++i) {
    SolverAction::Enum request = GetRequestedAction();
    // Check to see if stoppage of testing/training has been requested.
    while (request != SolverAction::NONE) {
        if (SolverAction::SNAPSHOT == request) {
          Snapshot();
        } else if (SolverAction::STOP == request) {
          requested_early_exit_ = true;
        }
        request = GetRequestedAction();
    }
    if (requested_early_exit_) {
      LOG(INFO) << "Test interrupted.";
      Finalize();
      return true;
    }

    float iter_loss;
    const vector<Blob*>& result = test_net->Forward(&iter_loss);
    if (param_.test_compute_loss()) {
      loss += iter_loss;
    }
    if (i == 0) {
      for (int j = 0; j < result.size(); ++j) {
        for (int k = 0; k < result[j]->count(); ++k) {
          test_score.push_back(result[j]->data_at(k));
          test_score_output_id.push_back(j);
        }
      }
    } else {
      int idx = 0;
      for (int j = 0; j < result.size(); ++j) {
        for (int k = 0; k < result[j]->count(); ++k) {
          test_score[idx++] += result[j]->data_at(k);
        }
      }
    }
  }
  callback_soft_barrier();

  if (use_multi_gpu) {
    // now we've done, transfer results
    for (int i = 0; i < root_callbacks_.size(); ++i) {
      root_callbacks_[i]->saveTestResults(loss, test_score);
    }
    callback_soft_barrier();
    float global_loss = 0.F;
    vector<float> global_scores(test_score.size());
    // aggregate test results from all solvers
    for (int i = 0; i < root_callbacks_.size(); ++i) {
      root_callbacks_[i]->aggregateTestResults(&global_loss, &global_scores);
    }
    callback_soft_barrier();
    loss = global_loss;
    for (int i = 0; i < test_score.size(); ++i) {
      test_score[i] = global_scores[i];
    }
  }

  if (param_.test_compute_loss()) {
    loss /= param_.test_iter(test_net_id);
    LOG(INFO) << "Test loss: " << loss;
  }
  for (int i = 0; i < test_score.size(); ++i) {
    const int output_blob_index =
        test_net->output_blob_indices()[test_score_output_id[i]];
    const string& output_name = test_net->blob_names()[output_blob_index];
    const float loss_weight = test_net->blob_loss_weights()[output_blob_index];
    ostringstream loss_msg_stream;
    const float mean_score = test_score[i] / test_iterations / Caffe::solver_count();
    if (loss_weight) {
      loss_msg_stream << " (* " << loss_weight
          << " = " << (loss_weight * mean_score) << " loss)";
    }
    LOG_IF(INFO, Caffe::root_solver()) << "    Test net output #" << i <<
        ": " << output_name << " = " << mean_score << loss_msg_stream.str();
  }
  return false;
}

void Solver::Snapshot() {
  CHECK(Caffe::root_solver());
  string model_filename;
  switch (param_.snapshot_format()) {
  case caffe::SolverParameter_SnapshotFormat_BINARYPROTO:
    model_filename = SnapshotToBinaryProto();
    break;
  case caffe::SolverParameter_SnapshotFormat_HDF5:
    model_filename = SnapshotToHDF5();
    break;
  default:
    LOG(FATAL) << "Unsupported snapshot format.";
  }
  SnapshotSolverState(model_filename);
}

void Solver::CheckSnapshotWritePermissions() {
  if (Caffe::root_solver() && param_.snapshot()) {
    CHECK(param_.has_snapshot_prefix())
        << "In solver params, snapshot is specified but snapshot_prefix is not";
    string probe_filename = SnapshotFilename(".tempfile");
    std::ofstream probe_ofs(probe_filename.c_str());
    if (probe_ofs.good()) {
      probe_ofs.close();
      std::remove(probe_filename.c_str());
    } else {
      LOG(FATAL) << "Cannot write to snapshot prefix '"
          << param_.snapshot_prefix() << "'.  Make sure "
          << "that the directory exists and is writeable.";
    }
  }
}

string Solver::SnapshotFilename(const string extension) {
  return param_.snapshot_prefix() + "_iter_" + caffe::format_int(iter_) + extension;
}

string Solver::SnapshotToBinaryProto() {
  string model_filename = SnapshotFilename(".caffemodel");
  LOG(INFO) << "Snapshotting to binary proto file " << model_filename;
  NetParameter net_param;
  net_->ToProto(&net_param, param_.snapshot_diff());
  WriteProtoToBinaryFile(net_param, model_filename);
  return model_filename;
}

string Solver::SnapshotToHDF5() {
  string model_filename = SnapshotFilename(".caffemodel.h5");
  LOG(INFO) << "Snapshotting to HDF5 file " << model_filename;
  net_->ToHDF5(model_filename, param_.snapshot_diff());
  return model_filename;
}

void Solver::Restore(const char* state_file) {
  CHECK(Caffe::root_solver());
  string state_filename(state_file);
  if (state_filename.size() >= 3 &&
      state_filename.compare(state_filename.size() - 3, 3, ".h5") == 0) {
    RestoreSolverStateFromHDF5(state_filename);
  } else {
    RestoreSolverStateFromBinaryProto(state_filename);
  }
}

void Solver::UpdateSmoothedLoss(float loss, int start_iter,
    int average_loss) {
  if (losses_.size() < average_loss) {
    losses_.push_back(loss);
    int size = losses_.size();
    smoothed_loss_ = (smoothed_loss_ * (size - 1) + loss) / size;
  } else {
    int idx = (iter_ - start_iter) % average_loss;
    smoothed_loss_ += (loss - losses_[idx]) / average_loss;
    losses_[idx] = loss;
  }
}

float Solver::perf_report(std::ostream& os, int device, int align) const {
  std::string al(align, ' ');
  float perf_ratio = total_lapse() > 0. ?
      (iter() > 2 ? iter() - 2 - iterations_restored_ : 0) / total_lapse() : 0.F;
  float perf = perf_ratio * net_->batch_per_solver();
  os << al << "Solver performance on device " << device << ": "
      << perf_ratio << " * " << net_->batch_per_solver()
      << " = " << perf << " img/sec";
  return perf;
}

}  // namespace caffe
