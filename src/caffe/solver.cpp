#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "caffe/layers/memory_data_layer.hpp"
#include "caffe/solver.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/hdf5.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/upgrade_proto.hpp"

#include "leveldb/db.h"
#include "leveldb/write_batch.h"

#include "lmdb.h"

namespace caffe {

template<typename Dtype>
void Solver<Dtype>::SetActionFunction(ActionCallback func) {
  action_request_function_ = func;
}

template<typename Dtype>
SolverAction::Enum Solver<Dtype>::GetRequestedAction() {
  if (action_request_function_) {
    // If the external request function has been set, call it.
    return action_request_function_();
  }
  return SolverAction::NONE;
}

template <typename Dtype>
Solver<Dtype>::Solver(const SolverParameter& param, const Solver* root_solver)
    : net_(), callbacks_(), root_solver_(root_solver),
      requested_early_exit_(false), iteration_timer_(), iterations_last_() {
  Init(param);
}

template <typename Dtype>
Solver<Dtype>::Solver(const string& param_file, const Solver* root_solver)
    : net_(), callbacks_(), root_solver_(root_solver),
      requested_early_exit_(false), iteration_timer_(), iterations_last_() {
  SolverParameter param;
  ReadSolverParamsFromTextFileOrDie(param_file, &param);
  Init(param);
}

template <typename Dtype>
void Solver<Dtype>::Init(const SolverParameter& param) {
  CHECK(Caffe::root_solver() || root_solver_)
      << "root_solver_ needs to be set for all non-root solvers";
  LOG_IF(INFO, Caffe::root_solver()) << "Initializing solver from parameters: "
    << std::endl << param.DebugString();
  param_ = param;
  CHECK_GE(param_.average_loss(), 1) << "average_loss should be non-negative.";
  CheckSnapshotWritePermissions();
  if (Caffe::root_solver() && param_.random_seed() >= 0) {
    Caffe::set_random_seed(param_.random_seed());
  }
  // Scaffolding code
  InitTrainNet();
  if (Caffe::root_solver()) {
    InitTestNets();
    LOG(INFO) << "Solver scaffolding done.";
  }
  iter_ = 0;
  current_step_ = 0;
}


template <typename Dtype>
unsigned Solver<Dtype>::get_train_net_batch_size() {
  const vector<shared_ptr<Layer<Dtype> > >& layers = this->net()->layers();
  for (int ind = 0; ind < layers.size(); ++ind) {
    if (strcmp(layers[ind]->type(), "Data") == 0) {
      const LayerParameter& layer_param = layers[ind]->layer_param();
      const DataParameter& data_param = layer_param.data_param();
      return data_param.batch_size();
    }
  }
  LOG(INFO) << "Failed to find the batch_size of train network.\n";
  return 0;
}


template <typename Dtype>
unsigned Solver<Dtype>::get_test_net_batch_size(int test_net_id) {
  const vector<shared_ptr<Net<Dtype> > >& test_nets_ = this->test_nets();
  assert(test_net_id < this->test_nets_.size());
  const vector<shared_ptr<Layer<Dtype> > >& layers
    = test_nets_[test_net_id]->layers();
  for (int ind = 0; ind < layers.size(); ++ind) {
    if (strcmp(layers[ind]->type(), "Data") == 0) {
      const LayerParameter& layer_param = layers[ind]->layer_param();
      const DataParameter& data_param = layer_param.data_param();
      return data_param.batch_size();
    }
  }
  LOG(INFO) << "Failed to find the batch_size of test network #"
            << test_net_id << ".\n";
  return 0;
}

template <typename Dtype>
unsigned Solver<Dtype>::get_num_entries_db_core(
  const DataParameter& data_param) {
#ifdef USE_LEVELDB
  if (data_param.backend() == DataParameter_DB_LEVELDB) {
      string level_db_name = data_param.source();
      LOG(INFO)
        << "Reading leveldb <" << level_db_name
        << "> to find the number of entries...\n";
      leveldb::DB* db_ = NULL;
      leveldb::Options options;
      options.block_size = 65536;
      options.write_buffer_size = 268435456;
      options.max_open_files = 100;
      options.error_if_exists = false;
      options.create_if_missing = false;
      // TODO: The following Open() fails because the level db is already
      // opened in data_reader.cpp and level db does not allow multiple
      // Open() of level db!  This means that the user has to manually
      // specify epoch_size parameter when using level dbs. (level db
      // needs to be fixed to allow multiple read-only Open()'s in the same
      // process.)
      leveldb::Status status = leveldb::DB::Open(options, level_db_name, &db_);
      CHECK(status.ok()) << "Failed to open leveldb " << level_db_name
                         << std::endl << status.ToString();
      // level db does not provide an API function to determine the number of
      // records.  Hence, we have to iterate through the db to count the
      // the number of records!
      leveldb::Iterator* iter_ = db_->NewIterator(leveldb::ReadOptions());
      iter_->SeekToFirst();
      unsigned num_level_db_entries = 0;
      while (iter_->Valid()) {
        ++num_level_db_entries;
        iter_->Next();
      }
      if (iter_ != NULL) {
          delete iter_;
      }
      if (db_ != NULL) {
          delete db_;
      }
      LOG(INFO) << "leveldb <" << level_db_name
                << "> has " << num_level_db_entries << " entries.\n";
      return(num_level_db_entries);
  }
#endif  // USE_LEVELDB
#ifdef USE_LMDB
  if (data_param.backend() == DataParameter_DB_LMDB) {
      LOG(INFO)
        << "Reading lmdb <" << data_param.source()
        << "> to find the number of entries...\n";
      // open the source lmdb, get the number of entries and close it.
      MDB_env* mdb_env_;
      CHECK_EQ(mdb_env_create(&mdb_env_), MDB_SUCCESS);
      int flags = MDB_RDONLY | MDB_NOTLS;
      CHECK_EQ(
        mdb_env_open(mdb_env_, data_param.source().c_str(), flags, 0664),
        MDB_SUCCESS);
      MDB_stat stat;
      CHECK_EQ(mdb_env_stat(mdb_env_, &stat), MDB_SUCCESS);
      unsigned num_lmdb_entries = stat.ms_entries;
      mdb_env_close(mdb_env_);
      LOG(INFO) << "lmdb <" << data_param.source()
                << "> has " << num_lmdb_entries << " entries.\n";
      return(num_lmdb_entries);
  }
#endif  // USE_LMDB
  LOG(FATAL) << "Unknown database backend";
  return 0;
}


template <typename Dtype>
unsigned Solver<Dtype>::get_num_entries_train_net_db() {
  const vector<shared_ptr<Layer<Dtype> > >& layers = this->net()->layers();
  for (int ind = 0; ind < layers.size(); ++ind) {
    if (strcmp(layers[ind]->type(), "Data") == 0) {
      const LayerParameter& layer_param = layers[ind]->layer_param();
      const DataParameter& data_param = layer_param.data_param();
      return(this->get_num_entries_db_core(data_param));
    }
  }
  LOG(INFO) << "Failed to find the number of records in the train network.\n";
  return 0;
}


template <typename Dtype>
unsigned Solver<Dtype>::get_num_entries_test_net_db(int test_net_id) {
  const vector<shared_ptr<Net<Dtype> > > test_nets_ = this->test_nets();
  assert(test_net_id < this->test_nets_.size());
  const vector<shared_ptr<Layer<Dtype> > >& layers
    = test_nets_[test_net_id]->layers();
  for (int ind = 0; ind < layers.size(); ++ind) {
    if (strcmp(layers[ind]->type(), "Data") == 0) {
      const LayerParameter& layer_param = layers[ind]->layer_param();
      const DataParameter& data_param = layer_param.data_param();
      return(this->get_num_entries_db_core(data_param));
    }
  }
  LOG(INFO)
    << "Failed to find the batch_size of test network #"
    << test_net_id << ".\n";
  return 0;
}


// If max_iter parameter is not specified but max_iter_epoch parameter is
// specified, then compute the max_iter parameter from the max_epoch parameter.
// max_iter parameter specifies the number of batches to train.  Alternatively,
// one can specify the number of training iterations in terms of epochs to
// train (max_epoc).  max_iter = (max_epoch*epoch_size)/batch_size where
// max_epoch is the number of epochs to train, epoch_size is the number of
// records in the train database and batch size is the batch size of the train
// network.  if epoch_size is not specified, then get the number of records by
// looking up the database.
template <typename Dtype>
void Solver<Dtype>::compute_max_iter() {
  if (!param_.has_max_iter() && param_.has_max_epoch()) {
    LOG(INFO)
      << "max_iter parameter is not specified."
      << " Computing max_iter parameter...\n";
    CHECK(param_.has_max_epoch());
    if (!param_.has_epoch_size()) {
      LOG(INFO)
        << "epoch_size parameter is not specified. "
        << "Reading it from db...\n";
      unsigned rec_count = this->get_num_entries_train_net_db();
      LOG(INFO)
        << "Setting epoch_size to " << rec_count << "\n";
      param_.set_epoch_size(rec_count);
    }
    CHECK(param_.has_epoch_size());
    int train_net_batch_size = this->get_train_net_batch_size();
    CHECK_GT(train_net_batch_size, 0);
    unsigned my_max_iter =
        (param_.max_epoch() * param_.epoch_size()) / train_net_batch_size;
    LOG(INFO)
      << "max_epoch = " << param_.max_epoch()
      << ", epoch_size = " << param_.epoch_size()
      << ", batch_size = " << train_net_batch_size
      << ". Setting max_iter to " << my_max_iter  << "\n";
    param_.set_max_iter(my_max_iter);
  }
}


// If test_iter parameter is not specified, then compute it from the
// test_epoch_size parameter and the test net's batch size parameter.
// If test_epoch_size parameter of the test net is not specified,
// read the db to get it.
template <typename Dtype>
void Solver<Dtype>::compute_test_iter(int num_test_net_instances) {
  for (int i = 0; i < num_test_net_instances; ++i) {
    if (param_.test_iter(i) == 0) {
      LOG(INFO)
        << "test_iter of test net #" << i << " is 0. "
        << "Computing it using other parameters...\n";
      int test_net_batch_size = get_test_net_batch_size(i);
      LOG(INFO)
        << "batch_size of test net #"
        << i << " is " << test_net_batch_size << "\n";
      CHECK_GT(test_net_batch_size, 0);
      if (param_.test_epoch_size(i) == 0) {
        LOG(INFO)
          << "test_epoch_size of test net #" << i << " is 0. "
          << "Reading it from db...\n";
        unsigned rec_count = this->get_num_entries_test_net_db(i);
        LOG(INFO)
          << "Setting test_epoch_size of test net #"
          << i << " to " << rec_count << "\n";
        param_.set_test_epoch_size(i, rec_count);
      }
      int test_iter_val = param_.test_epoch_size(i) / test_net_batch_size;
      LOG(INFO)
        << "Setting test_iter of test net #"
        << i << " to " << test_iter_val << "\n";
      param_.set_test_iter(i, test_iter_val);
    }
  }
}

// if test_interval is not specified, but test_interva_epoch is specified, then
// compute test_interval from test_interval_epoch and train net batch_size
template <typename Dtype>
void Solver<Dtype>::compute_test_interval() {
  if (!param_.has_test_interval() && param_.has_test_interval_epoch()) {
    LOG(INFO)
      << "test_interval parameter is not specified. "
      << "Computing it from other parameters...\n";
    CHECK(param_.has_test_interval_epoch());
    int train_net_batch_size = this->get_train_net_batch_size();
    CHECK_GT(train_net_batch_size, 0);

    // at this point we should have epoch_size either specified or computed.
    CHECK(param_.has_epoch_size());
    unsigned my_test_interval
       = (param_.test_interval_epoch() * param_.epoch_size());
    my_test_interval /= train_net_batch_size;
    LOG(INFO)
      << "test_interval_epoch = " << param_.test_interval_epoch()
      << ", epoch_size = " << param_.epoch_size()
      << ", batch_size = " << train_net_batch_size
      << ". Setting test_interval to " << my_test_interval << "\n";
    param_.set_test_interval(my_test_interval);
    CHECK(param_.has_test_interval());
  }
}

template <typename Dtype>
void Solver<Dtype>::InitTrainNet() {
  const int num_train_nets = param_.has_net() + param_.has_net_param() +
      param_.has_train_net() + param_.has_train_net_param();
  const string& field_names = "net, net_param, train_net, train_net_param";
  CHECK_GE(num_train_nets, 1)
    << "SolverParameter must specify a train net "
    << "using one of these fields: "
    << field_names;
  CHECK_LE(num_train_nets, 1)
    << "SolverParameter must not contain more than "
    << "one of these fields specifying a train_net: "
    << field_names;
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
    net_.reset(new Net<Dtype>(net_param));
  } else {
    net_.reset(new Net<Dtype>(net_param, root_solver_->net_.get()));
  }

  // If max_iter parameter is not specified, then compute it from
  // alternate parameters.  See function compute_max_iter for details.
  compute_max_iter();
}

template <typename Dtype>
void Solver<Dtype>::InitTestNets() {
  CHECK(Caffe::root_solver());
  const bool has_net_param = param_.has_net_param();
  const bool has_net_file = param_.has_net();
  const int num_generic_nets = has_net_param + has_net_file;
  CHECK_LE(num_generic_nets, 1)
      << "Both net_param and net_file may not be specified.";
  const int num_test_net_params = param_.test_net_param_size();
  const int num_test_net_files = param_.test_net_size();
  const int num_test_nets = num_test_net_params + num_test_net_files;

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
      test_nets_[i].reset(new Net<Dtype>(net_params[i]));
    } else {
      test_nets_[i].reset(new Net<Dtype>(net_params[i],
          root_solver_->test_nets_[i].get()));
    }
    test_nets_[i]->set_debug_info(param_.debug_info());
  }

  // If test_iter parameters are not specified, then compute them from
  // alternate parameters.  See compute_test_iter() function for details.
  compute_test_iter(num_test_net_instances);
  if (num_generic_nets) {
      CHECK_GE(param_.test_iter_size(), num_test_nets)
          << "test_iter must be specified for each test network.";
  } else {
      CHECK_EQ(param_.test_iter_size(), num_test_nets)
          << "test_iter must be specified for each test network.";
  }

  // if test_interval is not specified, then compute it from
  // alternate parameters.  See compute_test_interval() function for details.
  compute_test_interval();
  if (num_test_net_instances) {
    CHECK_GT(param_.test_interval(), 0);
  }
}

template <typename Dtype>
void Solver<Dtype>::Step(int iters) {
  const int start_iter = iter_;
  const int stop_iter = iter_ + iters;
  int average_loss = this->param_.average_loss();
  losses_.clear();
  smoothed_loss_ = 0;
  iteration_timer_.Start();

  for (int i = 0; i < callbacks_.size(); ++i) {
    // we need to sync all threads before starting, otherwise some cuda init,
    // malloc or other cuda stuff could interlock with in-loop cuda GPU sync
    // called in on_start.
    callbacks_[i]->soft_barrier();
    // Initial bcast of parameters
    callbacks_[i]->on_start();
  }

  net_->SetSolver(this);

  while (iter_ < stop_iter) {
    // zero-init the params
    net_->ClearParamDiffs();
    if (param_.test_interval() && iter_ % param_.test_interval() == 0
        && (iter_ > 0 || param_.test_initialization())) {
      if (Caffe::root_solver()) {
        TestAll();
      }
      if (requested_early_exit_) {
        // Break out of the while loop because stop was requested while testing.
        break;
      }
      for (int i = 0; i < callbacks_.size(); ++i) {
        callbacks_[i]->soft_barrier();
      }
    }
    const bool display = param_.display() && iter_ % param_.display() == 0;
    net_->set_debug_info(display && param_.debug_info());
    // accumulate the loss and gradient
    Dtype loss = 0;
    for (int i = 0; i < param_.iter_size(); ++i) {
      loss += net_->ForwardBackward();
    }
    loss /= param_.iter_size();
    // average the loss across iterations for smoothed reporting
    UpdateSmoothedLoss(loss, start_iter, average_loss);
    if (display) {
      float lapse = iteration_timer_.Seconds();
      float per_s = (iter_ - iterations_last_) / (lapse ? lapse : 1);
      LOG_IF(INFO, Caffe::root_solver()) << "Iteration " << iter_
          << " (" << per_s << " iter/s, " << lapse << "s/"
          << param_.display() <<" iter), loss = " << smoothed_loss_;
      iteration_timer_.Start();
      iterations_last_ = iter_;
      const vector<Blob<Dtype>*>& result = net_->output_blobs();
      int score_index = 0;
      for (int j = 0; j < result.size(); ++j) {
        const Dtype* result_vec = result[j]->cpu_data();
        const string& output_name =
            net_->blob_names()[net_->output_blob_indices()[j]];
        const Dtype loss_weight =
            net_->blob_loss_weights()[net_->output_blob_indices()[j]];
        for (int k = 0; k < result[j]->count(); ++k) {
          ostringstream loss_msg_stream;
          if (loss_weight) {
            loss_msg_stream << " (* " << loss_weight
                            << " = " << loss_weight * result_vec[k] << " loss)";
          }
          LOG_IF(INFO, Caffe::root_solver()) << "    Train net output #"
              << score_index++ << ": " << output_name << " = "
              << result_vec[k] << loss_msg_stream.str();
        }
      }
    }
#ifndef CPU_ONLY
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
#endif
    for (int i = 0; i < callbacks_.size(); ++i) {
      callbacks_[i]->allreduce();
    }
    // Make sure all gradient exchanges have finished in per-level scheme
    for (int i = 0; i < callbacks_.size(); ++i) {
      callbacks_[i]->syncCommStream();
    }

    ApplyUpdate();

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
      // Break out of training loop.
      break;
    }
  }
}


template <typename Dtype>
void Solver<Dtype>::Solve(const char* resume_file) {
  CHECK(Caffe::root_solver());
  LOG(INFO) << "Solving " << net_->name();
  LOG(INFO) << "Learning Rate Policy: " << param_.lr_policy();

  // Initialize to false every time we start solving.
  requested_early_exit_ = false;

  if (resume_file) {
    LOG(INFO) << "Restoring previous solver status from " << resume_file;
    Restore(resume_file);
  }

  // For a network that is trained by the solver, no bottom or top vecs
  // should be given, and we will just provide dummy vecs.
  int start_iter = iter_;
  Step(param_.max_iter() - iter_);

  // If we haven't already, save a snapshot after optimization, unless
  // overridden by setting snapshot_after_train := false
  if (param_.snapshot_after_train()
      && (!param_.snapshot() || iter_ % param_.snapshot() != 0)) {
    Snapshot();
  }
  if (requested_early_exit_) {
    LOG(INFO) << "Optimization stopped early.";
    return;
  }
  // After the optimization is done, run an additional train and test pass to
  // display the train and test loss/outputs if appropriate (based on the
  // display and test_interval settings, respectively).  Unlike in the rest of
  // training, for the train net we only run a forward pass as we've already
  // updated the parameters "max_iter" times -- this final pass is only done to
  // display the loss, which is computed in the forward pass.
  if (param_.display() && iter_ % param_.display() == 0) {
    int average_loss = this->param_.average_loss();
    Dtype loss;
    net_->Forward(&loss);

    UpdateSmoothedLoss(loss, start_iter, average_loss);

    LOG(INFO) << "Iteration " << iter_ << ", loss = " << smoothed_loss_;
  }
  if (param_.test_interval() && iter_ % param_.test_interval() == 0) {
    TestAll();
  }
  LOG(INFO) << "Optimization Done.";
}

template <typename Dtype>
void Solver<Dtype>::TestAll() {
  for (int test_net_id = 0;
       test_net_id < test_nets_.size() && !requested_early_exit_;
       ++test_net_id) {
    Test(test_net_id);
  }
}

template <typename Dtype>
void Solver<Dtype>::Test(const int test_net_id) {
  CHECK(Caffe::root_solver());
  LOG(INFO) << "Iteration " << iter_
            << ", Testing net (#" << test_net_id << ")";
  CHECK_NOTNULL(test_nets_[test_net_id].get())->
      ShareTrainedLayersWith(net_.get());
  vector<Dtype> test_score;
  vector<int> test_score_output_id;
  const shared_ptr<Net<Dtype> >& test_net = test_nets_[test_net_id];
  Dtype loss = 0;
  for (int i = 0; i < param_.test_iter(test_net_id); ++i) {
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
      // break out of test loop.
      break;
    }

    Dtype iter_loss;
    const vector<Blob<Dtype>*>& result =
        test_net->Forward(&iter_loss);
    if (param_.test_compute_loss()) {
      loss += iter_loss;
    }
    if (i == 0) {
      for (int j = 0; j < result.size(); ++j) {
        const Dtype* result_vec = result[j]->cpu_data();
        for (int k = 0; k < result[j]->count(); ++k) {
          test_score.push_back(result_vec[k]);
          test_score_output_id.push_back(j);
        }
      }
    } else {
      int idx = 0;
      for (int j = 0; j < result.size(); ++j) {
        const Dtype* result_vec = result[j]->cpu_data();
        for (int k = 0; k < result[j]->count(); ++k) {
          test_score[idx++] += result_vec[k];
        }
      }
    }
  }
  if (requested_early_exit_) {
    LOG(INFO)     << "Test interrupted.";
    return;
  }
  if (param_.test_compute_loss()) {
    loss /= param_.test_iter(test_net_id);
    LOG(INFO) << "Test loss: " << loss;
  }
  for (int i = 0; i < test_score.size(); ++i) {
    const int output_blob_index =
        test_net->output_blob_indices()[test_score_output_id[i]];
    const string& output_name = test_net->blob_names()[output_blob_index];
    const Dtype loss_weight = test_net->blob_loss_weights()[output_blob_index];
    ostringstream loss_msg_stream;
    const Dtype mean_score = test_score[i] / param_.test_iter(test_net_id);
    if (loss_weight) {
      loss_msg_stream << " (* " << loss_weight
                      << " = " << loss_weight * mean_score << " loss)";
    }
    LOG(INFO) << "    Test net output #" << i << ": " << output_name << " = "
              << mean_score << loss_msg_stream.str();
  }
}

template <typename Dtype>
void Solver<Dtype>::Snapshot() {
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

template <typename Dtype>
void Solver<Dtype>::CheckSnapshotWritePermissions() {
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

template <typename Dtype>
string Solver<Dtype>::SnapshotFilename(const string extension) {
  return param_.snapshot_prefix() + "_iter_" + caffe::format_int(iter_)
    + extension;
}

template <typename Dtype>
string Solver<Dtype>::SnapshotToBinaryProto() {
  string model_filename = SnapshotFilename(".caffemodel");
  LOG(INFO) << "Snapshotting to binary proto file " << model_filename;
  NetParameter net_param;
  net_->ToProto(&net_param, param_.snapshot_diff());
  WriteProtoToBinaryFile(net_param, model_filename);
  return model_filename;
}

template <typename Dtype>
string Solver<Dtype>::SnapshotToHDF5() {
  string model_filename = SnapshotFilename(".caffemodel.h5");
  LOG(INFO) << "Snapshotting to HDF5 file " << model_filename;
  net_->ToHDF5(model_filename, param_.snapshot_diff());
  return model_filename;
}

template <typename Dtype>
void Solver<Dtype>::Restore(const char* state_file) {
  CHECK(Caffe::root_solver());
  string state_filename(state_file);
  if (state_filename.size() >= 3 &&
      state_filename.compare(state_filename.size() - 3, 3, ".h5") == 0) {
    RestoreSolverStateFromHDF5(state_filename);
  } else {
    RestoreSolverStateFromBinaryProto(state_filename);
  }
}

template <typename Dtype>
void Solver<Dtype>::UpdateSmoothedLoss(Dtype loss, int start_iter,
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

INSTANTIATE_CLASS(Solver);

}  // namespace caffe
