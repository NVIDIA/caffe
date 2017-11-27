#ifndef CAFFE_SOLVER_HPP_
#define CAFFE_SOLVER_HPP_
#include <boost/function.hpp>

#include <atomic>
#include <string>
#include <vector>

#include "caffe/net.hpp"
#include "caffe/solver_factory.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

/**
  * @brief Enumeration of actions that a client of the Solver may request by
  * implementing the Solver's action request function, which a
  * a client may optionally provide in order to request early termination
  * or saving a snapshot without exiting. In the executable caffe, this
  * mechanism is used to allow the snapshot to be saved when stopping
  * execution with a SIGINT (Ctrl-C).
  */
  namespace SolverAction {
    enum Enum {
      NONE = 0,  // Take no special action.
      STOP = 1,  // Stop training. snapshot_after_train controls whether a
                 // snapshot is created.
      SNAPSHOT = 2  // Take a snapshot, and keep training.
    };
  }

/**
 * @brief Type of a function that returns a Solver Action enumeration.
 */
typedef boost::function<SolverAction::Enum()> ActionCallback;

/**
 * @brief An interface for classes that perform optimization on Net%s.
 *
 * Requires implementation of ApplyUpdate to compute a parameter update
 * given the current state of the Net parameters.
 */
class Solver {
 public:
  explicit Solver(const SolverParameter& param, size_t rank, const Solver* root_solver = NULL);
  explicit Solver(const string& param_file, size_t rank, const Solver* root_solver = NULL);
  void Init();
  void InitTrainNet();
  void InitTestNets();

  // Client of the Solver optionally may call this in order to set the function
  // that the solver uses to see what action it should take (e.g. snapshot or
  // exit training early).
  void SetActionFunction(ActionCallback func);
  SolverAction::Enum GetRequestedAction();
  // The main entry of the solver function. In default, iter will be zero. Pass
  // in a non-zero iter number to resume training for a pre-trained net.
  virtual bool Solve(const char* resume_file = NULL);
  void Solve(const string resume_file) { Solve(resume_file.c_str()); }
  void Step(int iters);
  // The Restore method simply dispatches to one of the
  // RestoreSolverStateFrom___ protected methods. You should implement these
  // methods to restore the state from the appropriate snapshot type.
  void Restore(const char* resume_file);
  // The Solver::Snapshot function implements the basic snapshotting utility
  // that stores the learned net. You should implement the SnapshotSolverState()
  // function that produces a SolverState protocol buffer that needs to be
  // written to disk together with the learned net.
  void Snapshot();
  virtual ~Solver();
  const SolverParameter& param() const { return param_; }
  shared_ptr<Net> net() { return net_; }
  const vector<shared_ptr<Net>>& test_nets() { return test_nets_; }
  int iter() const { return iter_; }
  int relative_iter() const { return iter_ - iterations_restored_; }
  float total_lapse() const { return total_lapse_; }
  bool is_root() const { return rank_ == 0; }
  float perf_report(std::ostream& os, int device, int align = 0) const;

  // Invoked at specific points during an iteration
  class Callback {
   public:
    virtual void allreduce(int param_id) = 0;
    virtual void allreduce_bucket(int count, void* bucket, Type type) = 0;
    virtual void soft_barrier() = 0;
    virtual void reduce_barrier() = 0;
    virtual void saveTestResults(float loss, const vector<float>& scores) = 0;
    virtual void aggregateTestResults(float* loss, vector<float>* scores) = 0;

#ifndef CPU_ONLY
    virtual cublasHandle_t cublas_handle() const = 0;
#endif

   protected:
    virtual void on_start(const vector<shared_ptr<Blob>>& net) = 0;
    friend class Solver;
  };

  Callback* callback() const {
    return callback_;
  }
  void set_callback(Callback* value) {
    callback_ = value;
  }
  void root_add_callback(Callback* value) {
    root_callbacks_.push_back(value);
  }

  Flag iter_flag_;

  void iteration_complete_signal() {
    iter_flag_.set();
  }
  void iteration_start_signal() {
    iter_flag_.reset();
  }
  void iteration_wait() {
    iter_flag_.wait();
  }

  void CheckSnapshotWritePermissions();
  void Finalize();

  void request_early_exit() {
    requested_early_exit_ = true;
    iteration_complete_signal();
  }

  bool display() const {
    return param_.display() && iter_ % param_.display() == 0;
  }

  bool param_display() const {
    return param_.display() > 0;
  }

  bool initialized() const {
    return init_flag_.is_set();
  }

  /**
   * @brief Returns the solver type.
   */
  virtual const char* type() const { return ""; }
  virtual void PrintRate(float rate = 0) {}
  virtual void ApplyUpdate(int param_id, void* handle, bool clear_grads) = 0;

 protected:
  string SnapshotFilename(const string extension);
  string SnapshotToBinaryProto();
  string SnapshotToHDF5();
  // The test routine
  bool TestAll(const int iters = 0, bool use_multi_gpu = false);
  bool Test(const int test_net_id = 0, const int iters = 0, bool use_multi_gpu = false);
  void TestDetection(const int test_net_id = 0);
  virtual void SnapshotSolverState(const string& model_filename) = 0;
  virtual void RestoreSolverStateFromHDF5(const string& state_file) = 0;
  virtual void RestoreSolverStateFromBinaryProto(const string& state_file) = 0;
  void UpdateSmoothedLoss(float loss, int start_iter, int average_loss);
  void Reduce(int device, Caffe::Brew mode, uint64_t rand_seed,
      int solver_count, bool root_solver);

  void callback_soft_barrier() {
    if (callback_ != nullptr) {
      callback_->soft_barrier();
    }
  }

  const SolverParameter param_;
  const Type data_type_;
  int iter_;
  int id_;
  float total_lapse_;
  int current_step_;
  shared_ptr<Net> net_;
  vector<shared_ptr<Net>> test_nets_;
  Callback* callback_;
  vector<Callback*> root_callbacks_;
  vector<float> losses_;
  float smoothed_loss_;
  unique_ptr<boost::thread> reduce_thread_;

  // The root solver that holds root nets (actually containing shared layers)
  // in data parallelism
  const Solver* const root_solver_;
  const size_t rank_;

  // A function that can be set by a client of the Solver to provide indication
  // that it wants a snapshot saved and/or to exit early.
  ActionCallback action_request_function_;

  // True iff a request to stop early was received.
  bool requested_early_exit_;

  // some layers like Data have to wait for this one
  Flag init_flag_, iter0_flag_;

  // Timing information
  Timer iteration_timer_;
  Timer test_timer_;
  int iterations_last_;
  int iterations_restored_;

  DISABLE_COPY_MOVE_AND_ASSIGN(Solver);
};

}  // namespace caffe

#endif  // CAFFE_SOLVER_HPP_
