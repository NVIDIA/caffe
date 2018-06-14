#ifndef CAFFE_SGD_SOLVERS_HPP_
#define CAFFE_SGD_SOLVERS_HPP_

#include "caffe/common.hpp"
#include "caffe/solver.hpp"

namespace caffe {

/**
 * @brief Optimizes the parameters of a Net using
 *        stochastic gradient descent (SGD) with momentum.
 */
template <typename Dtype>
class SGDSolver : public Solver {
 public:
  explicit SGDSolver(const SolverParameter& param,
      size_t rank = 0U, Solver *root_solver = NULL)
      : Solver(param, rank, root_solver) {
    PreSolve();
  }
  explicit SGDSolver(const string& param_file,
      size_t rank = 0U, Solver *root_solver = NULL)
      : Solver(param_file, rank) {
    PreSolve();
  }
  ~SGDSolver() {}

  const char* type() const override { return "SGD"; }
  const vector<shared_ptr<TBlob<Dtype> > >& history() { return history_; }
  float GetLearningRate() override;
  void ClipGradientsAndNormalize(void* handle, int type_id,
      const std::set<int>& param_ids) override;
  void PrintRate(float rate = 0) override;

 protected:
  void PreSolve();
  float GetMomentum();
  float GetWeightDecay() const;
  float GetLocalRate(int param_id, float& wgrad_sq) ;
  float local_decay(int param_id) const;

  float ApplyUpdate(int param_id, void* handle, float rate, bool normalize,
      bool clear_grads) override;
  void Normalize(int param_id, void* handle);
  void Regularize(int param_id);

  virtual float ComputeUpdateValue(int param_id, void* handle, float rate, bool clear_grads);
  virtual void SnapshotSolverState(const string& model_filename);
  virtual void SnapshotSolverStateToBinaryProto(const string& model_filename);
  virtual void SnapshotSolverStateToHDF5(const string& model_filename);
  virtual void RestoreSolverStateFromHDF5(const string& state_file);
  virtual void RestoreSolverStateFromBinaryProto(const string& state_file);
  void PrintParams(int param_id);

  // history maintains the historical momentum data.
  // update maintains update related data and is not needed in snapshots.
  // temp maintains other information that might be needed in computation
  //   of gradients/updates and is not needed in snapshots
  vector<shared_ptr<TBlob<Dtype> > > history_, update_, temp_;

  DISABLE_COPY_MOVE_AND_ASSIGN(SGDSolver);
};

template <typename Dtype>
class NesterovSolver : public SGDSolver<Dtype> {
 public:
  explicit NesterovSolver(const SolverParameter& param,
      size_t rank = 0U, Solver *root_solver = NULL)
      : SGDSolver<Dtype>(param, rank, root_solver) {}
  explicit NesterovSolver(const string& param_file,
      size_t rank = 0U, Solver *root_solver = NULL)
      : SGDSolver<Dtype>(param_file, rank, root_solver) {}
  virtual inline const char* type() const { return "Nesterov"; }

 protected:
  float ComputeUpdateValue(int param_id, void* handle, float rate, bool clear_grads) override;

  DISABLE_COPY_MOVE_AND_ASSIGN(NesterovSolver);
};

template <typename Dtype>
class AdaGradSolver : public SGDSolver<Dtype> {
 public:
  explicit AdaGradSolver(const SolverParameter& param,
      size_t rank = 0U, Solver *root_solver = NULL)
      : SGDSolver<Dtype>(param, rank, root_solver)
        { constructor_sanity_check(); }
  explicit AdaGradSolver(const string& param_file,
      size_t rank = 0U, Solver *root_solver = NULL)
      : SGDSolver<Dtype>(param_file, rank, root_solver)
        { constructor_sanity_check(); }
  virtual inline const char* type() const { return "AdaGrad"; }

 protected:
  float ComputeUpdateValue(int param_id, void* handle, float rate, bool clear_grads) override;
  void constructor_sanity_check() {
    CHECK_EQ(0, this->param_.momentum())
        << "Momentum cannot be used with AdaGrad.";
  }

  DISABLE_COPY_MOVE_AND_ASSIGN(AdaGradSolver);
};


template <typename Dtype>
class RMSPropSolver : public SGDSolver<Dtype> {
 public:
  explicit RMSPropSolver(const SolverParameter& param,
      size_t rank = 0U, Solver *root_solver = NULL)
      : SGDSolver<Dtype>(param, rank, root_solver)
        { constructor_sanity_check(); }
  explicit RMSPropSolver(const string& param_file,
      size_t rank = 0U, Solver *root_solver = NULL)
      : SGDSolver<Dtype>(param_file, rank, root_solver)
        { constructor_sanity_check(); }
  virtual inline const char* type() const { return "RMSProp"; }

 protected:
  float ComputeUpdateValue(int param_id, void* handle, float rate, bool clear_grads) override;
  void constructor_sanity_check() {
    CHECK_EQ(0, this->param_.momentum())
        << "Momentum cannot be used with RMSProp.";
    CHECK_GE(this->param_.rms_decay(), 0)
        << "rms_decay should lie between 0 and 1.";
    CHECK_LT(this->param_.rms_decay(), 1)
        << "rms_decay should lie between 0 and 1.";
  }

  DISABLE_COPY_MOVE_AND_ASSIGN(RMSPropSolver);
};

template <typename Dtype>
class AdaDeltaSolver : public SGDSolver<Dtype> {
 public:
  explicit AdaDeltaSolver(const SolverParameter& param,
      size_t rank = 0U, Solver *root_solver = NULL)
      : SGDSolver<Dtype>(param, rank, root_solver) { AdaDeltaPreSolve(); }
  explicit AdaDeltaSolver(const string& param_file,
      size_t rank = 0U, Solver *root_solver = NULL)
      : SGDSolver<Dtype>(param_file, rank, root_solver) { AdaDeltaPreSolve(); }
  virtual inline const char* type() const { return "AdaDelta"; }

 protected:
  void AdaDeltaPreSolve();
  float ComputeUpdateValue(int param_id, void* handle, float rate, bool clear_grads) override;

  DISABLE_COPY_MOVE_AND_ASSIGN(AdaDeltaSolver);
};

/**
 * @brief AdamSolver, an algorithm for first-order gradient-based optimization
 *        of stochastic objective functions, based on adaptive estimates of
 *        lower-order moments. Described in [1].
 *
 * [1] D. P. Kingma and J. L. Ba, "ADAM: A Method for Stochastic Optimization."
 *     arXiv preprint arXiv:1412.6980v8 (2014).
 */
template <typename Dtype>
class AdamSolver : public SGDSolver<Dtype> {
 public:
  explicit AdamSolver(const SolverParameter& param,
      size_t rank = 0U, Solver *root_solver = NULL)
      : SGDSolver<Dtype>(param, rank, root_solver) { AdamPreSolve();}
  explicit AdamSolver(const string& param_file,
      size_t rank = 0U, Solver *root_solver = NULL)
      : SGDSolver<Dtype>(param_file, rank, root_solver) { AdamPreSolve(); }
  virtual inline const char* type() const { return "Adam"; }

 protected:
  void AdamPreSolve();
  float ComputeUpdateValue(int param_id, void* handle, float rate, bool clear_grads) override;

  DISABLE_COPY_MOVE_AND_ASSIGN(AdamSolver);
};

}  // namespace caffe

#endif  // CAFFE_SGD_SOLVERS_HPP_
