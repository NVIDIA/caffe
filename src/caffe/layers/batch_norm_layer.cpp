#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/batch_norm_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Ftype, typename Btype>
void
BatchNormLayer<Ftype, Btype>::LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top) {
  BatchNormParameter param = this->layer_param_.batch_norm_param();
  moving_average_fraction_ = param.moving_average_fraction();

  clip_variance_ = false;
  //use_global_stats_ = false;
  use_global_stats_= param.use_global_stats();

  if (bottom[0]->num_axes() == 1)
    channels_ = 1;
  else
    channels_ = bottom[0]->shape(1);
  eps_ = std::max<float>(param.eps(), 0.00001f);

  scale_bias_ = false;
  scale_bias_ = param.scale_bias(); // by default = false;
  if (param.has_scale_filler() || param.has_bias_filler()) { // implicit set
    scale_bias_ = true;
  }

  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (scale_bias_)
      this->blobs_.resize(5);
    else
      this->blobs_.resize(3);

    const Type btype = blobs_type();
    const vector<int> shape { channels_ };
    const vector<int> shape1 { 1 };
    this->blobs_[0] = Blob::create(btype, btype);  // mean
    this->blobs_[0]->Reshape(shape);
    this->blobs_[0]->set_data(0.);
    this->blobs_[1] = Blob::create(btype, btype);  // variance1
    this->blobs_[1]->Reshape(shape);
    this->blobs_[1]->set_data(0.);
    this->blobs_[2] = Blob::create(btype, btype);  // variance correction
    this->blobs_[2]->Reshape(shape1);
    this->blobs_[2]->set_data(1.);
    if (scale_bias_) {
      this->blobs_[3] = Blob::create(btype, btype);  // scale
      this->blobs_[3]->Reshape(shape);
      this->blobs_[4] = Blob::create(btype, btype);  // bias
      this->blobs_[4]->Reshape(shape);
      if (param.has_scale_filler()) {
        // TODO
        if (btype == tp<Ftype>()) {
          shared_ptr<Filler<Ftype>> scale_filler(
              GetFiller<Ftype>(this->layer_param_.batch_norm_param().scale_filler()));
          scale_filler->Fill(this->blobs_[3].get());
        } else {
          shared_ptr<Filler<float>> scale_filler(
              GetFiller<float>(this->layer_param_.batch_norm_param().scale_filler()));
          scale_filler->Fill(this->blobs_[3].get());
        }
      } else {
        this->blobs_[3]->set_data(1.);
      }
      if (param.has_bias_filler()) {
        // TODO
        if (btype == tp<Ftype>()) {
          shared_ptr<Filler<Ftype>> bias_filler(
              GetFiller<Ftype>(this->layer_param_.batch_norm_param().bias_filler()));
          bias_filler->Fill(this->blobs_[4].get());
        } else {
          shared_ptr<Filler<float>> bias_filler(
              GetFiller<float>(this->layer_param_.batch_norm_param().bias_filler()));
          bias_filler->Fill(this->blobs_[4].get());
        }
      } else {
        this->blobs_[4]->set_data(0.);
      }
    }
    iter_ = 0;
  }

  // Mask statistics from optimization by setting local learning rates
  // for mean, variance, and the var_correction to zero.
  for (int i = 0; i < 3; ++i) {
    if (this->layer_param_.param_size() == i) {
      this->layer_param_.add_param();
    }
    //set lr and decay = 0 for global mean and variance
    this->layer_param_.mutable_param(i)->set_lr_mult(0.f);
    this->layer_param_.mutable_param(i)->set_decay_mult(0.f);
  }
  // set lr for scale and bias to 1
  if (scale_bias_) {
    for (int i = 3; i < 5; ++i) {
      if (this->layer_param_.param_size() == i) {
        this->layer_param_.add_param();
      }
      //set lr and decay = 1 for scale and bias
      this->layer_param_.mutable_param(i)->set_lr_mult(1.f);
      this->layer_param_.mutable_param(i)->set_decay_mult(1.f);
    }
  }

  // =====================================
  int N = bottom[0]->shape(0);
  int C = channels_;
  int H = (bottom[0]->num_axes() > 2) ? bottom[0]->shape(2) : 1;
  int W = (bottom[0]->num_axes() > 3) ? bottom[0]->shape(3) : 1;

  mean_ = Blob::create<Ftype>(C);
  var_ = Blob::create<Ftype>(C);
  inv_var_ = Blob::create<Ftype>(C);

  ones_C_ = Blob::create<Ftype>(C);
  ones_C_->set_data(1.);
  ones_N_ = Blob::create<Ftype>(N);
  ones_N_->set_data(1.);
  ones_HW_ = Blob::create<Ftype>(H*W);
  ones_HW_->set_data(1.);

  temp_C_ = Blob::create<Ftype>(C);
  temp_C_->set_data(0.);
  temp_NC_ = Blob::create<Ftype>(N*C);
  temp_NC_->set_data(1.);
  temp_NCHW_= Blob::create<Ftype>(N, C, H, W);
  x_norm_ = Blob::create<Ftype>(N, C, H, W);
}

template<typename Ftype, typename Btype>
void BatchNormLayer<Ftype, Btype>::Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top) {
  if (bottom[0]->num_axes() > 1)
    CHECK_EQ(bottom[0]->shape(1), channels_);
  top[0]->ReshapeLike(*bottom[0]);

  int N = bottom[0]->shape(0);
  int C = channels_;
  int H = (bottom[0]->num_axes() > 2) ? bottom[0]->shape(2) : 1;
  int W = (bottom[0]->num_axes() > 3) ? bottom[0]->shape(3) : 1;

  mean_->Reshape(C);
  var_->Reshape(C);
  inv_var_->Reshape(C);
  temp_C_->Reshape(C);

  ones_N_->Reshape(N);
  ones_N_->set_data(1.);
  ones_C_->Reshape(C);
  ones_C_->set_data(1.);
  ones_HW_->Reshape(H*W);
  ones_HW_->set_data(1.);

  temp_NC_->Reshape(N*C);
  temp_NCHW_->ReshapeLike(*bottom[0]);
  x_norm_->ReshapeLike(*bottom[0]);
}

template<typename Ftype, typename Btype>
void
BatchNormLayer<Ftype, Btype>::Forward_cpu(const vector<Blob*>& bottom, const vector<Blob*>& top) {
  int N = bottom[0]->shape(0);
  int C = channels_;
  int S = bottom[0]->count(0) / (N * C);
  int top_size = top[0]->count();

  const Ftype* bottom_data = bottom[0]->cpu_data<Ftype>();
  Ftype* top_data = top[0]->mutable_cpu_data<Ftype>();
  const Ftype* global_mean = this->blobs_[0]->template cpu_data<Ftype>();
  const Ftype* global_var  = this->blobs_[1]->template cpu_data<Ftype>();

  if (this->phase_ == TEST) {
    if (bottom[0] != top[0]) {
      caffe_copy(top_size, bottom_data, top_data);
    }
    //  Y = X- EX
    multicast_cpu<Ftype>(N, C, S, global_mean, temp_NCHW_->template mutable_cpu_data<Ftype>());
    caffe_axpy<Ftype>(top_size, Ftype(-1.), temp_NCHW_->template mutable_cpu_data<Ftype>(),
        top_data);
    //  inv_var = (eps + var)^(-0.5)
    caffe_copy<Ftype>(C, global_var, var_->template mutable_cpu_data<Ftype>());
    caffe_add_scalar<Ftype>(C, Ftype(eps_), var_->template mutable_cpu_data<Ftype>());
    caffe_powx<Ftype>(C, var_->template cpu_data<Ftype>(), Ftype(-0.5),
        inv_var_->template mutable_cpu_data<Ftype>());
      //  X_norm = (X-EX) * inv_var
    multicast_cpu<Ftype>(N, C, S, inv_var_->template cpu_data<Ftype>(),
        temp_NCHW_->template mutable_cpu_data<Ftype>());
    caffe_mul<Ftype>(top_size, top_data, temp_NCHW_->template cpu_data<Ftype>(), top_data);
  } else {
    compute_mean_per_channel_cpu<Ftype>(N, C, S, bottom_data,
        mean_->template mutable_cpu_data<Ftype>());
    multicast_cpu<Ftype>(N, C, S, mean_->template mutable_cpu_data<Ftype>(),
        temp_NCHW_-> template mutable_cpu_data<Ftype>());
    //  Y = X- EX
    if (bottom[0] != top[0]) {
      caffe_copy(top_size, bottom_data, top_data);
    }
    caffe_axpy<Ftype>(top_size, Ftype(-1.), temp_NCHW_->template mutable_cpu_data<Ftype>(),
        top_data);
    // compute variance E (X-EX)^2
    caffe_powx<Ftype>(top_size, top_data, Ftype(2.),
        temp_NCHW_->template mutable_cpu_data<Ftype>());
    compute_mean_per_channel_cpu<Ftype>(N, C, S, temp_NCHW_->template mutable_cpu_data<Ftype>(),
        var_->template mutable_cpu_data<Ftype>());
    //  inv_var= ( eps+ variance)^(-0.5)
    caffe_add_scalar<Ftype>(C, Ftype(eps_), var_->template mutable_cpu_data<Ftype>());
    caffe_powx<Ftype>(C, var_->template cpu_data<Ftype>(), Ftype(-0.5),
        inv_var_->template mutable_cpu_data<Ftype>());
    // X_norm = (X-EX) * inv_var
    multicast_cpu<Ftype>(N, C, S, inv_var_->template cpu_data<Ftype>(),
        temp_NCHW_->template mutable_cpu_data<Ftype>());
    caffe_mul<Ftype>(top_size, top_data, temp_NCHW_->template cpu_data<Ftype>(), top_data);
    // copy top to x_norm for backward
    caffe_copy<Ftype>(top_size, top_data, x_norm_->template mutable_cpu_data<Ftype>());

    // clip variance
    //  update global mean and variance
    if (iter_ > 1) {
      caffe_cpu_axpby<Ftype>(C, Ftype(1. - moving_average_fraction_),
          mean_->template cpu_data<Ftype>(), Ftype(moving_average_fraction_),
          this->blobs_[0]->template mutable_cpu_data<Ftype>());
      caffe_cpu_axpby<Ftype>(C, Ftype(1. - moving_average_fraction_),
          var_->template cpu_data<Ftype>(), Ftype(moving_average_fraction_),
          this->blobs_[1]->template mutable_cpu_data<Ftype>());
    } else {
      caffe_copy<Ftype>(C, mean_->template cpu_data<Ftype>(),
          this->blobs_[0]->template mutable_cpu_data<Ftype>());
      caffe_copy<Ftype>(C, var_->template cpu_data<Ftype>(),
          this->blobs_[1]->template mutable_cpu_data<Ftype>());
    }
    iter_++;
  }

  // -- STAGE 2:  Y = X_norm * scale[c] + shift[c]  -----------------
  if (scale_bias_) {
    // Y = X_norm * scale[c]
    const Blob& scale_data = *(this->blobs_[3]);
    multicast_cpu<Ftype>(N, C, S, scale_data.cpu_data<Ftype>(),
       temp_NCHW_->template mutable_cpu_data<Ftype>());
    caffe_mul<Ftype>(top_size, top_data, temp_NCHW_->template cpu_data<Ftype>(), top_data);
    // Y = Y + shift[c]
    const Blob& shift_data = *(this->blobs_[4]);
    multicast_cpu<Ftype>(N, C, S, shift_data.cpu_data<Ftype>(),
       temp_NCHW_->template mutable_cpu_data<Ftype>());
    caffe_add<Ftype>(top_size, top_data, temp_NCHW_->template mutable_cpu_data<Ftype>(), top_data);
  }
}

template<typename Ftype, typename Btype>
void BatchNormLayer<Ftype, Btype>::Backward_cpu(const vector<Blob*>& top,
    const vector<bool>& propagate_down, const vector<Blob*>& bottom) {
  int N = bottom[0]->shape(0);
  int C = channels_;
  int S = bottom[0]->count(0) / (N * C);
  int top_size = top[0]->count();
  const Btype* top_diff = top[0]->cpu_diff<Btype>();

  // --  STAGE 1: compute dE/d(scale) and dE/d(shift) ---------------
  if (scale_bias_) {
    // scale_diff: dE/d(scale)  =  sum(dE/dY .* X_norm)
    Btype* scale_diff = this->blobs_[3]->template mutable_cpu_diff<Btype>();
    caffe_mul<Btype>(top_size, top_diff, x_norm_->template cpu_data<Btype>(),
       temp_NCHW_->template mutable_cpu_diff<Btype>());
    compute_sum_per_channel_cpu<Btype>(N, C, S, temp_NCHW_->template cpu_diff<Btype>(), scale_diff);
    // shift_diff: dE/d(shift) = sum (dE/dY)
    Btype* shift_diff = this->blobs_[4]->template mutable_cpu_diff<Btype>();
    compute_sum_per_channel_cpu(N, C, S, top_diff, shift_diff);

    // --  STAGE 2: backprop dE/d(x_norm) = dE/dY .* scale ------------
    // dE/d(X_norm) = dE/dY * scale[c]
    const Btype* scale_data = this->blobs_[3]->template cpu_data<Btype>();
    multicast_cpu<Btype>(N, C, S, scale_data, temp_NCHW_->template mutable_cpu_data<Btype>());
    caffe_mul<Btype>(top_size, top_diff, temp_NCHW_->template cpu_data<Btype>(),
        x_norm_->template mutable_cpu_diff<Btype>());
    top_diff = x_norm_->template cpu_diff<Btype>();
  }

  // --  STAGE 3: backprop dE/dY --> dE/dX --------------------------
  // ATTENTION: from now on we will use notation Y:= X_norm
  //
  // if Y = (X-mean(X))/(sqrt(var(X)+eps)), then
  //    dE(Y)/dX =  (dE/dY - mean(dE/dY) - mean(dE/dY .* Y) .* Y) ./ sqrt(var(X) + eps)
  // where .* and ./ are element-wise product and division,
  //    mean, var, sum are computed along all dimensions except the channels.

  const Btype* top_data = x_norm_->template cpu_data<Btype>();
  Btype* bottom_diff = bottom[0]->template mutable_cpu_diff<Btype>();

  // temp = mean(dE/dY .* Y)
  caffe_mul<Btype>(top_size, top_diff, top_data, temp_NCHW_->template mutable_cpu_diff<Btype>());
  compute_mean_per_channel_cpu<Btype>(N, C, S, temp_NCHW_->template cpu_diff<Btype>(),
      temp_C_->template mutable_cpu_diff<Btype>());
  multicast_cpu<Btype>(N, C, S, temp_C_->template cpu_diff<Btype>(),
     temp_NCHW_->template mutable_cpu_diff<Btype>());
  // bottom = mean(dE/dY .* Y) .* Y
  caffe_mul(top_size, temp_NCHW_->template cpu_diff<Btype>(), top_data, bottom_diff);
  // temp = mean(dE/dY)
  compute_mean_per_channel_cpu<Btype>(N, C, S, top_diff,
      temp_C_->template mutable_cpu_diff<Btype>());
  multicast_cpu<Btype>(N, C, S, temp_C_->template cpu_diff<Btype>(),
     temp_NCHW_->template mutable_cpu_diff<Btype>());
  // bottom = mean(dE/dY) + mean(dE/dY .* Y) .* Y
  caffe_add(top_size, temp_NCHW_->template cpu_diff<Btype>(), bottom_diff, bottom_diff);
  // bottom = dE/dY - mean(dE/dY)-mean(dE/dY \cdot Y) \cdot Y
  caffe_cpu_axpby(top_size, Btype(1.), top_diff, Btype(-1.), bottom_diff);
  // dE/dX = dE/dX ./ sqrt(var(X) + eps)
  multicast_cpu<Btype>(N, C, S, inv_var_->template cpu_data<Btype>(),
     temp_NCHW_->template mutable_cpu_data<Btype>());
  caffe_mul(top_size, bottom_diff, temp_NCHW_->template cpu_data<Btype>(), bottom_diff);
}

INSTANTIATE_CLASS_FB(BatchNormLayer);

}  // namespace caffe
