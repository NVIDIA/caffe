#ifndef CAFFE_MULTIBOX_LOSS_LAYER_HPP_
#define CAFFE_MULTIBOX_LOSS_LAYER_HPP_

#include <map>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/bbox_util.hpp"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

/**
 * @brief Perform MultiBox operations. Including the following:
 *
 *  - decode the predictions.
 *  - perform matching between priors/predictions and ground truth.
 *  - use matched boxes and confidences to compute loss.
 *
 */

  //@RefineDet
  //Diff: @DataType 
template <typename Ftype, typename Btype>
class MultiBoxLossLayer : public LossLayer<Ftype, Btype> {
  typedef Ftype Dtype;
//Blob<Dtype> -> Blob 
 public:
  explicit MultiBoxLossLayer(const LayerParameter& param)
      : LossLayer<Ftype, Btype>(param) {}
  virtual void LayerSetUp(const vector<Blob*>& bottom,
      const vector<Blob*>& top);
  virtual void Reshape(const vector<Blob*>& bottom,
      const vector<Blob*>& top);

  virtual inline const char* type() const { return "MultiBoxLoss"; }
  // bottom[0] stores the location predictions.
  // bottom[1] stores the confidence predictions.
  // bottom[2] stores the prior bounding boxes.
  // bottom[3] stores the ground truth bounding boxes.
  //@Original 
  //virtual inline int ExactNumBottomBlobs() const { return 4; }
  //@RefineDet
  //Diff: @Function
  virtual inline int ExactNumBottomBlobs() const { return -1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob*>& bottom,
      const vector<Blob*>& top);
  virtual void Backward_cpu(const vector<Blob*>& top,
      const vector<bool>& propagate_down, const vector<Blob*>& bottom);

  // The internal localization loss layer.
  shared_ptr<LayerBase> loc_loss_layer_;
  LocLossType loc_loss_type_;
  float loc_weight_;
  // bottom vector holder used in Forward function.
  vector<Blob*> loc_bottom_vec_;
  // top vector holder used in Forward function.
  vector<Blob*> loc_top_vec_;
  // blob which stores the matched location prediction.
  shared_ptr<Blob> loc_pred_;
  // blob which stores the corresponding matched ground truth.
  shared_ptr<Blob> loc_gt_;
  // localization loss.
  shared_ptr<Blob> loc_loss_;

  // The internal confidence loss layer.
  shared_ptr<LayerBase> conf_loss_layer_;
  ConfLossType conf_loss_type_;
  // bottom vector holder used in Forward function.
  vector<Blob*> conf_bottom_vec_;
  // top vector holder used in Forward function.
  vector<Blob*> conf_top_vec_;
  // blob which stores the confidence prediction.
  shared_ptr<Blob> conf_pred_;
  // blob which stores the corresponding ground truth label.
  shared_ptr<Blob> conf_gt_;
  // confidence loss.
  shared_ptr<Blob> conf_loss_;

  MultiBoxLossParameter multibox_loss_param_;
  int num_classes_;
  bool share_location_;
  MatchType match_type_;
  float overlap_threshold_;
  bool use_prior_for_matching_;
  int background_label_id_;
  bool use_difficult_gt_;
  bool do_neg_mining_;
  float neg_pos_ratio_;
  float neg_overlap_;
  CodeType code_type_;
  bool encode_variance_in_target_;
  bool map_object_to_agnostic_;
  bool ignore_cross_boundary_bbox_;
  bool bp_inside_;
  MiningType mining_type_;

  int loc_classes_;
  int num_gt_;
  int num_;
  int num_priors_;

  int num_matches_;
  int num_conf_;
  vector<map<int, vector<int> > > all_match_indices_;
  vector<vector<int> > all_neg_indices_;

  // How to normalize the loss.
  LossParameter_NormalizationMode normalization_;
};

}  // namespace caffe

#endif  // CAFFE_MULTIBOX_LOSS_LAYER_HPP_
