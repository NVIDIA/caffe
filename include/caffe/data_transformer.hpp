#ifndef CAFFE_DATA_TRANSFORMER_HPP
#define CAFFE_DATA_TRANSFORMER_HPP

#ifdef USE_OPENCV

#include <opencv2/core/core.hpp>

#endif  // USE_OPENCV

#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/blocking_queue.hpp"

#include "google/protobuf/repeated_field.h"
using google::protobuf::RepeatedPtrField;

namespace caffe {

/**
 * @brief Applies common transformations to the input data, such as
 * scaling, mirroring, substracting the image mean...
 */
template<typename Dtype>
class DataTransformer {
 public:
  DataTransformer(const TransformationParameter& param, Phase phase);
  virtual ~DataTransformer() {}

  /**
   * @brief Initialize the Random number generations if needed by the
   *    transformation.
   */
  void InitRand();

  /**
   * @brief Generates a random integer from Uniform({0, 1, ..., n-1}).
   *
   * @param n
   *    The upperbound (exclusive) value of the random number.
   * @return
   *    A uniformly random integer value from ({0, 1, ..., n-1}).
   */
  unsigned int Rand(int n) const {
    CHECK_GT(n, 0);
    return Rand() % n;
  }

#ifndef CPU_ONLY
  void TransformGPU(int N, int C, int H, int W, size_t sizeof_element, const Dtype* in, Dtype* out,
      const unsigned int* rands);
#endif
  void Copy(const Datum& datum, Dtype* data, size_t& out_sizeof_element);
  void Copy(const cv::Mat& datum, Dtype* data);
  void CopyPtrEntry(shared_ptr<Datum> datum, Dtype* transformed_ptr, size_t& out_sizeof_element,
      bool output_labels, Dtype* label);

#ifdef USE_OPENCV
  /**
   * @brief Whether there are any "variable_sized" transformations defined
   * in the data layer's transform_param block.
   */
  bool var_sized_transforms_enabled() const;

  /**
   * @brief Calculate the final shape from applying the "variable_sized"
   * transformations defined in the data layer's transform_param block
   * on the provided image, without actually performing any transformations.
   *
   * @param orig_shape
   *    The shape of the data to be transformed.
   */
  vector<int> var_sized_transforms_shape(const vector<int>& orig_shape) const;

  /**
   * @brief Applies "variable_sized" transformations defined in the data layer's
   * transform_param block to the data.
   *
   * @param old_datum
   *    The source Datum containing data of arbitrary shape.
   * @param new_datum
   *    The destination Datum that will store transformed data of a fixed
   *    shape. Suitable for other transformations.
   */
  void VariableSizedTransforms(Datum* datum);

  bool        var_sized_image_random_resize_enabled() const;
  vector<int> var_sized_image_random_resize_shape(const vector<int>& prev_shape) const;
  void        var_sized_image_random_resize(cv::Mat& img);
  bool        var_sized_image_random_crop_enabled() const;
  vector<int> var_sized_image_random_crop_shape(const vector<int>& prev_shape) const;
  void        var_sized_image_random_crop(cv::Mat& img);
  bool        var_sized_image_center_crop_enabled() const;
  vector<int> var_sized_image_center_crop_shape(const vector<int>& prev_shape) const;
  void        var_sized_image_center_crop(cv::Mat& img);
#endif

  /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to the data.
   *
   * @param datum
   *    Datum containing the data to be transformed.
   * @param transformed_blob
   *    This is destination blob. It can be part of top blob's data if
   *    set_cpu_data() is used. See data_layer.cpp for an example.
   */
  void Transform(const Datum& datum, TBlob<Dtype>* transformed_blob);

  /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to the data.
   *
   * @param datum
   *    Datum containing the data to be transformed.
   * @param rand1
   *    Random value (0, RAND_MAX+1]
   * @param rand2
   *    Random value (0, RAND_MAX+1]
   * @param rand3
   *    Random value (0, RAND_MAX+1]
   * @param transformed_blob
   *    This is destination blob. It can be part of top blob's data if
   *    set_cpu_data() is used. See data_layer.cpp for an example.
   */
  void TransformPtrEntry(shared_ptr<Datum> datum, Dtype* transformed_ptr,
      std::array<unsigned int, 3> rand, bool output_labels, Dtype* label);

  /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to a vector of Datum.
   *
   * @param datum_vector
   *    A vector of Datum containing the data to be transformed.
   * @param transformed_blob
   *    This is destination blob. It can be part of top blob's data if
   *    set_cpu_data() is used. See memory_layer.cpp for an example.
   */
  void Transform(const vector<Datum> & datum_vector,
                TBlob<Dtype>* transformed_blob);

  /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to the annotated data.
   *
   * @param anno_datum
   *    AnnotatedDatum containing the data and annotation to be transformed.
   * @param transformed_blob
   *    This is destination blob. It can be part of top blob's data if
   *    set_cpu_data() is used. See annotated_data_layer.cpp for an example.
   * @param transformed_anno_vec
   *    This is destination annotation.
   */
  void Transform(const AnnotatedDatum& anno_datum,
                 TBlob<Dtype>* transformed_blob,
                 RepeatedPtrField<AnnotationGroup>* transformed_anno_vec);
  void Transform(const AnnotatedDatum& anno_datum,
                 TBlob<Dtype>* transformed_blob,
                 RepeatedPtrField<AnnotationGroup>* transformed_anno_vec,
                 bool* do_mirror);
  void Transform(const AnnotatedDatum& anno_datum,
                 TBlob<Dtype>* transformed_blob,
                 vector<AnnotationGroup>* transformed_anno_vec,
                 bool* do_mirror);
  void Transform(const AnnotatedDatum& anno_datum,
                 TBlob<Dtype>* transformed_blob,
                 vector<AnnotationGroup>* transformed_anno_vec);
				 

  /**
   * @brief Transform the annotation according to the transformation applied
   * to the datum.
   *
   * @param anno_datum
   *    AnnotatedDatum containing the data and annotation to be transformed.
   * @param do_resize
   *    If true, resize the annotation accordingly before crop.
   * @param crop_bbox
   *    The cropped region applied to anno_datum.datum()
   * @param do_mirror
   *    If true, meaning the datum has mirrored.
   * @param transformed_anno_group_all
   *    Stores all transformed AnnotationGroup.
   */
  void TransformAnnotation(
      const AnnotatedDatum& anno_datum, const bool do_resize,
      const NormalizedBBox& crop_bbox, const bool do_mirror,
      RepeatedPtrField<AnnotationGroup>* transformed_anno_group_all);

  /**
   * @brief Crops the datum according to bbox.
   */
  void CropImage(const Datum& datum, const NormalizedBBox& bbox,
                 Datum* crop_datum);

  /**
   * @brief Crops the datum and AnnotationGroup according to bbox.
   */
  void CropImage(const AnnotatedDatum& anno_datum, const NormalizedBBox& bbox,
                 AnnotatedDatum* cropped_anno_datum);

  /**
   * @brief Expand the datum.
   */
  void ExpandImage(const Datum& datum, const float expand_ratio,
                   NormalizedBBox* expand_bbox, Datum* expanded_datum);

  /**
   * @brief Expand the datum and adjust AnnotationGroup.
   */
  void ExpandImage(const AnnotatedDatum& anno_datum,
                   AnnotatedDatum* expanded_anno_datum);

  /**
   * @brief Apply distortion to the datum.
   */
  void DistortImage(const Datum& datum, Datum* distort_datum);
  
#ifdef USE_OPENCV
  /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to a vector of Mat.
   *
   * @param mat_vector
   *    A vector of Mat containing the data to be transformed.
   * @param transformed_blob
   *    This is destination blob. It can be part of top blob's data if
   *    set_cpu_data() is used. See memory_layer.cpp for an example.
   */
  void Transform(const vector<cv::Mat>& mat_vector, TBlob<Dtype>* transformed_blob);

  /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to a cv::Mat
   *
   * @param cv_img
   *    cv::Mat containing the data to be transformed.
   * @param transformed_blob
   *    This is destination blob. It can be part of top blob's data if
   *    set_cpu_data() is used. See image_data_layer.cpp for an example.
   */
  void Transform(const cv::Mat& cv_img, TBlob<Dtype>* transformed_blob);

  /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to a cv::Mat
   *
   * @param cv_img
   *    cv::Mat containing the data to be transformed.
   * @param transformed_blob
   *    This is destination blob. It can be part of top blob's data if
   *    set_cpu_data() is used. See image_data_layer.cpp for an example.
   * @param rand1
   *    Random value (0, RAND_MAX+1]
   * @param rand2
   *    Random value (0, RAND_MAX+1]
   * @param rand3
   *    Random value (0, RAND_MAX+1]
   */
  void TransformPtr(const cv::Mat& cv_img, Dtype* transformed_ptr,
      const std::array<unsigned int, 3>& rand);
#endif  // USE_OPENCV

  vector<int> InferDatumShape(const Datum& datum);
#ifdef USE_OPENCV
  vector<int> InferCVMatShape(const cv::Mat& img);
  /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to a cv::Mat
   *
   * @param cv_img
   *    cv::Mat containing the data to be transformed.
   * @param transformed_blob
   *    This is destination blob. It can be part of top blob's data if
   *    set_cpu_data() is used. See image_data_layer.cpp for an example.
   */
  void Transform(const cv::Mat& cv_img, TBlob<Dtype>* transformed_blob,
                 NormalizedBBox* crop_bbox, bool* do_mirror);

  /**
   * @brief Crops img according to bbox.
   */
  void CropImage(const cv::Mat& img, const NormalizedBBox& bbox,
                 cv::Mat* crop_img);

  /**
   * @brief Expand img to include mean value as background.
   */
  void ExpandImage(const cv::Mat& img, const float expand_ratio,
                   NormalizedBBox* expand_bbox, cv::Mat* expand_img);

  void TransformInv(const TBlob<Dtype>* blob, vector<cv::Mat>* cv_imgs);
  void TransformInv(const Dtype* data, cv::Mat* cv_img, const int height,
                    const int width, const int channels);
					
#endif  // USE_OPENCV

  /**
   * @brief Infers the shape of transformed_blob will have when
   *    the transformation is applied to the data.
   *
   * @param bottom_shape
   *    The shape of the data to be transformed.
   */
  vector<int> InferBlobShape(const vector<int>& bottom_shape, bool use_gpu = false);

  /**
   * @brief Infers the shape of transformed_blob will have when
   *    the transformation is applied to the data.
   *
   * @param datum
   *    Datum containing the data to be transformed.
   */
  vector<int> InferBlobShape(const Datum& datum, bool use_gpu /*= false*/);
  vector<int> InferBlobShape(const Datum& datum);
  
  /**
   * @brief Infers the shape of transformed_blob will have when
   *    the transformation is applied to the data.
   *    It uses the first element to infer the shape of the blob.
   *
   * @param datum_vector
   *    A vector of Datum containing the data to be transformed.
   */
  vector<int> InferBlobShape(const vector<Datum> & datum_vector);
  
#ifdef USE_OPENCV
  vector<int> InferBlobShape(const vector<cv::Mat> & mat_vector);
  /**
   * @brief Infers the shape of transformed_blob will have when
   *    the transformation is applied to the data.
   *
   * @param cv_img
   *    cv::Mat containing the data to be transformed.
   */
  vector<int> InferBlobShape(const cv::Mat& cv_img, bool use_gpu /*= false*/);
  
  vector<int> InferBlobShape(const cv::Mat& cv_img);  
#endif  // USE_OPENCV

  void Fill3Randoms(unsigned int* rand) const;
  const TransformationParameter& transform_param() const {
    return param_;
  }

 protected:
  unsigned int Rand() const;
  void TransformGPU(const Datum& datum, Dtype* transformed_data,
      const std::array<unsigned int, 3>& rand);
  void Transform(const Datum& datum, Dtype* transformed_data,
      const std::array<unsigned int, 3>& rand);
  void TransformPtrInt(Datum& datum, Dtype* transformed_data,
      const std::array<unsigned int, 3>& rand);

  // Transform and return the transformation information.
  void Transform(const Datum& datum, Dtype* transformed_data,
                 NormalizedBBox* crop_bbox, bool* do_mirror);
  void Transform(const Datum& datum, Dtype* transformed_data);

  /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to the data and return transform information.
   */
  void Transform(const Datum& datum, TBlob<Dtype>* transformed_blob,
                 NormalizedBBox* crop_bbox, bool* do_mirror);

  // Tranformation parameters
  TransformationParameter param_;
  shared_ptr<Caffe::RNG> rng_;
  Phase phase_;
  TBlob<float> data_mean_;
  vector<float> mean_values_;
#ifndef CPU_ONLY
  GPUMemory::Workspace mean_values_gpu_;
#endif
};

}  // namespace caffe

#endif  // CAFFE_DATA_TRANSFORMER_HPP_
