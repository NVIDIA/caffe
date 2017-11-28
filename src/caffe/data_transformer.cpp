#ifdef USE_OPENCV

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#endif  // USE_OPENCV

#include <string>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/util/bbox_util.hpp"
#include "caffe/util/im_transforms.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"


namespace caffe {

template<typename Dtype>
DataTransformer<Dtype>::DataTransformer(const TransformationParameter& param, Phase phase)
    : param_(param), phase_(phase) {
  // check if we want to use mean_file
  if (param_.has_mean_file()) {
    CHECK_EQ(param_.mean_value_size(), 0)
      << "Cannot specify mean_file and mean_value at the same time";
    const string& mean_file = param.mean_file();
    if (Caffe::root_solver()) {
      LOG(INFO) << "Loading mean file from: " << mean_file;
    }
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
    data_mean_.FromProto(blob_proto);
  }
  // check if we want to use mean_value
  if (param_.mean_value_size() > 0) {
    CHECK(!param_.has_mean_file())
    << "Cannot specify mean_file and mean_value at the same time";
    for (int c = 0; c < param_.mean_value_size(); ++c) {
      mean_values_.push_back(param_.mean_value(c));
    }
  }
}

#ifdef USE_OPENCV

template<typename Dtype>
void DataTransformer<Dtype>::Copy(const cv::Mat& cv_img, Dtype *data) {
  const int channels = cv_img.channels();
  const int height = cv_img.rows;
  const int width = cv_img.cols;

  CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";

  int top_index;
  for (int c = 0; c < channels; ++c) {
    for (int h = 0; h < height; ++h) {
      const uchar *ptr = cv_img.ptr<uchar>(h);
      for (int w = 0; w < width; ++w) {
        int img_index = w * channels + c;
        top_index = (c * height + h) * width + w;
        data[top_index] = static_cast<Dtype>(ptr[img_index]);
      }
    }
  }
}
#endif

template<typename Dtype>
void DataTransformer<Dtype>::Copy(const Datum& datum, Dtype* data, size_t& out_sizeof_element) {
  // If datum is encoded, decoded and transform the cv::image.
  if (datum.encoded()) {
#ifdef USE_OPENCV
    CHECK(!(param_.force_color() && param_.force_gray()))
    << "cannot set both force_color and force_gray";
    cv::Mat cv_img;
    if (param_.force_color() || param_.force_gray()) {
      // If force_color then decode in color otherwise decode in gray.
      cv_img = DecodeDatumToCVMat(datum, param_.force_color());
    } else {
      cv_img = DecodeDatumToCVMatNative(datum);
    }
    // Transform the cv::image into blob.
    Copy(cv_img, data);
    out_sizeof_element = sizeof(Dtype);
    return;
#else
    LOG(FATAL) << "Encoded datum requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  } else {
    if (param_.force_color() || param_.force_gray()) {
      LOG(ERROR) << "Force_color and force_gray are for encoded datum only";
    }
  }

#ifndef CPU_ONLY
  const string& datum_data = datum.data();
  const int N = datum.channels() * datum.height() * datum.width();
  const void* src_ptr;
  if (datum_data.size() > 0) {
    CHECK_LE(sizeof(uint8_t), sizeof(Dtype));
    CHECK_EQ(N, datum_data.size());
    out_sizeof_element = sizeof(uint8_t);
    src_ptr = &datum_data.front();
  } else {
    CHECK_LE(sizeof(float), sizeof(Dtype));
    out_sizeof_element = sizeof(float);
    src_ptr = &datum.float_data().Get(0);
  }
  cudaStream_t stream = Caffe::th_stream_aux(Caffe::STREAM_ID_TRANSFORMER);
  CUDA_CHECK(cudaMemcpyAsync(data, src_ptr, N * out_sizeof_element,
      cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
#else
  NO_GPU;
#endif
}

template<typename Dtype>
void DataTransformer<Dtype>::CopyPtrEntry(
    shared_ptr<Datum> datum,
    Dtype* transformed_ptr,
    size_t& out_sizeof_element,
    bool output_labels, Dtype *label) {
  if (output_labels) {
    *label = datum->label();
  }
  Copy(*datum, transformed_ptr, out_sizeof_element);
}

template<typename Dtype>
void DataTransformer<Dtype>::Fill3Randoms(unsigned int *rand) const {
  rand[0] = rand[1] = rand[2] = 0;
  if (param_.mirror()) {
    rand[0] = Rand() + 1;
  }
  if (phase_ == TRAIN && param_.crop_size()) {
    rand[1] = Rand() + 1;
    rand[2] = Rand() + 1;
  }
}

#ifdef USE_OPENCV
template<typename Dtype>
bool DataTransformer<Dtype>::var_sized_transforms_enabled() const {
  return param_.var_sz_img_enabled();
}

template<typename Dtype>
vector<int> DataTransformer<Dtype>::var_sized_transforms_shape(
    const vector<int>& orig_shape) const {
  CHECK_EQ(orig_shape.size(), 4);
  // All of the transforms (random resize, random crop, center crop)
  // can be enabled, and they operate sequentially, one after the other.
  vector<int> shape(orig_shape);
  if (var_sized_image_random_resize_enabled()) {
    shape = var_sized_image_random_resize_shape(shape);
  }
  if (var_sized_image_random_crop_enabled()) {
    shape = var_sized_image_random_crop_shape(shape);
  }
  if (var_sized_image_center_crop_enabled()) {
    shape = var_sized_image_center_crop_shape(shape);
  }
  CHECK_NE(shape[2], 0)
      << "variable sized transform has invalid output height; did you forget to crop?";
  CHECK_NE(shape[3], 0)
      << "variable sized transform has invalid output width; did you forget to crop?";
  return shape;
}

template<typename Dtype>
void DataTransformer<Dtype>::VariableSizedTransforms(Datum* datum) {
  cv::Mat varsz_img;
  if (datum->encoded()) {
    CHECK(!(param_.force_color() && param_.force_gray()))
        << "cannot set both force_color and force_gray";
    if (param_.force_color() || param_.force_gray()) {
      // If force_color then decode in color otherwise decode in gray.
      DecodeDatumToCVMat(*datum, param_.force_color(), varsz_img);
    } else {
      DecodeDatumToCVMatNative(*datum, varsz_img);
    }
  } else {
    DatumToCVMat(*datum, varsz_img);
  }
  if (var_sized_image_random_resize_enabled()) {
    var_sized_image_random_resize(varsz_img);
  }
  if (var_sized_image_random_crop_enabled()) {
    var_sized_image_random_crop(varsz_img);
  }
  if (var_sized_image_center_crop_enabled()) {
    var_sized_image_center_crop(varsz_img);
  }
  CVMatToDatum(varsz_img, *datum);
}

template<typename Dtype>
bool DataTransformer<Dtype>::var_sized_image_random_resize_enabled() const {
  const int resize_lower = param_.img_rand_resize_lower();
  const int resize_upper = param_.img_rand_resize_upper();
  if (resize_lower == 0 && resize_upper == 0) {
    return false;
  } else if (resize_lower != 0 && resize_upper != 0) {
    return true;
  }
  LOG(FATAL)
      << "random resize 'lower' and 'upper' parameters must either "
      "both be zero or both be nonzero";
  return false;
}

template<typename Dtype>
vector<int> DataTransformer<Dtype>::var_sized_image_random_resize_shape(
    const vector<int>& prev_shape) const {
  CHECK(var_sized_image_random_resize_enabled())
      << "var sized transform must be enabled";
  CHECK_EQ(prev_shape.size(), 4)
      << "input shape should always have 4 axes (NCHW)";
  vector<int> shape(4);
  shape[0] = 1;
  shape[1] = prev_shape[1];
  // The output of a random resize is itself a variable sized image.
  // By itself a random resize cannot produce an image that is valid input for
  // downstream transformations, and must instead be terminated by a
  // variable-sized crop (either random or center).
  shape[2] = 0;
  shape[3] = 0;
  return shape;
}

template<typename Dtype>
void DataTransformer<Dtype>::var_sized_image_random_resize(cv::Mat& img) {
  const int resize_lower = param_.img_rand_resize_lower();
  const int resize_upper = param_.img_rand_resize_upper();
  CHECK_GT(resize_lower, 0)
      << "random resize lower bound parameter must be positive";
  CHECK_GT(resize_upper, 0)
      << "random resize lower bound parameter must be positive";
  int resize_size = -1;
  caffe_rng_uniform(
      1,
      static_cast<float>(resize_lower), static_cast<float>(resize_upper),
      &resize_size);
  CHECK_NE(resize_size, -1)
      << "uniform random sampling inexplicably failed";
  const int img_height = img.rows;
  const int img_width = img.cols;
  const double scale = (img_width >= img_height) ?
      ((static_cast<double>(resize_size)) / (static_cast<double>(img_height))) :
      ((static_cast<double>(resize_size)) / (static_cast<double>(img_width)));
  const int resize_height = static_cast<int>(std::round(scale * static_cast<double>(img_height)));
  const int resize_width = static_cast<int>(std::round(scale * static_cast<double>(img_width)));
  if (resize_height < img_height || resize_width < img_width) {
    // Downsample with pixel area relation interpolation.
    CHECK_LE(scale, 1.0);
    CHECK_LE(resize_height, img_height)
        << "cannot downsample width without downsampling height";
    CHECK_LE(resize_width, img_width)
        << "cannot downsample height without downsampling width";
    cv::resize(
        img, img,
        cv::Size(resize_width, resize_height),
        0.0, 0.0,
        cv::INTER_AREA);
  } else if (resize_height > img_height || resize_width > img_width) {
    // Upsample with cubic interpolation.
    CHECK_GE(scale, 1.0);
    CHECK_GE(resize_height, img_height)
        << "cannot upsample width without upsampling height";
    CHECK_GE(resize_width, img_width)
        << "cannot upsample height without upsampling width";
    cv::resize(
        img, img,
        cv::Size(resize_width, resize_height),
        0.0, 0.0,
        cv::INTER_CUBIC);
  } else if (resize_height == img_height && resize_width == img_width) {
  } else {
    LOG(FATAL)
        << "unreachable random resize shape: ("
        << img_width << ", " << img_height << ") => ("
        << resize_width << ", " << resize_height << ")";
  }
}

template<typename Dtype>
bool DataTransformer<Dtype>::var_sized_image_random_crop_enabled() const {
  return this->phase_ == TRAIN && param_.crop_size() > 0;
}

template<typename Dtype>
vector<int> DataTransformer<Dtype>::var_sized_image_random_crop_shape(
    const vector<int>& prev_shape) const {
  CHECK(var_sized_image_random_crop_enabled())
      << "var sized transform must be enabled";
  const int crop_size = param_.crop_size();
  CHECK_EQ(prev_shape.size(), 4)
      << "input shape should always have 4 axes (NCHW)";
  vector<int> shape(4);
  shape[0] = 1;
  shape[1] = prev_shape[1];
  shape[2] = crop_size;
  shape[3] = crop_size;
  return shape;
}

template<typename Dtype>
void DataTransformer<Dtype>::var_sized_image_random_crop(cv::Mat& img) {
  const int crop_size = param_.crop_size();
  CHECK_GT(crop_size, 0)
      << "random crop size parameter must be positive";
  const int img_height = img.rows;
  const int img_width = img.cols;
  CHECK_GE(img_height, crop_size)
      << "crop size parameter must be at least as large as the image height";
  CHECK_GE(img_width, crop_size)
      << "crop size parameter must be at least as large as the image width";
  int crop_offset_h = -1;
  int crop_offset_w = -1;
  caffe_rng_uniform(1, 0.0f, static_cast<float>(img_height - crop_size), &crop_offset_h);
  caffe_rng_uniform(1, 0.0f, static_cast<float>(img_width - crop_size), &crop_offset_w);
  CHECK_NE(crop_offset_h, -1)
      << "uniform random sampling inexplicably failed";
  CHECK_NE(crop_offset_w, -1)
      << "uniform random sampling inexplicably failed";
  cv::Mat ROI(img, cv::Rect(crop_offset_w, crop_offset_h, crop_size, crop_size));
  ROI.copyTo(img);
}

template<typename Dtype>
bool DataTransformer<Dtype>::var_sized_image_center_crop_enabled() const {
  return this->phase_ == TEST && param_.crop_size() > 0;
}

template<typename Dtype>
vector<int> DataTransformer<Dtype>::var_sized_image_center_crop_shape(
    const vector<int>& prev_shape) const {
  CHECK(var_sized_image_center_crop_enabled())
      << "var sized transform must be enabled";
  const int crop_size = param_.crop_size();
  CHECK_EQ(prev_shape.size(), 4)
      << "input shape should always have 4 axes (NCHW)";
  vector<int> shape(4);
  shape[0] = 1;
  shape[1] = prev_shape[1];
  shape[2] = crop_size;
  shape[3] = crop_size;
  return shape;
}

template<typename Dtype>
void DataTransformer<Dtype>::var_sized_image_center_crop(cv::Mat& img) {
  const int crop_size = param_.crop_size();
  CHECK_GT(crop_size, 0)
      << "center crop size parameter must be positive";
  const int img_height = img.rows;
  const int img_width = img.cols;
  CHECK_GE(img_height, crop_size)
      << "crop size parameter must be at least as large as the image height";
  CHECK_GE(img_width, crop_size)
      << "crop size parameter must be at least as large as the image width";
  const int crop_offset_h = (img_height - crop_size) / 2;
  const int crop_offset_w = (img_width - crop_size) / 2;
  cv::Mat ROI(img, cv::Rect(crop_offset_w, crop_offset_h, crop_size, crop_size));
  ROI.copyTo(img);
}
#endif

#ifndef CPU_ONLY

template<typename Dtype>
void DataTransformer<Dtype>::TransformGPU(const Datum& datum,
    Dtype *transformed_data, const std::array<unsigned int, 3>& rand) {

  unsigned int *randoms =
      reinterpret_cast<unsigned int *>(GPUMemory::thread_pinned_buffer(sizeof(unsigned int) * 3));
  std::memcpy(randoms, &rand.front(), sizeof(unsigned int) * 3);  // NOLINT(caffe/alt_fn)

  vector<int> datum_shape = InferBlobShape(datum, 1);
  TBlob<Dtype> original_data;
  original_data.Reshape(datum_shape);

  Dtype *original_data_gpu_ptr;
  size_t out_sizeof_element = 0;
  if (datum.encoded()) {
    Dtype *original_data_cpu_ptr = original_data.mutable_cpu_data();
    Copy(datum, original_data_cpu_ptr, out_sizeof_element);
    original_data_gpu_ptr = original_data.mutable_gpu_data();
  } else {
    original_data_gpu_ptr = original_data.mutable_gpu_data();
    Copy(datum, original_data_gpu_ptr, out_sizeof_element);
  }

  TransformGPU(1,
      datum.channels(),
      datum.height(),
      datum.width(),
      out_sizeof_element,
      original_data_gpu_ptr,
      transformed_data,
      randoms);
}

#endif


template<typename Dtype>
void DataTransformer<Dtype>::Transform(const Datum& datum,
    Dtype *transformed_data, const std::array<unsigned int, 3>& rand) {
  const string& data = datum.data();
  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();

  const int crop_size = param_.crop_size();
  const float scale = param_.scale();
  const bool do_mirror = param_.mirror() && (rand[0] % 2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_uint8 = data.size() > 0;
  const bool has_mean_values = mean_values_.size() > 0;

  CHECK_GT(datum_channels, 0);
  CHECK_GE(datum_height, crop_size);
  CHECK_GE(datum_width, crop_size);

  const float* mean = NULL;
  if (has_mean_file) {
    CHECK_EQ(datum_channels, data_mean_.channels());
    CHECK_EQ(datum_height, data_mean_.height());
    CHECK_EQ(datum_width, data_mean_.width());
    mean = data_mean_.cpu_data();
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == datum_channels)
        << "Specify either 1 mean_value or as many as channels: " << datum_channels;
    if (datum_channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < datum_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }

  int height = datum_height;
  int width = datum_width;

  int h_off = 0;
  int w_off = 0;
  if (crop_size) {
    height = crop_size;
    width = crop_size;
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = rand[1] % (datum_height - crop_size + 1);
      w_off = rand[2] % (datum_width - crop_size + 1);
    } else {
      h_off = (datum_height - crop_size) / 2;
      w_off = (datum_width - crop_size) / 2;
    }
  }

  int top_index, data_index, ch, cdho;
  const int m = do_mirror ? -1 : 1;

  if (has_uint8) {
    Dtype datum_element, mnv;

    if (scale == 1.F) {
      for (int c = 0; c < datum_channels; ++c) {
        cdho = c * datum_height + h_off;
        ch = c * height;
        mnv = has_mean_values && !has_mean_file ? mean_values_[c] : 0.F;
        for (int h = 0; h < height; ++h) {
          top_index = do_mirror ? (ch + h + 1) * width - 1 : (ch + h) * width;
          data_index = (cdho + h) * datum_width + w_off;
          for (int w = 0; w < width; ++w) {
            datum_element = static_cast<unsigned char>(data[data_index]);
            if (has_mean_file) {
              transformed_data[top_index] = datum_element - mean[data_index];
            } else {
              if (has_mean_values) {
                transformed_data[top_index] = datum_element - mnv;
              } else {
                transformed_data[top_index] = datum_element;
              }
            }
            ++data_index;
            top_index += m;
          }
        }
      }
    } else {
      for (int c = 0; c < datum_channels; ++c) {
        cdho = c * datum_height + h_off;
        ch = c * height;
        mnv = has_mean_values && !has_mean_file ? mean_values_[c] : 0.F;
        for (int h = 0; h < height; ++h) {
          top_index = do_mirror ? (ch + h + 1) * width - 1 : (ch + h) * width;
          data_index = (cdho + h) * datum_width + w_off;
          for (int w = 0; w < width; ++w) {
            datum_element = static_cast<unsigned char>(data[data_index]);
            if (has_mean_file) {
              transformed_data[top_index] = (datum_element - mean[data_index]) * scale;
            } else {
              if (has_mean_values) {
                transformed_data[top_index] = (datum_element - mnv) * scale;
              } else {
                transformed_data[top_index] = datum_element * scale;
              }
            }
            ++data_index;
            top_index += m;
          }
        }
      }
    }
  } else {
    Dtype datum_element;
    for (int c = 0; c < datum_channels; ++c) {
      cdho = c * datum_height + h_off;
      ch = c * height;
      for (int h = 0; h < height; ++h) {
        top_index = do_mirror ? (ch + h + 1) * width - 1 : (ch + h) * width;
        data_index = (cdho + h) * datum_width + w_off;
        for (int w = 0; w < width; ++w) {
          datum_element = datum.float_data(data_index);
          if (has_mean_file) {
            transformed_data[top_index] = (datum_element - mean[data_index]) * scale;
          } else {
            if (has_mean_values) {
              transformed_data[top_index] = (datum_element - mean_values_[c]) * scale;
            } else {
              transformed_data[top_index] = datum_element * scale;
            }
          }
          ++data_index;
          top_index += m;
        }
      }
    }
  }
}

// do_mirror, h_off, w_off require that random values be passed in,
// because the random draws should have been taken in deterministic order
template<typename Dtype>
void DataTransformer<Dtype>::TransformPtrInt(Datum& datum,
    Dtype *transformed_data, const std::array<unsigned int, 3>& rand) {
  Transform(datum, transformed_data, rand);
}

template<typename Dtype>
void DataTransformer<Dtype>::TransformPtrEntry(shared_ptr<Datum> datum,
    Dtype *transformed_ptr,
    std::array<unsigned int, 3> rand,
    bool output_labels,
    Dtype *label) {
  // Get label from datum if needed
  if (output_labels) {
    *label = datum->label();
  }

  // If datum is encoded, decoded and transform the cv::image.
  if (datum->encoded()) {
#ifdef USE_OPENCV
    CHECK(!(param_.force_color() && param_.force_gray()))
    << "cannot set both force_color and force_gray";
    cv::Mat cv_img;
    if (param_.force_color() || param_.force_gray()) {
      // If force_color then decode in color otherwise decode in gray.
      cv_img = DecodeDatumToCVMat(*datum, param_.force_color());
    } else {
      cv_img = DecodeDatumToCVMatNative(*datum);
    }
    // Transform the cv::image into blob.
    TransformPtr(cv_img, transformed_ptr, rand);
#else
    LOG(FATAL) << "Encoded datum requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  } else {
    TransformPtrInt(*datum, transformed_ptr, rand);
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const Datum& datum,
    TBlob<Dtype> *transformed_blob) {
  // If datum is encoded, decoded and transform the cv::image.
  if (datum.encoded()) {
#ifdef USE_OPENCV
    CHECK(!(param_.force_color() && param_.force_gray()))
    << "cannot set both force_color and force_gray";
    cv::Mat cv_img;
    if (param_.force_color() || param_.force_gray()) {
      // If force_color then decode in color otherwise decode in gray.
      cv_img = DecodeDatumToCVMat(datum, param_.force_color());
    } else {
      cv_img = DecodeDatumToCVMatNative(datum);
    }
    // Transform the cv::image into blob.
    Transform(cv_img, transformed_blob);
    return;
#else
    LOG(FATAL) << "Encoded datum requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  } else {
    if (param_.force_color() || param_.force_gray()) {
      LOG(ERROR) << "force_color and force_gray only for encoded datum";
    }
  }

  const int crop_size = param_.crop_size();
  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();

  // Check dimensions.
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int num = transformed_blob->num();

  CHECK_EQ(channels, datum_channels);
  CHECK_LE(height, datum_height);
  CHECK_LE(width, datum_width);
  CHECK_GE(num, 1);

  if (crop_size) {
    CHECK_EQ(crop_size, height);
    CHECK_EQ(crop_size, width);
  } else {
    CHECK_EQ(datum_height, height);
    CHECK_EQ(datum_width, width);
  }

  bool use_gpu_transform = param_.use_gpu_transform() && Caffe::mode() == Caffe::GPU;
  std::array<unsigned int, 3> rand;
  Fill3Randoms(&rand.front());
  if (use_gpu_transform) {
#ifndef CPU_ONLY
    Dtype *transformed_data_gpu = transformed_blob->mutable_gpu_data();
    TransformGPU(datum, transformed_data_gpu, rand);
    transformed_blob->cpu_data();
#else
  NO_GPU;
#endif
  } else {
    Dtype* transformed_data = transformed_blob->mutable_cpu_data();
    Transform(datum, transformed_data, rand);
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const Datum& datum,
                                       Dtype* transformed_data,
                                       NormalizedBBox* crop_bbox,
                                       bool* do_mirror) {
  const string& data = datum.data();
  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();

  const int crop_size = param_.crop_size();
  const Dtype scale = param_.scale();
  *do_mirror = param_.mirror() && Rand(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_uint8 = data.size() > 0;
  const bool has_mean_values = mean_values_.size() > 0;

  CHECK_GT(datum_channels, 0);
  CHECK_GE(datum_height, crop_size);
  CHECK_GE(datum_width, crop_size);

  float* mean = NULL;
  if (has_mean_file) {
    CHECK_EQ(datum_channels, data_mean_.channels());
    CHECK_EQ(datum_height, data_mean_.height());
    CHECK_EQ(datum_width, data_mean_.width());
    mean = data_mean_.mutable_cpu_data();
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == datum_channels) <<
     "Specify either 1 mean_value or as many as channels: " << datum_channels;
    if (datum_channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < datum_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }

  int height = datum_height;
  int width = datum_width;

  int h_off = 0;
  int w_off = 0;
  if (crop_size) {
    height = crop_size;
    width = crop_size;
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = Rand(datum_height - crop_size + 1);
      w_off = Rand(datum_width - crop_size + 1);
    } else {
      h_off = (datum_height - crop_size) / 2;
      w_off = (datum_width - crop_size) / 2;
    }
  }

  // Return the normalized crop bbox.
  crop_bbox->set_xmin(Dtype(w_off) / datum_width);
  crop_bbox->set_ymin(Dtype(h_off) / datum_height);
  crop_bbox->set_xmax(Dtype(w_off + width) / datum_width);
  crop_bbox->set_ymax(Dtype(h_off + height) / datum_height);

  Dtype datum_element;
  int top_index, data_index;
  for (int c = 0; c < datum_channels; ++c) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        data_index = (c * datum_height + h_off + h) * datum_width + w_off + w;
        if (*do_mirror) {
          top_index = (c * height + h) * width + (width - 1 - w);
        } else {
          top_index = (c * height + h) * width + w;
        }
        if (has_uint8) {
          datum_element =
            static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
        } else {
          datum_element = datum.float_data(data_index);
        }
        if (has_mean_file) {
          transformed_data[top_index] =
            (datum_element - mean[data_index]) * scale;
        } else {
          if (has_mean_values) {
            transformed_data[top_index] =
              (datum_element - mean_values_[c]) * scale;
          } else {
            transformed_data[top_index] = datum_element * scale;
          }
        }
      }
    }
  }
}


/*
template<typename Dtype>
void DataTransformer<Dtype>::Transform(const Datum& datum,
                                       Dtype* transformed_data) {
  NormalizedBBox crop_bbox;
  bool do_mirror;
  Transform(datum, transformed_data, &crop_bbox, &do_mirror);
}
*/
template<typename Dtype>
void DataTransformer<Dtype>::Transform(const Datum& datum,
                                       TBlob<Dtype>* transformed_blob,
                                       NormalizedBBox* crop_bbox,
                                       bool* do_mirror) {
  // If datum is encoded, decoded and transform the cv::image.
  if (datum.encoded()) {
#ifdef USE_OPENCV
    CHECK(!(param_.force_color() && param_.force_gray()))
        << "cannot set both force_color and force_gray";
    cv::Mat cv_img;
    if (param_.force_color() || param_.force_gray()) {
    // If force_color then decode in color otherwise decode in gray.
      cv_img = DecodeDatumToCVMat(datum, param_.force_color());
    } else {
      cv_img = DecodeDatumToCVMatNative(datum);
    }
    // Transform the cv::image into blob.
    return Transform(cv_img, transformed_blob, crop_bbox, do_mirror);
#else
    LOG(FATAL) << "Encoded datum requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  } else {
    if (param_.force_color() || param_.force_gray()) {
      LOG(ERROR) << "force_color and force_gray only for encoded datum";
    }
  }

  const int crop_size = param_.crop_size();
  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();

  // Check dimensions.
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int num = transformed_blob->num();

  CHECK_EQ(channels, datum_channels);
  CHECK_LE(height, datum_height);
  CHECK_LE(width, datum_width);
  CHECK_GE(num, 1);

  if (crop_size) {
    CHECK_EQ(crop_size, height);
    CHECK_EQ(crop_size, width);
  } else {
    CHECK_EQ(datum_height, height);
    CHECK_EQ(datum_width, width);
  }

  Dtype* transformed_data = transformed_blob->mutable_cpu_data();
  Transform(datum, transformed_data, crop_bbox, do_mirror);
}


/*
template<typename Dtype>
void DataTransformer<Dtype>::Transform(const Datum& datum,
                                       TBlob<Dtype>* transformed_blob) {
  NormalizedBBox crop_bbox;
  bool do_mirror;
  Transform(datum, transformed_blob, &crop_bbox, &do_mirror);
}
*/
template<typename Dtype>
void DataTransformer<Dtype>::Transform(const vector<Datum>& datum_vector,
    TBlob<Dtype> *transformed_blob) {
  const int datum_num = datum_vector.size();
  const int num = transformed_blob->num();
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();

  CHECK_GT(datum_num, 0) << "There is no datum to add";
  CHECK_LE(datum_num, num)
    << "The size of datum_vector must be no greater than transformed_blob->num()";
  TBlob<Dtype> uni_blob(1, channels, height, width);
  for (int item_id = 0; item_id < datum_num; ++item_id) {
    int offset = transformed_blob->offset(item_id);
    uni_blob.set_cpu_data(transformed_blob->mutable_cpu_data() + offset);
    Transform(datum_vector[item_id], &uni_blob);
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(
    const AnnotatedDatum& anno_datum, TBlob<Dtype>* transformed_blob,
    RepeatedPtrField<AnnotationGroup>* transformed_anno_group_all,
    bool* do_mirror) {
  // Transform datum.
  const Datum& datum = anno_datum.datum();
  NormalizedBBox crop_bbox;
  Transform(datum, transformed_blob, &crop_bbox, do_mirror);

  // Transform annotation.
  const bool do_resize = true;
  TransformAnnotation(anno_datum, do_resize, crop_bbox, *do_mirror,
                      transformed_anno_group_all);
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(
    const AnnotatedDatum& anno_datum, TBlob<Dtype>* transformed_blob,
    RepeatedPtrField<AnnotationGroup>* transformed_anno_group_all) {
  bool do_mirror;
  Transform(anno_datum, transformed_blob, transformed_anno_group_all,
            &do_mirror);
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(
    const AnnotatedDatum& anno_datum, TBlob<Dtype>* transformed_blob,
    vector<AnnotationGroup>* transformed_anno_vec, bool* do_mirror) {
  RepeatedPtrField<AnnotationGroup> transformed_anno_group_all;
  Transform(anno_datum, transformed_blob, &transformed_anno_group_all,
            do_mirror);
  for (int g = 0; g < transformed_anno_group_all.size(); ++g) {
    transformed_anno_vec->push_back(transformed_anno_group_all.Get(g));
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(
    const AnnotatedDatum& anno_datum, TBlob<Dtype>* transformed_blob,
    vector<AnnotationGroup>* transformed_anno_vec) {
  bool do_mirror;
  Transform(anno_datum, transformed_blob, transformed_anno_vec, &do_mirror);
}

template<typename Dtype>
void DataTransformer<Dtype>::TransformAnnotation(
    const AnnotatedDatum& anno_datum, const bool do_resize,
    const NormalizedBBox& crop_bbox, const bool do_mirror,
    RepeatedPtrField<AnnotationGroup>* transformed_anno_group_all) {
  const int img_height = anno_datum.datum().height();
  const int img_width = anno_datum.datum().width();
  if (anno_datum.type() == AnnotatedDatum_AnnotationType_BBOX) {
    // Go through each AnnotationGroup.
    for (int g = 0; g < anno_datum.annotation_group_size(); ++g) {
      const AnnotationGroup& anno_group = anno_datum.annotation_group(g);
      AnnotationGroup transformed_anno_group;
      // Go through each Annotation.
      bool has_valid_annotation = false;
      for (int a = 0; a < anno_group.annotation_size(); ++a) {
        const Annotation& anno = anno_group.annotation(a);
        const NormalizedBBox& bbox = anno.bbox();
        // Adjust bounding box annotation.
        NormalizedBBox resize_bbox = bbox;
        if (do_resize && param_.has_resize_param()) {
          CHECK_GT(img_height, 0);
          CHECK_GT(img_width, 0);
          UpdateBBoxByResizePolicy(param_.resize_param(), img_width, img_height,
                                   &resize_bbox);
        }
        if (param_.has_emit_constraint() &&
            !MeetEmitConstraint(crop_bbox, resize_bbox,
                                param_.emit_constraint())) {
          continue;
        }
        NormalizedBBox proj_bbox;
        if (ProjectBBox(crop_bbox, resize_bbox, &proj_bbox)) {
          has_valid_annotation = true;
          Annotation* transformed_anno =
              transformed_anno_group.add_annotation();
          transformed_anno->set_instance_id(anno.instance_id());
          NormalizedBBox* transformed_bbox = transformed_anno->mutable_bbox();
          transformed_bbox->CopyFrom(proj_bbox);
          if (do_mirror) {
            Dtype temp = transformed_bbox->xmin();
            transformed_bbox->set_xmin(1 - transformed_bbox->xmax());
            transformed_bbox->set_xmax(1 - temp);
          }
          if (do_resize && param_.has_resize_param()) {
            ExtrapolateBBox(param_.resize_param(), img_height, img_width,
                crop_bbox, transformed_bbox);
          }
        }
      }
      // Save for output.
      if (has_valid_annotation) {
        transformed_anno_group.set_group_label(anno_group.group_label());
        transformed_anno_group_all->Add()->CopyFrom(transformed_anno_group);
      }
    }
  } else {
    LOG(FATAL) << "Unknown annotation type.";
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::CropImage(const Datum& datum,
                                       const NormalizedBBox& bbox,
                                       Datum* crop_datum) {
  // If datum is encoded, decode and crop the cv::image.
  if (datum.encoded()) {
#ifdef USE_OPENCV
    CHECK(!(param_.force_color() && param_.force_gray()))
        << "cannot set both force_color and force_gray";
    cv::Mat cv_img;
    if (param_.force_color() || param_.force_gray()) {
      // If force_color then decode in color otherwise decode in gray.
      cv_img = DecodeDatumToCVMat(datum, param_.force_color());
    } else {
      cv_img = DecodeDatumToCVMatNative(datum);
    }
    // Crop the image.
    cv::Mat crop_img;
    CropImage(cv_img, bbox, &crop_img);
    // Save the image into datum.
    EncodeCVMatToDatum(crop_img, "jpg", crop_datum);
    crop_datum->set_label(datum.label());
    return;
#else
    LOG(FATAL) << "Encoded datum requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  } else {
    if (param_.force_color() || param_.force_gray()) {
      LOG(ERROR) << "force_color and force_gray only for encoded datum";
    }
  }

  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();

  // Get the bbox dimension.
  NormalizedBBox clipped_bbox;
  ClipBBox(bbox, &clipped_bbox);
  NormalizedBBox scaled_bbox;
  ScaleBBox(clipped_bbox, datum_height, datum_width, &scaled_bbox);
  const int w_off = static_cast<int>(scaled_bbox.xmin());
  const int h_off = static_cast<int>(scaled_bbox.ymin());
  const int width = static_cast<int>(scaled_bbox.xmax() - scaled_bbox.xmin());
  const int height = static_cast<int>(scaled_bbox.ymax() - scaled_bbox.ymin());

  // Crop the image using bbox.
  crop_datum->set_channels(datum_channels);
  crop_datum->set_height(height);
  crop_datum->set_width(width);
  crop_datum->set_label(datum.label());
  crop_datum->clear_data();
  crop_datum->clear_float_data();
  crop_datum->set_encoded(false);
  const int crop_datum_size = datum_channels * height * width;
  const std::string& datum_buffer = datum.data();
  std::string buffer(crop_datum_size, ' ');
  for (int h = h_off; h < h_off + height; ++h) {
    for (int w = w_off; w < w_off + width; ++w) {
      for (int c = 0; c < datum_channels; ++c) {
        int datum_index = (c * datum_height + h) * datum_width + w;
        int crop_datum_index = (c * height + h - h_off) * width + w - w_off;
        buffer[crop_datum_index] = datum_buffer[datum_index];
      }
    }
  }
  crop_datum->set_data(buffer);
}

template<typename Dtype>
void DataTransformer<Dtype>::CropImage(const AnnotatedDatum& anno_datum,
                                       const NormalizedBBox& bbox,
                                       AnnotatedDatum* cropped_anno_datum) {
  // Crop the datum.
  CropImage(anno_datum.datum(), bbox, cropped_anno_datum->mutable_datum());
  cropped_anno_datum->set_type(anno_datum.type());

  // Transform the annotation according to crop_bbox.
  const bool do_resize = false;
  const bool do_mirror = false;
  NormalizedBBox crop_bbox;
  ClipBBox(bbox, &crop_bbox);
  TransformAnnotation(anno_datum, do_resize, crop_bbox, do_mirror,
                      cropped_anno_datum->mutable_annotation_group());
}

template<typename Dtype>
void DataTransformer<Dtype>::ExpandImage(const Datum& datum,
                                         const float expand_ratio,
                                         NormalizedBBox* expand_bbox,
                                         Datum* expand_datum) {
  // If datum is encoded, decode and crop the cv::image.
  if (datum.encoded()) {
#ifdef USE_OPENCV
    CHECK(!(param_.force_color() && param_.force_gray()))
        << "cannot set both force_color and force_gray";
    cv::Mat cv_img;
    if (param_.force_color() || param_.force_gray()) {
      // If force_color then decode in color otherwise decode in gray.
      cv_img = DecodeDatumToCVMat(datum, param_.force_color());
    } else {
      cv_img = DecodeDatumToCVMatNative(datum);
    }
    // Expand the image.
    cv::Mat expand_img;
    ExpandImage(cv_img, expand_ratio, expand_bbox, &expand_img);
    // Save the image into datum.
    EncodeCVMatToDatum(expand_img, "jpg", expand_datum);
    expand_datum->set_label(datum.label());
    return;
#else
    LOG(FATAL) << "Encoded datum requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  } else {
    if (param_.force_color() || param_.force_gray()) {
      LOG(ERROR) << "force_color and force_gray only for encoded datum";
    }
  }

  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();

  // Get the bbox dimension.
  int height = static_cast<int>(datum_height * expand_ratio);
  int width = static_cast<int>(datum_width * expand_ratio);
  float h_off, w_off;
  caffe_rng_uniform(1, 0.f, static_cast<float>(height - datum_height), &h_off);
  caffe_rng_uniform(1, 0.f, static_cast<float>(width - datum_width), &w_off);
  h_off = floor(h_off);
  w_off = floor(w_off);
  expand_bbox->set_xmin(-w_off/datum_width);
  expand_bbox->set_ymin(-h_off/datum_height);
  expand_bbox->set_xmax((width - w_off)/datum_width);
  expand_bbox->set_ymax((height - h_off)/datum_height);

  // Crop the image using bbox.
  expand_datum->set_channels(datum_channels);
  expand_datum->set_height(height);
  expand_datum->set_width(width);
  expand_datum->set_label(datum.label());
  expand_datum->clear_data();
  expand_datum->clear_float_data();
  expand_datum->set_encoded(false);
  const int expand_datum_size = datum_channels * height * width;
  const std::string& datum_buffer = datum.data();
  std::string buffer(expand_datum_size, ' ');
  for (int h = h_off; h < h_off + datum_height; ++h) {
    for (int w = w_off; w < w_off + datum_width; ++w) {
      for (int c = 0; c < datum_channels; ++c) {
        int datum_index =
            (c * datum_height + h - h_off) * datum_width + w - w_off;
        int expand_datum_index = (c * height + h) * width + w;
        buffer[expand_datum_index] = datum_buffer[datum_index];
      }
    }
  }
  expand_datum->set_data(buffer);
}

template<typename Dtype>
void DataTransformer<Dtype>::ExpandImage(const AnnotatedDatum& anno_datum,
                                         AnnotatedDatum* expanded_anno_datum) {
  if (!param_.has_expand_param()) {
    expanded_anno_datum->CopyFrom(anno_datum);
    return;
  }
  const ExpansionParameter& expand_param = param_.expand_param();
  const float expand_prob = expand_param.prob();
  float prob;
  caffe_rng_uniform(1, 0.f, 1.f, &prob);
  if (prob > expand_prob) {
    expanded_anno_datum->CopyFrom(anno_datum);
    return;
  }
  const float max_expand_ratio = expand_param.max_expand_ratio();
  if (fabs(max_expand_ratio - 1.) < 1e-2) {
    expanded_anno_datum->CopyFrom(anno_datum);
    return;
  }
  float expand_ratio;
  caffe_rng_uniform(1, 1.f, max_expand_ratio, &expand_ratio);
  // Expand the datum.
  NormalizedBBox expand_bbox;
  ExpandImage(anno_datum.datum(), expand_ratio, &expand_bbox,
              expanded_anno_datum->mutable_datum());
  expanded_anno_datum->set_type(anno_datum.type());

  // Transform the annotation according to crop_bbox.
  const bool do_resize = false;
  const bool do_mirror = false;
  TransformAnnotation(anno_datum, do_resize, expand_bbox, do_mirror,
                      expanded_anno_datum->mutable_annotation_group());
}

template<typename Dtype>
void DataTransformer<Dtype>::DistortImage(const Datum& datum,
                                          Datum* distort_datum) {
  if (!param_.has_distort_param()) {
    distort_datum->CopyFrom(datum);
    return;
  }
  // If datum is encoded, decode and crop the cv::image.
  if (datum.encoded()) {
#ifdef USE_OPENCV
    CHECK(!(param_.force_color() && param_.force_gray()))
        << "cannot set both force_color and force_gray";
    cv::Mat cv_img;
    if (param_.force_color() || param_.force_gray()) {
      // If force_color then decode in color otherwise decode in gray.
      cv_img = DecodeDatumToCVMat(datum, param_.force_color());
    } else {
      cv_img = DecodeDatumToCVMatNative(datum);
    }
    // Distort the image.
    cv::Mat distort_img = ApplyDistort(cv_img, param_.distort_param());
    // Save the image into datum.
    EncodeCVMatToDatum(distort_img, "jpg", distort_datum);
    distort_datum->set_label(datum.label());
    return;
#else
    LOG(FATAL) << "Encoded datum requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  } else {
    LOG(ERROR) << "Only support encoded datum now";
  }
}
#ifdef USE_OPENCV

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const vector<cv::Mat>& mat_vector,
    TBlob<Dtype> *transformed_blob) {
  const int mat_num = mat_vector.size();
  const int num = transformed_blob->num();
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();

  CHECK_GT(mat_num, 0) << "There is no MAT to add";
  CHECK_EQ(mat_num, num) <<
                         "The size of mat_vector must be equals to transformed_blob->num()";
  TBlob<Dtype> uni_blob(1, channels, height, width);
  for (int item_id = 0; item_id < mat_num; ++item_id) {
    int offset = transformed_blob->offset(item_id);
    uni_blob.set_cpu_data(transformed_blob->mutable_cpu_data() + offset);
    Transform(mat_vector[item_id], &uni_blob);
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const cv::Mat& cv_img,
    TBlob<Dtype> *transformed_blob) {
  const int crop_size = param_.crop_size();
  const int img_channels = cv_img.channels();
  const int img_height = cv_img.rows;
  const int img_width = cv_img.cols;

  // Check dimensions.
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int num = transformed_blob->num();

  CHECK_EQ(channels, img_channels);
  CHECK_LE(height, img_height);
  CHECK_LE(width, img_width);
  CHECK_GE(num, 1);

  CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";

  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && Rand(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_values = mean_values_.size() > 0;

  CHECK_GT(img_channels, 0);
  CHECK_GE(img_height, crop_size);
  CHECK_GE(img_width, crop_size);

  float* mean = NULL;
  if (has_mean_file) {
    CHECK_EQ(img_channels, data_mean_.channels());
    CHECK_EQ(img_height, data_mean_.height());
    CHECK_EQ(img_width, data_mean_.width());
    mean = data_mean_.mutable_cpu_data();
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == img_channels)
        << "Specify either 1 mean_value or as many as channels: " << img_channels;
    if (img_channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < img_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }

  int h_off = 0;
  int w_off = 0;
  cv::Mat cv_cropped_img = cv_img;
  if (crop_size) {
    CHECK_EQ(crop_size, height);
    CHECK_EQ(crop_size, width);
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = Rand(img_height - crop_size + 1);
      w_off = Rand(img_width - crop_size + 1);
    } else {
      h_off = (img_height - crop_size) / 2;
      w_off = (img_width - crop_size) / 2;
    }
    cv::Rect roi(w_off, h_off, crop_size, crop_size);
    cv_cropped_img = cv_img(roi);
  } else {
    CHECK_EQ(img_height, height);
    CHECK_EQ(img_width, width);
  }

  CHECK(cv_cropped_img.data);

  Dtype *transformed_data = transformed_blob->mutable_cpu_data();
  int top_index;
  for (int h = 0; h < height; ++h) {
    const uchar *ptr = cv_cropped_img.ptr<uchar>(h);
    int img_index = 0;
    for (int w = 0; w < width; ++w) {
      for (int c = 0; c < img_channels; ++c) {
        if (do_mirror) {
          top_index = (c * height + h) * width + (width - 1 - w);
        } else {
          top_index = (c * height + h) * width + w;
        }
        // int top_index = (c * height + h) * width + w;
        Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
        if (has_mean_file) {
          int mean_index = (c * img_height + h_off + h) * img_width + w_off + w;
          transformed_data[top_index] =
              (pixel - mean[mean_index]) * scale;
        } else {
          if (has_mean_values) {
            transformed_data[top_index] =
                (pixel - mean_values_[c]) * scale;
          } else {
            transformed_data[top_index] = pixel * scale;
          }
        }
      }
    }
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::TransformPtr(const cv::Mat& cv_img,
    Dtype *transformed_ptr, const std::array<unsigned int, 3>& rand) {
  const int crop_size = param_.crop_size();
  const int img_channels = cv_img.channels();
  const int img_height = cv_img.rows;
  const int img_width = cv_img.cols;

  CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";

  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && (rand[0] % 2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_values = mean_values_.size() > 0;

  CHECK_GT(img_channels, 0);
  CHECK_GE(img_height, crop_size);
  CHECK_GE(img_width, crop_size);

  const float* mean = NULL;
  if (has_mean_file) {
    CHECK_EQ(img_channels, data_mean_.channels());
    CHECK_EQ(img_height, data_mean_.height());
    CHECK_EQ(img_width, data_mean_.width());
    mean = data_mean_.cpu_data();
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == img_channels)
        << "Specify either 1 mean_value or as many as channels: " << img_channels;
    if (img_channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < img_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }

  int h_off = 0;
  int w_off = 0;
  int height = img_height;
  int width = img_width;
  cv::Mat cv_cropped_img = cv_img;
  if (crop_size) {
    height = width = crop_size;
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = rand[1] % (img_height - crop_size + 1);
      w_off = rand[2] % (img_width - crop_size + 1);
    } else {
      h_off = (img_height - crop_size) / 2;
      w_off = (img_width - crop_size) / 2;
    }
    cv::Rect roi(w_off, h_off, crop_size, crop_size);
    cv_cropped_img = cv_img(roi);
  }

  CHECK(cv_cropped_img.data);

  int top_index;
  for (int h = 0; h < height; ++h) {
    const uchar *ptr = cv_cropped_img.ptr<uchar>(h);
    int img_index = 0;
    for (int w = 0; w < width; ++w) {
      for (int c = 0; c < img_channels; ++c) {
        if (do_mirror) {
          top_index = (c * height + h) * width + (width - 1 - w);
        } else {
          top_index = (c * height + h) * width + w;
        }
        // int top_index = (c * height + h) * width + w;
        Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
        if (has_mean_file) {
          int mean_index = (c * img_height + h_off + h) * img_width + w_off + w;
          transformed_ptr[top_index] =
              (pixel - mean[mean_index]) * scale;
        } else {
          if (has_mean_values) {
            transformed_ptr[top_index] =
                (pixel - mean_values_[c]) * scale;
          } else {
            transformed_ptr[top_index] = pixel * scale;
          }
        }
      }
    }
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const cv::Mat& cv_img,
                                       TBlob<Dtype>* transformed_blob,
                                       NormalizedBBox* crop_bbox,
                                       bool* do_mirror) {
  // Check dimensions.
  const int img_channels = cv_img.channels();
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int num = transformed_blob->num();

  CHECK_GT(img_channels, 0);
  CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";
  CHECK_EQ(channels, img_channels);
  CHECK_GE(num, 1);

  const int crop_size = param_.crop_size();
  const Dtype scale = param_.scale();
  *do_mirror = param_.mirror() && Rand(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_values = mean_values_.size() > 0;

  float* mean = NULL;
  if (has_mean_file) {
    CHECK_EQ(img_channels, data_mean_.channels());
    mean = data_mean_.mutable_cpu_data();
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == img_channels) <<
        "Specify either 1 mean_value or as many as channels: " << img_channels;
    if (img_channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < img_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }

  int crop_h = param_.crop_h();
  int crop_w = param_.crop_w();
  if (crop_size) {
    crop_h = crop_size;
    crop_w = crop_size;
  }

  cv::Mat cv_resized_image, cv_noised_image, cv_cropped_image;
  if (param_.has_resize_param()) {
    cv_resized_image = ApplyResize(cv_img, param_.resize_param());
  } else {
    cv_resized_image = cv_img;
  }
  if (param_.has_noise_param()) {
    cv_noised_image = ApplyNoise(cv_resized_image, param_.noise_param());
  } else {
    cv_noised_image = cv_resized_image;
  }
  int img_height = cv_noised_image.rows;
  int img_width = cv_noised_image.cols;
  CHECK_GE(img_height, crop_h);
  CHECK_GE(img_width, crop_w);

  int h_off = 0;
  int w_off = 0;
  if ((crop_h > 0) && (crop_w > 0)) {
    CHECK_EQ(crop_h, height);
    CHECK_EQ(crop_w, width);
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = Rand(img_height - crop_h + 1);
      w_off = Rand(img_width - crop_w + 1);
    } else {
      h_off = (img_height - crop_h) / 2;
      w_off = (img_width - crop_w) / 2;
    }
    cv::Rect roi(w_off, h_off, crop_w, crop_h);
    cv_cropped_image = cv_noised_image(roi);
  } else {
    cv_cropped_image = cv_noised_image;
  }

  // Return the normalized crop bbox.
  crop_bbox->set_xmin(Dtype(w_off) / img_width);
  crop_bbox->set_ymin(Dtype(h_off) / img_height);
  crop_bbox->set_xmax(Dtype(w_off + width) / img_width);
  crop_bbox->set_ymax(Dtype(h_off + height) / img_height);

  if (has_mean_file) {
    CHECK_EQ(cv_cropped_image.rows, data_mean_.height());
    CHECK_EQ(cv_cropped_image.cols, data_mean_.width());
  }
  CHECK(cv_cropped_image.data);

  Dtype* transformed_data = transformed_blob->mutable_cpu_data();
  int top_index;
  for (int h = 0; h < height; ++h) {
    const uchar* ptr = cv_cropped_image.ptr<uchar>(h);
    int img_index = 0;
    int h_idx = h;
    for (int w = 0; w < width; ++w) {
      int w_idx = w;
      if (*do_mirror) {
        w_idx = (width - 1 - w);
      }
      int h_idx_real = h_idx;
      int w_idx_real = w_idx;
      for (int c = 0; c < img_channels; ++c) {
        top_index = (c * height + h_idx_real) * width + w_idx_real;
        Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
        if (has_mean_file) {
          int mean_index = (c * img_height + h_off + h_idx_real) * img_width
              + w_off + w_idx_real;
          transformed_data[top_index] =
              (pixel - mean[mean_index]) * scale;
        } else {
          if (has_mean_values) {
            transformed_data[top_index] =
                (pixel - mean_values_[c]) * scale;
          } else {
            transformed_data[top_index] = pixel * scale;
          }
        }
      }
    }
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::TransformInv(const Dtype* data, cv::Mat* cv_img,
                                          const int height, const int width,
                                          const int channels) {
  const Dtype scale = param_.scale();
  const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_values = mean_values_.size() > 0;

  float* mean = NULL;
  if (has_mean_file) {
    CHECK_EQ(channels, data_mean_.channels());
    CHECK_EQ(height, data_mean_.height());
    CHECK_EQ(width, data_mean_.width());
    mean = data_mean_.mutable_cpu_data();
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == channels) <<
        "Specify either 1 mean_value or as many as channels: " << channels;
    if (channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }

  const int img_type = channels == 3 ? CV_8UC3 : CV_8UC1;
  cv::Mat orig_img(height, width, img_type, cv::Scalar(0, 0, 0));
  for (int h = 0; h < height; ++h) {
    uchar* ptr = orig_img.ptr<uchar>(h);
    int img_idx = 0;
    for (int w = 0; w < width; ++w) {
      for (int c = 0; c < channels; ++c) {
        int idx = (c * height + h) * width + w;
        if (has_mean_file) {
          ptr[img_idx++] = static_cast<uchar>(data[idx] / scale + mean[idx]);
        } else {
          if (has_mean_values) {
            ptr[img_idx++] =
                static_cast<uchar>(data[idx] / scale + mean_values_[c]);
          } else {
            ptr[img_idx++] = static_cast<uchar>(data[idx] / scale);
          }
        }
      }
    }
  }

  if (param_.has_resize_param()) {
    *cv_img = ApplyResize(orig_img, param_.resize_param());
  } else {
    *cv_img = orig_img;
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::TransformInv(const TBlob<Dtype>* blob,
                                          vector<cv::Mat>* cv_imgs) {
  const int channels = blob->channels();
  const int height = blob->height();
  const int width = blob->width();
  const int num = blob->num();
  CHECK_GE(num, 1);
  const Dtype* image_data = blob->cpu_data();

  for (int i = 0; i < num; ++i) {
    cv::Mat cv_img;
    TransformInv(image_data, &cv_img, height, width, channels);
    cv_imgs->push_back(cv_img);
    image_data += blob->offset(1);
  }
}
/*
template<typename Dtype>
void DataTransformer<Dtype>::Transform(const cv::Mat& cv_img,
                                       TBlob<Dtype>* transformed_blob) {
  NormalizedBBox crop_bbox;
  bool do_mirror;
  Transform(cv_img, transformed_blob, &crop_bbox, &do_mirror);
}
*/
template <typename Dtype>
void DataTransformer<Dtype>::CropImage(const cv::Mat& img,
                                       const NormalizedBBox& bbox,
                                       cv::Mat* crop_img) {
  const int img_height = img.rows;
  const int img_width = img.cols;

  // Get the bbox dimension.
  NormalizedBBox clipped_bbox;
  ClipBBox(bbox, &clipped_bbox);
  NormalizedBBox scaled_bbox;
  ScaleBBox(clipped_bbox, img_height, img_width, &scaled_bbox);

  // Crop the image using bbox.
  int w_off = static_cast<int>(scaled_bbox.xmin());
  int h_off = static_cast<int>(scaled_bbox.ymin());
  int width = static_cast<int>(scaled_bbox.xmax() - scaled_bbox.xmin());
  int height = static_cast<int>(scaled_bbox.ymax() - scaled_bbox.ymin());
  cv::Rect bbox_roi(w_off, h_off, width, height);

  img(bbox_roi).copyTo(*crop_img);
}

template <typename Dtype>
void DataTransformer<Dtype>::ExpandImage(const cv::Mat& img,
                                         const float expand_ratio,
                                         NormalizedBBox* expand_bbox,
                                         cv::Mat* expand_img) {
  const int img_height = img.rows;
  const int img_width = img.cols;
  const int img_channels = img.channels();

  // Get the bbox dimension.
  int height = static_cast<int>(img_height * expand_ratio);
  int width = static_cast<int>(img_width * expand_ratio);
  float h_off, w_off;
  caffe_rng_uniform(1, 0.f, static_cast<float>(height - img_height), &h_off);
  caffe_rng_uniform(1, 0.f, static_cast<float>(width - img_width), &w_off);
  h_off = floor(h_off);
  w_off = floor(w_off);
  expand_bbox->set_xmin(-w_off/img_width);
  expand_bbox->set_ymin(-h_off/img_height);
  expand_bbox->set_xmax((width - w_off)/img_width);
  expand_bbox->set_ymax((height - h_off)/img_height);

  expand_img->create(height, width, img.type());
  expand_img->setTo(cv::Scalar(0));
  const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_values = mean_values_.size() > 0;

  if (has_mean_file) {
    CHECK_EQ(img_channels, data_mean_.channels());
    CHECK_EQ(height, data_mean_.height());
    CHECK_EQ(width, data_mean_.width());
    float* mean = data_mean_.mutable_cpu_data();
    for (int h = 0; h < height; ++h) {
      uchar* ptr = expand_img->ptr<uchar>(h);
      int img_index = 0;
      for (int w = 0; w < width; ++w) {
        for (int c = 0; c < img_channels; ++c) {
          int blob_index = (c * height + h) * width + w;
          ptr[img_index++] = static_cast<char>(mean[blob_index]);
        }
      }
    }
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == img_channels) <<
        "Specify either 1 mean_value or as many as channels: " << img_channels;
    if (img_channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < img_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
    vector<cv::Mat> channels(img_channels);
    cv::split(*expand_img, channels);
    CHECK_EQ(channels.size(), mean_values_.size());
    for (int c = 0; c < img_channels; ++c) {
      channels[c] = mean_values_[c];
    }
    cv::merge(channels, *expand_img);
  }

  cv::Rect bbox_roi(w_off, h_off, img_width, img_height);
  img.copyTo((*expand_img)(bbox_roi));
}

#endif  // USE_OPENCV

template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferDatumShape(const Datum& datum) {
  if (datum.encoded()) {
#ifdef USE_OPENCV
    CHECK(!(param_.force_color() && param_.force_gray()))
    << "cannot set both force_color and force_gray";
    cv::Mat cv_img;
    if (param_.force_color() || param_.force_gray()) {
      // If force_color then decode in color otherwise decode in gray.
      cv_img = DecodeDatumToCVMat(datum, param_.force_color());
    } else {
      cv_img = DecodeDatumToCVMatNative(datum);
    }
    // Infer shape using the cv::image.
    return InferCVMatShape(cv_img);
#else
    LOG(FATAL) << "Encoded datum requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  }
  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();
  vector<int> datum_shape(4);
  datum_shape[0] = 1;
  datum_shape[1] = datum_channels;
  datum_shape[2] = datum_height;
  datum_shape[3] = datum_width;
  return datum_shape;
}

template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferBlobShape(const Datum& datum) {
  if (datum.encoded()) {
#ifdef USE_OPENCV
    CHECK(!(param_.force_color() && param_.force_gray()))
        << "cannot set both force_color and force_gray";
    cv::Mat cv_img;
    if (param_.force_color() || param_.force_gray()) {
    // If force_color then decode in color otherwise decode in gray.
      cv_img = DecodeDatumToCVMat(datum, param_.force_color());
    } else {
      cv_img = DecodeDatumToCVMatNative(datum);
    }
    // InferBlobShape using the cv::image.
    return InferBlobShape(cv_img);
#else
    LOG(FATAL) << "Encoded datum requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  }

  const int crop_size = param_.crop_size();
  int crop_h = param_.crop_h();
  int crop_w = param_.crop_w();
  if (crop_size) {
    crop_h = crop_size;
    crop_w = crop_size;
  }
  const int datum_channels = datum.channels();
  int datum_height = datum.height();
  int datum_width = datum.width();

  // Check dimensions.
  CHECK_GT(datum_channels, 0);
  if (param_.has_resize_param()) {
    InferNewSize(param_.resize_param(), datum_width, datum_height,
                 &datum_width, &datum_height);
  }
  CHECK_GE(datum_height, crop_h);
  CHECK_GE(datum_width, crop_w);

  // Build BlobShape.
  vector<int> shape(4);
  shape[0] = 1;
  shape[1] = datum_channels;
  shape[2] = (crop_h)? crop_h: datum_height;
  shape[3] = (crop_w)? crop_w: datum_width;
  return shape;
}

template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferBlobShape(
    const vector<Datum> & datum_vector) {
  const int num = datum_vector.size();
  CHECK_GT(num, 0) << "There is no datum to in the vector";
  // Use first datum in the vector to InferBlobShape.
  vector<int> shape = InferBlobShape(datum_vector[0]);
  // Adjust num to the size of the vector.
  shape[0] = num;
  return shape;
}

#ifdef USE_OPENCV
template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferBlobShape(const cv::Mat& cv_img) {
  const int crop_size = param_.crop_size();
  int crop_h = param_.crop_h();
  int crop_w = param_.crop_w();
  if (crop_size) {
    crop_h = crop_size;
    crop_w = crop_size;
  }
  const int img_channels = cv_img.channels();
  int img_height = cv_img.rows;
  int img_width = cv_img.cols;
  // Check dimensions.
  CHECK_GT(img_channels, 0);
  if (param_.has_resize_param()) {
    InferNewSize(param_.resize_param(), img_width, img_height,
                 &img_width, &img_height);
  }
  CHECK_GE(img_height, crop_h);
  CHECK_GE(img_width, crop_w);

  // Build BlobShape.
  vector<int> shape(4);
  shape[0] = 1;
  shape[1] = img_channels;
  shape[2] = (crop_h)? crop_h: img_height;
  shape[3] = (crop_w)? crop_w: img_width;
  return shape;
}

template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferCVMatShape(const cv::Mat& cv_img) {
  int img_channels = cv_img.channels();
  int img_height = cv_img.rows;
  int img_width = cv_img.cols;
  vector<int> shape(4);
  shape[0] = 1;
  shape[1] = img_channels;
  shape[2] = img_height;
  shape[3] = img_width;
  
  const int crop_size = param_.crop_size();
  int crop_h = param_.crop_h();
  int crop_w = param_.crop_w();
  if (crop_size) {
    crop_h = crop_size;
    crop_w = crop_size;
  }

  // Check dimensions.
  if (param_.has_resize_param()) {
    InferNewSize(param_.resize_param(), img_width, img_height,
                 &img_width, &img_height);
  }
  CHECK_GE(img_height, crop_h);
  CHECK_GE(img_width, crop_w);

  // Build BlobShape.
  shape[0] = 1;
  shape[1] = img_channels;
  shape[2] = (crop_h)? crop_h: img_height;
  shape[3] = (crop_w)? crop_w: img_width;
    
  return shape;
}

#endif  // USE_OPENCV

template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferBlobShape(const vector<int>& bottom_shape, bool use_gpu) {
  const int crop_size = param_.crop_size();
  CHECK_EQ(bottom_shape.size(), 4);
  CHECK_EQ(bottom_shape[0], 1);
  const int bottom_channels = bottom_shape[1];
  const int bottom_height = bottom_shape[2];
  const int bottom_width = bottom_shape[3];
  // Check dimensions.
  CHECK_GT(bottom_channels, 0);
  CHECK_GE(bottom_height, crop_size);
  CHECK_GE(bottom_width, crop_size);
  // Build BlobShape.
  vector<int> top_shape(4);
  top_shape[0] = 1;
  top_shape[1] = bottom_channels;
  // if using GPU transform, don't crop
  if (use_gpu) {
    top_shape[2] = bottom_height;
    top_shape[3] = bottom_width;
  } else {
    top_shape[2] = (crop_size) ? crop_size : bottom_height;
    top_shape[3] = (crop_size) ? crop_size : bottom_width;
  }
  return top_shape;
}

template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferBlobShape(const Datum& datum, bool use_gpu) {
  return InferBlobShape(InferDatumShape(datum), use_gpu);
}

#ifdef USE_OPENCV

template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferBlobShape(const cv::Mat& cv_img, bool use_gpu) {
  return InferBlobShape(InferCVMatShape(cv_img), use_gpu);
}

#endif  // USE_OPENCV

template<typename Dtype>
void DataTransformer<Dtype>::InitRand() {
  const bool needs_rand = param_.mirror() || (phase_ == TRAIN && param_.crop_size());
  if (needs_rand) {
    // Use random_seed setting for deterministic transformations
    const uint64_t random_seed = param_.random_seed() >= 0 ?
        static_cast<uint64_t>(param_.random_seed()) : Caffe::next_seed();
    rng_.reset(new Caffe::RNG(random_seed));
  } else {
    rng_.reset();
  }
}

template<typename Dtype>
unsigned int DataTransformer<Dtype>::Rand() const {
  CHECK(rng_);
  caffe::rng_t *rng =
      static_cast<caffe::rng_t *>(rng_->generator());
  // this doesn't actually produce a uniform distribution
  return static_cast<unsigned int>((*rng)());
}

INSTANTIATE_CLASS(DataTransformer);

}  // namespace caffe
