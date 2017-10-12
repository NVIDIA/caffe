// Make sure we include Python.h before any system header
// to avoid _POSIX_C_SOURCE redefinition
#ifdef WITH_PYTHON_LAYER
#include <boost/python.hpp>
#endif
#include <string>

#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/layers/batch_norm_layer.hpp"
#include "caffe/layers/conv_layer.hpp"
#include "caffe/layers/lrn_layer.hpp"
#include "caffe/layers/pooling_layer.hpp"
#include "caffe/layers/relu_layer.hpp"
#include "caffe/layers/sigmoid_layer.hpp"
#include "caffe/layers/softmax_layer.hpp"
#include "caffe/layers/tanh_layer.hpp"
#include "caffe/layers/dropout_layer.hpp"
#include "caffe/layers/detectnet_transform_layer.hpp"
#include "caffe/layers/data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

#ifdef USE_CUDNN
#include "caffe/layers/cudnn_batch_norm_layer.hpp"
#include "caffe/layers/cudnn_conv_layer.hpp"
#include "caffe/layers/cudnn_lcn_layer.hpp"
#include "caffe/layers/cudnn_lrn_layer.hpp"
#include "caffe/layers/cudnn_pooling_layer.hpp"
#include "caffe/layers/cudnn_relu_layer.hpp"
#include "caffe/layers/cudnn_sigmoid_layer.hpp"
#include "caffe/layers/cudnn_softmax_layer.hpp"
#include "caffe/layers/cudnn_tanh_layer.hpp"
#include "caffe/layers/cudnn_dropout_layer.hpp"
#endif

#ifdef WITH_PYTHON_LAYER
#include "caffe/layers/python_layer.hpp"

__attribute__((constructor)) void loadso() {
//  PyEval_InitThreads();
  Py_Initialize();
}

__attribute__((destructor))  void unloadso() {
}
#endif

#pragma GCC diagnostic ignored "-Wreturn-type"

namespace caffe {

// Get convolution layer according to engine.
shared_ptr<LayerBase> GetConvolutionLayer(const LayerParameter& param,
    Type ftype, Type btype) {
  ConvolutionParameter conv_param = param.convolution_param();
  ConvolutionParameter_Engine engine = conv_param.engine();
#ifdef USE_CUDNN
  bool use_dilation = false;
  for (int i = 0; i < conv_param.dilation_size(); ++i) {
    if (conv_param.dilation(i) > 1) {
      use_dilation = true;
    }
  }
#endif
  if (engine == ConvolutionParameter_Engine_DEFAULT) {
    engine = ConvolutionParameter_Engine_CAFFE;
#ifdef USE_CUDNN
    if (!use_dilation && Caffe::mode() == Caffe::GPU) {
      engine = ConvolutionParameter_Engine_CUDNN;
    }
#endif
  }
  if (engine == ConvolutionParameter_Engine_CAFFE) {
    return CreateLayerBase<ConvolutionLayer>(param, ftype, btype);
#ifdef USE_CUDNN
  } else if (engine == ConvolutionParameter_Engine_CUDNN) {
    if (use_dilation) {
      LOG(FATAL) << "CuDNN doesn't support the dilated convolution at Layer "
                 << param.name();
    }
    return CreateLayerBase<CuDNNConvolutionLayer>(param, ftype, btype);
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
  }
}

REGISTER_LAYER_CREATOR(Convolution, GetConvolutionLayer);

// Get BN layer according to engine.
shared_ptr<LayerBase> GetBatchNormLayer(const LayerParameter& param,
    Type ftype, Type btype) {
  BatchNormParameter_Engine engine = param.batch_norm_param().engine();
  if (engine == BatchNormParameter_Engine_DEFAULT) {
    engine = BatchNormParameter_Engine_CAFFE;
#ifdef USE_CUDNN
    if (Caffe::mode() == Caffe::GPU)
      engine = BatchNormParameter_Engine_CUDNN;
#endif
  }
  if (engine == BatchNormParameter_Engine_CAFFE) {
    return CreateLayerBase<BatchNormLayer>(param, ftype, btype);
#ifdef USE_CUDNN
  } else if (engine == BatchNormParameter_Engine_CUDNN) {
    return CreateLayerBase<CuDNNBatchNormLayer>(param, ftype, btype);
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
  }
}

REGISTER_LAYER_CREATOR(BatchNorm, GetBatchNormLayer);

// Get pooling layer according to engine.
shared_ptr<LayerBase> GetPoolingLayer(const LayerParameter& param,
    Type ftype, Type btype) {
  PoolingParameter_Engine engine = param.pooling_param().engine();
  if (engine == PoolingParameter_Engine_DEFAULT) {
    engine = PoolingParameter_Engine_CAFFE;
#ifdef USE_CUDNN
    if (Caffe::mode() == Caffe::GPU)
      engine = PoolingParameter_Engine_CUDNN;
#endif
  }
  if (engine == PoolingParameter_Engine_CAFFE) {
    return CreateLayerBase<PoolingLayer>(param, ftype, btype);
#ifdef USE_CUDNN
  } else if (engine == PoolingParameter_Engine_CUDNN) {
    if (param.top_size() > 1) {
      LOG(INFO) << "cuDNN does not support multiple tops. "
                << "Using Caffe's own pooling layer.";
      return CreateLayerBase<PoolingLayer>(param, ftype, btype);
    }
    return CreateLayerBase<CuDNNPoolingLayer>(param, ftype, btype);
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
  }
}

REGISTER_LAYER_CREATOR(Pooling, GetPoolingLayer);

// Get LRN layer according to engine
shared_ptr<LayerBase> GetLRNLayer(const LayerParameter& param,
    Type ftype, Type btype) {
  LRNParameter_Engine engine = param.lrn_param().engine();

  if (engine == LRNParameter_Engine_DEFAULT) {
    engine = LRNParameter_Engine_CAFFE;
#ifdef USE_CUDNN
    if (Caffe::mode() == Caffe::GPU)
      engine = LRNParameter_Engine_CUDNN;
#endif
  }

  if (engine == LRNParameter_Engine_CAFFE) {
    return CreateLayerBase<LRNLayer>(param, ftype, btype);
#ifdef USE_CUDNN
  } else if (engine == LRNParameter_Engine_CUDNN) {
    LRNParameter lrn_param = param.lrn_param();

    if (lrn_param.norm_region() ==LRNParameter_NormRegion_WITHIN_CHANNEL) {
      return CreateLayerBase<CuDNNLCNLayer>(param, ftype, btype);
    } else {
      // local size is too big to be handled through cuDNN
      if (param.lrn_param().local_size() > CUDNN_LRN_MAX_N) {
        return CreateLayerBase<LRNLayer>(param, ftype, btype);
      } else {
        return CreateLayerBase<CuDNNLRNLayer>(param, ftype, btype);
      }
    }
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
  }
}

REGISTER_LAYER_CREATOR(LRN, GetLRNLayer);

// Get relu layer according to engine.
shared_ptr<LayerBase> GetReLULayer(const LayerParameter& param,
    Type ftype, Type btype) {
  ReLUParameter_Engine engine = param.relu_param().engine();
  if (engine == ReLUParameter_Engine_DEFAULT) {
    engine = ReLUParameter_Engine_CAFFE;
#ifdef USE_CUDNN
    if (Caffe::mode() == Caffe::GPU)
      engine = ReLUParameter_Engine_CUDNN;
#endif
  }
  if (engine == ReLUParameter_Engine_CAFFE) {
    return CreateLayerBase<ReLULayer>(param, ftype, btype);
#ifdef USE_CUDNN
  } else if (engine == ReLUParameter_Engine_CUDNN) {
    return CreateLayerBase<CuDNNReLULayer>(param, ftype, btype);
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
  }
}

REGISTER_LAYER_CREATOR(ReLU, GetReLULayer);

// Get sigmoid layer according to engine.
shared_ptr<LayerBase> GetSigmoidLayer(const LayerParameter& param,
    Type ftype, Type btype) {
  SigmoidParameter_Engine engine = param.sigmoid_param().engine();
  if (engine == SigmoidParameter_Engine_DEFAULT) {
    engine = SigmoidParameter_Engine_CAFFE;
#ifdef USE_CUDNN
    if (Caffe::mode() == Caffe::GPU)
      engine = SigmoidParameter_Engine_CUDNN;
#endif
  }
  if (engine == SigmoidParameter_Engine_CAFFE) {
    return CreateLayerBase<SigmoidLayer>(param, ftype, btype);
#ifdef USE_CUDNN
  } else if (engine == SigmoidParameter_Engine_CUDNN) {
    return CreateLayerBase<CuDNNSigmoidLayer>(param, ftype, btype);
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
  }
}

REGISTER_LAYER_CREATOR(Sigmoid, GetSigmoidLayer);

// Get softmax layer according to engine.
shared_ptr<LayerBase> GetSoftmaxLayer(const LayerParameter& param,
    Type ftype, Type btype) {
  LayerParameter lparam(param);
  SoftmaxParameter_Engine engine = lparam.softmax_param().engine();
  if (engine == SoftmaxParameter_Engine_DEFAULT) {
    engine = SoftmaxParameter_Engine_CAFFE;
#ifdef USE_CUDNN
    if (Caffe::mode() == Caffe::GPU)
      engine = SoftmaxParameter_Engine_CUDNN;
#endif
  }
  if (engine == SoftmaxParameter_Engine_CAFFE) {
    return CreateLayerBase<SoftmaxLayer>(lparam, ftype, btype);
#ifdef USE_CUDNN
  } else if (engine == SoftmaxParameter_Engine_CUDNN) {
    return CreateLayerBase<CuDNNSoftmaxLayer>(lparam, ftype, btype);
#endif
  } else {
    LOG(FATAL) << "Layer " << lparam.name() << " has unknown engine.";
  }
}

REGISTER_LAYER_CREATOR(Softmax, GetSoftmaxLayer);

// Get tanh layer according to engine.
shared_ptr<LayerBase> GetTanHLayer(const LayerParameter& param,
    Type ftype, Type btype) {
  TanHParameter_Engine engine = param.tanh_param().engine();
  if (engine == TanHParameter_Engine_DEFAULT) {
    engine = TanHParameter_Engine_CAFFE;
#ifdef USE_CUDNN
    if (Caffe::mode() == Caffe::GPU)
      engine = TanHParameter_Engine_CUDNN;
#endif
  }
  if (engine == TanHParameter_Engine_CAFFE) {
    return CreateLayerBase<TanHLayer>(param, ftype, btype);
#ifdef USE_CUDNN
  } else if (engine == TanHParameter_Engine_CUDNN) {
    return CreateLayerBase<CuDNNTanHLayer>(param, ftype, btype);
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
  }
}

REGISTER_LAYER_CREATOR(TanH, GetTanHLayer);

// Get dropout layer according to engine
shared_ptr<LayerBase> GetDropoutLayer(const LayerParameter& param,
  Type ftype, Type btype) {
  DropoutParameter_Engine engine = param.dropout_param().engine();
  if (engine == DropoutParameter_Engine_DEFAULT) {
    engine = DropoutParameter_Engine_CAFFE;
#ifdef USE_CUDNN
    if (Caffe::mode() == Caffe::GPU) {
      engine = DropoutParameter_Engine_CUDNN;
    }
#endif
  }
  if (engine == DropoutParameter_Engine_CAFFE) {
    return CreateLayerBase<DropoutLayer>(param, ftype, btype);
  }
#ifdef USE_CUDNN
  else if (engine == DropoutParameter_Engine_CUDNN) {
    return CreateLayerBase<CuDNNDropoutLayer>(param, ftype, btype);
  }
#endif
  else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
  }
}

REGISTER_LAYER_CREATOR(Dropout, GetDropoutLayer);

#ifdef USE_OPENCV
shared_ptr<LayerBase> GetDetectNetTransformationLayer(const LayerParameter& param,
    Type ftype, Type btype) {
  LayerParameter lparam(param);
  check_precision_support(ftype, btype, lparam);
  shared_ptr<LayerBase> ret;
  if (is_type<double>(ftype)) {
    ret.reset(new DetectNetTransformationLayer<double>(lparam));
  } else {
    ret.reset(new DetectNetTransformationLayer<float>(lparam));
  }
  return ret;
}
REGISTER_LAYER_CREATOR(DetectNetTransformation, GetDetectNetTransformationLayer);
#endif

#ifdef WITH_PYTHON_LAYER
shared_ptr<LayerBase> GetPythonLayer(const LayerParameter& param, Type, Type) {
  try {
    std::lock_guard<std::mutex> lock(PythonLayer<float, float>::mutex());
    bp::object module;
    PYTHON_CALL_BEGIN
    LOG(INFO) << "Importing Python module '" << param.python_param().module() << "'";
    module = bp::import(param.python_param().module().c_str());
    PYTHON_CALL_END
    bp::object layer = module.attr(param.python_param().layer().c_str())(param);
    shared_ptr<LayerBase> ret = bp::extract<shared_ptr<LayerBase>>(layer)();
    CHECK(ret);
    return ret;
  } catch (...) {
    PyErrFatal();
  }
}

REGISTER_LAYER_CREATOR(Python, GetPythonLayer);
#endif

void check_precision_support(Type& ftype, Type& btype, LayerParameter& param) {
  if (!is_precise(ftype) || !is_precise(btype)) {
    Type MT = tp<float>();
    if (Caffe::is_main_thread()) {
      LOG(WARNING) << "Layer '" << param.name() << "' of type '"
          << param.type() << "' is not supported in " << Type_Name(FLOAT16)
          << " precision. Falling back to " << Type_Name(MT) << ". You might use "
              "'forward_type: FLOAT' and 'backward_type: FLOAT' "
              "settings to suppress this warning.";
    }
    ftype = MT;
    btype = MT;
    param.set_forward_type(MT);
    param.set_backward_type(MT);
    param.set_forward_math(MT);
    param.set_backward_math(MT);
  }
}

// Layers that use their constructor as their default creator should be
// registered in their corresponding cpp files. Do not register them here.
}  // namespace caffe
