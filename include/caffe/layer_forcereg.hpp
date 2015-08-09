//////////////////////////////////////////////////////////////////////////
// This file must be included in every windows program / dll that is going
// to link libcaffe.lib. Without including this file the unused symbols
// will not be used and the layers are not registered.
//////////////////////////////////////////////////////////////////////////

#ifdef _WIN32

#pragma once

#include "caffe/vision_layers.hpp"

namespace caffe
{
    REGISTER_LAYER_CLASS(AbsVal);
    REGISTER_LAYER_CLASS(Accuracy);
    REGISTER_LAYER_CLASS(ArgMax);
    REGISTER_LAYER_CLASS(BNLL);
    REGISTER_LAYER_CLASS(Concat);
    REGISTER_LAYER_CLASS(ContrastiveLoss);
    REGISTER_LAYER_CLASS(Data);
    REGISTER_LAYER_CLASS(Deconvolution);
    REGISTER_LAYER_CLASS(Dropout);
    REGISTER_LAYER_CLASS(DummyData);
    REGISTER_LAYER_CLASS(Eltwise);
    REGISTER_LAYER_CLASS(EuclideanLoss);
    REGISTER_LAYER_CLASS(Exp);
    REGISTER_LAYER_CLASS(Flatten);
    REGISTER_LAYER_CLASS(HDF5Data);
    REGISTER_LAYER_CLASS(HDF5Output);
    REGISTER_LAYER_CLASS(HingeLoss);
    REGISTER_LAYER_CLASS(Im2col);
    REGISTER_LAYER_CLASS(ImageData);
    REGISTER_LAYER_CLASS(InfogainLoss);
    REGISTER_LAYER_CLASS(InnerProduct);
    REGISTER_LAYER_CLASS(MemoryData);
    REGISTER_LAYER_CLASS(MultinomialLogisticLoss);
    REGISTER_LAYER_CLASS(MVN);
    REGISTER_LAYER_CLASS(Power);
    REGISTER_LAYER_CLASS(PReLU);
    REGISTER_LAYER_CLASS(Reshape);
    REGISTER_LAYER_CLASS(SigmoidCrossEntropyLoss);
    REGISTER_LAYER_CLASS(Silence);
    REGISTER_LAYER_CLASS(Slice);
    REGISTER_LAYER_CLASS(SoftmaxWithLoss);
    REGISTER_LAYER_CLASS(Split);
    REGISTER_LAYER_CLASS(SPP);
    REGISTER_LAYER_CLASS(Threshold);
    REGISTER_LAYER_CLASS(WindowData);
}

#endif
