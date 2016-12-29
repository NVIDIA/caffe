#pragma once

#ifdef _MSC_VER
  #define TEMPLATE_EXTERN
  #if defined libcaffe_EXPORTS
    #define CAFFE_EXPORT __declspec(dllexport)
  #else
    #define CAFFE_EXPORT
  #endif
#else
  #define CAFFE_EXPORT
  #define CAFFE_TEMPLATE_EXTERN extern
#endif
