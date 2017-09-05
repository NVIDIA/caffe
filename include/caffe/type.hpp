#ifndef INCLUDE_CAFFE_TYPE_HPP_
#define INCLUDE_CAFFE_TYPE_HPP_

//  enum Type
#include "caffe/proto/caffe.pb.h"

#ifndef CPU_ONLY
  #include "caffe/util/float16.hpp"
#endif

namespace caffe {

template <typename Dtype>
constexpr Type tp();

template <>
inline constexpr Type tp<double>() {
  return DOUBLE;
}
template <>
inline constexpr Type tp<float>() {
  return FLOAT;
}
#ifndef CPU_ONLY
template <>
inline constexpr Type tp<float16>() {
  return FLOAT16;
}
#endif
template <>
inline constexpr Type tp<int>() {
  return INT;
}
template <>
inline constexpr Type tp<unsigned int>() {
  return UINT;
}
template <>
inline constexpr Type tp<bool>() {
  //CHECK(false) << "Should not reach here: tp<bool>()";
  return BOOL;
}

template <typename Dtype>
constexpr DatumTypeInfo datum_tp();
template <>
inline constexpr DatumTypeInfo datum_tp<Datum>() {
  return DatumTypeInfo_DATUM;
}
template <>
inline constexpr DatumTypeInfo datum_tp<AnnotatedDatum>() {
  return DatumTypeInfo_ANNOTATED_DATUM;
}

template <typename T1, typename T2>
Type tpmax() {
  // min because DOUBLE < FLOAT < FLOAT16
  return (Type) std::min<int>((int)tp<T1>(), (int)tp<T2>());
};

inline Type tpm(Type t1, Type t2) {
  // min because DOUBLE < FLOAT < FLOAT16
  return (Type) std::min<int>((int)t1, (int)t2);
}

template <typename Dtype>
bool is_type(Type dtype);

template <>
inline bool is_type<double>(Type dtype) {
  return dtype == DOUBLE;
}
template <>
inline bool is_type<float>(Type dtype) {
  return dtype == FLOAT;
}
#ifndef CPU_ONLY
template <>
inline bool is_type<float16>(Type dtype) {
  return dtype == FLOAT16;
}
#endif
template <>
inline bool is_type<int>(Type dtype) {
  return dtype == INT;
}
template <>
inline bool is_type<unsigned int>(Type dtype) {
  return dtype == UINT;
}

inline bool is_precise(Type dtype) {
  return dtype == FLOAT || dtype == DOUBLE;
}

/**
 * @brief This function supports math types only
 */
size_t tsize(Type dtype);

}  // namespace caffe

#endif /* INCLUDE_CAFFE_TYPE_HPP_ */
