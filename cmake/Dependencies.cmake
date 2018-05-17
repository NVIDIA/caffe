# This list is required for static linking and exported to CaffeConfig.cmake
set(Caffe_LINKER_LIBS "")

# ---[ Boost
find_package(Boost 1.54 REQUIRED COMPONENTS system thread filesystem regex)
include_directories(SYSTEM ${Boost_INCLUDE_DIR})
list(APPEND Caffe_LINKER_LIBS ${Boost_LIBRARIES})

# ---[ Threads
find_package(Threads REQUIRED)
list(APPEND Caffe_LINKER_LIBS ${CMAKE_THREAD_LIBS_INIT})

# ---[ Google-glog
include("cmake/External/glog.cmake")
include_directories(SYSTEM ${GLOG_INCLUDE_DIRS})
list(APPEND Caffe_LINKER_LIBS ${GLOG_LIBRARIES})

# ---[ Google-gflags
include("cmake/External/gflags.cmake")
include_directories(SYSTEM ${GFLAGS_INCLUDE_DIRS})
list(APPEND Caffe_LINKER_LIBS ${GFLAGS_LIBRARIES})

# ---[ Google-protobuf
include(cmake/ProtoBuf.cmake)

# ---[ HDF5
find_package(HDF5 COMPONENTS HL REQUIRED)
include_directories(SYSTEM ${HDF5_INCLUDE_DIRS} ${HDF5_HL_INCLUDE_DIR})
list(APPEND Caffe_LINKER_LIBS ${HDF5_LIBRARIES} ${HDF5_HL_LIBRARIES})

# ---[ LMDB
if(USE_LMDB)
  find_package(LMDB REQUIRED)
  include_directories(SYSTEM ${LMDB_INCLUDE_DIR})
  list(APPEND Caffe_LINKER_LIBS ${LMDB_LIBRARIES})
  add_definitions(-DUSE_LMDB)
endif()

# ---[ LevelDB
if(USE_LEVELDB)
  find_package(LevelDB REQUIRED)
  include_directories(SYSTEM ${LevelDB_INCLUDE})
  list(APPEND Caffe_LINKER_LIBS ${LevelDB_LIBRARIES})
  add_definitions(-DUSE_LEVELDB)
endif()

# ---[ Snappy
if(USE_LEVELDB)
  find_package(Snappy REQUIRED)
  include_directories(SYSTEM ${Snappy_INCLUDE_DIR})
  list(APPEND Caffe_LINKER_LIBS ${Snappy_LIBRARIES})
endif()

find_package(JPEGTurbo REQUIRED)
include_directories(SYSTEM ${JPEGTurbo_INCLUDE_DIR})
list(APPEND Caffe_LINKER_LIBS ${JPEGTurbo_LIBRARIES})

# ---[ CUDA
include(cmake/Cuda.cmake)
if(NOT HAVE_CUDA)
  message(SEND_ERROR "-- CUDA is not detected by cmake. Building without it...")
endif()

# ---[ OpenCV
find_package(OpenCV QUIET COMPONENTS imgcodecs)
if(OPENCV_IMGCODECS_FOUND)
  find_package(OpenCV REQUIRED COMPONENTS core imgcodecs highgui imgproc videoio)
  message(STATUS "Found OpenCV 3.x: ${OpenCV_CONFIG_PATH}")
else()
  find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc)
  message(STATUS "Found OpenCV 2.x: ${OpenCV_CONFIG_PATH}")
endif()
include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})
list(APPEND Caffe_LINKER_LIBS ${OpenCV_LIBS})

# ---[ BLAS
if(NOT APPLE)
  set(BLAS "Open" CACHE STRING "Selected BLAS library")
  set_property(CACHE BLAS PROPERTY STRINGS "Atlas;Open;MKL")

  if(BLAS STREQUAL "Atlas" OR BLAS STREQUAL "atlas")
    find_package(Atlas REQUIRED)
    include_directories(SYSTEM ${Atlas_INCLUDE_DIR})
    list(APPEND Caffe_LINKER_LIBS ${Atlas_LIBRARIES})
  elseif(BLAS STREQUAL "Open" OR BLAS STREQUAL "open")
    find_package(OpenBLAS REQUIRED)
    include_directories(SYSTEM ${OpenBLAS_INCLUDE_DIR})
    list(APPEND Caffe_LINKER_LIBS ${OpenBLAS_LIB})
  elseif(BLAS STREQUAL "MKL" OR BLAS STREQUAL "mkl")
    find_package(MKL REQUIRED)
    include_directories(SYSTEM ${MKL_INCLUDE_DIR})
    list(APPEND Caffe_LINKER_LIBS ${MKL_LIBRARIES})
    add_definitions(-DUSE_MKL)
  endif()
elseif(APPLE)
  find_package(vecLib REQUIRED)
  include_directories(SYSTEM ${vecLib_INCLUDE_DIR})
  list(APPEND Caffe_LINKER_LIBS ${vecLib_LINKER_LIBS})
endif()

# ---[ Python
if(BUILD_python)
  find_package(PythonInterp ${python_version})

  find_library(Boost_PYTHON_FOUND NAMES
          python-py${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR}
          boost_python-py${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR}
          boost_python${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR}
          PATHS ${LIBDIR})
  if ("${Boost_PYTHON_FOUND}" STREQUAL "Boost_PYTHON_FOUND-NOTFOUND")
    message(SEND_ERROR "Could NOT find Boost Python Library")
  else()
    message(STATUS "Found Boost Python Library ${Boost_PYTHON_FOUND}")
    list(APPEND Caffe_LINKER_LIBS ${Boost_PYTHON_FOUND})
  endif()

  find_package(PythonLibs ${python_version})
  find_package(NumPy 1.7.1)
  if(PYTHONLIBS_FOUND AND NUMPY_FOUND AND Boost_PYTHON_FOUND)
    set(HAVE_PYTHON TRUE)
    if(BUILD_python_layer)
      add_definitions(-DWITH_PYTHON_LAYER)
      include_directories(SYSTEM ${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIR} ${Boost_INCLUDE_DIRS})
      list(APPEND Caffe_LINKER_LIBS ${PYTHON_LIBRARIES})
    endif()
  endif()
endif()

# ---[ Matlab
if(BUILD_matlab)
  find_package(MatlabMex)
  if(MATLABMEX_FOUND)
    set(HAVE_MATLAB TRUE)
  endif()

  # sudo apt-get install liboctave-dev
  find_program(Octave_compiler NAMES mkoctfile DOC "Octave C++ compiler")

  if(HAVE_MATLAB AND Octave_compiler)
    set(Matlab_build_mex_using "Matlab" CACHE STRING "Select Matlab or Octave if both detected")
    set_property(CACHE Matlab_build_mex_using PROPERTY STRINGS "Matlab;Octave")
  endif()
endif()

if(USE_MPI)
  find_package(MPI REQUIRED)
  include_directories(SYSTEM ${MPI_CXX_INCLUDE_PATH})
  list(APPEND Caffe_LINKER_LIBS ${MPI_CXX_LIBRARIES})
  add_definitions(-DOMPI_SKIP_MPICXX)
  add_definitions(-DUSE_MPI)
  message(STATUS "MPI found (${MPI_CXX_LIBRARIES}, ${MPI_CXX_INCLUDE_PATH})")
endif()
# ---[ Doxygen
if(BUILD_docs)
  find_package(Doxygen)
endif()

# ---[ NCCL
if(USE_NCCL_SET)
  if(USE_NCCL)
    find_package(NCCL REQUIRED)
  endif()
else()
  find_package(NCCL)
endif()
if(NCCL_FOUND)
  add_definitions(-DUSE_NCCL)
  include_directories(SYSTEM ${NCCL_INCLUDE_DIR})
  list(APPEND Caffe_LINKER_LIBS ${NCCL_LIBRARY})
endif()

# ---[ NVML
if(NOT NO_NVML)
  find_package(NVML)
endif()
if(NVML_FOUND)
  include_directories(SYSTEM ${NVML_INCLUDE})
  list(APPEND Caffe_LINKER_LIBS ${NVML_LIBRARY})
endif()

if(NO_NVML)
  add_definitions(-DNO_NVML=1)
endif()

if(TEST_FP16)
  add_definitions(-DTEST_FP16=1)
endif()

