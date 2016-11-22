#include <stdlib.h>
#include "caffe/util/coll.h"

#define SYNC_PERIOD 8

template <typename Dtype>
__global__ void pipeline_bcast_kernel(int* my_progress, Dtype* my_data,
  int* next_progress, Dtype* next_data, const int size) {
  int tid = threadIdx.x;

  if (tid == 0) {
    if (my_progress != NULL) {
      my_progress[blockIdx.x] = 0;  // Signal receiver is ready
    }
    // Wait for receiver to be ready
    if (next_progress != NULL) {
      while (((volatile int*)next_progress)[blockIdx.x] != 0) {}
    }
  }
  __syncthreads();

  // Each block manages its portion of the buffer
  int block_size = (size+gridDim.x-1)/gridDim.x;
  int start = block_size*blockIdx.x;
  int end = block_size*(blockIdx.x+1);
  if (end > size) end = size;
  int progress = (my_progress == NULL) ? \
                 end : ((volatile int*)my_progress)[blockIdx.x];
  __threadfence_system();

  int sync = SYNC_PERIOD;

  for (int index = start + tid;
       index < end;
       index += blockDim.x) {
    if (progress < index) {
      while ((progress = ((volatile int*)my_progress)[blockIdx.x]) < index) {}
      __threadfence_system();
    }
    if (next_data != NULL) {
      next_data[index] = my_data[index];  // Copy data
    }
    if (next_progress != NULL) {
      if (--sync == 0) {
        __syncthreads();
        if (tid == 0) {
          __threadfence_system();
          next_progress[blockIdx.x] = index + blockDim.x;
        }
        sync = SYNC_PERIOD;
      }
    }
  }
  if (next_progress != NULL) {
    // Don't forget the last update
    __syncthreads();
    if (tid == 0) {
      __threadfence_system();
      next_progress[blockIdx.x] = size;
    }
  }
}

template <typename Dtype>
__global__ void pipeline_sum_kernel(int* my_progress, Dtype* red_data,
  Dtype* my_data, int* next_progress, Dtype* next_data, Dtype factor,
  const int size) {
  int tid = threadIdx.x;

  if (tid == 0) {
    if (my_progress != NULL) {
      my_progress[blockIdx.x] = 0;  // Signal receiver is ready
    }
    // Wait for receiver to be ready
    if (next_progress != NULL) {
      while (((volatile int*)next_progress)[blockIdx.x] != 0) {}
    }
  }
  __syncthreads();

  // Each block manages its portion of the buffer
  int block_size = (size+gridDim.x-1)/gridDim.x;
  int start = block_size*blockIdx.x;
  int end = block_size*(blockIdx.x+1);
  if (end > size) end = size;
  int progress = (my_progress == NULL) ? \
                 end : ((volatile int*)my_progress)[blockIdx.x];
  __threadfence_system();

  int sync = SYNC_PERIOD;

  for (int index = start + tid;
       index < end;
       index += blockDim.x) {
    if (progress < index) {
      while ((progress = ((volatile int*)my_progress)[blockIdx.x]) < index) {}
      __threadfence_system();
    }
    if (my_progress != NULL) {
      // Add it to my data
      red_data[index] += my_data[index];
    }
    if (next_progress != NULL) {
      next_data[index] = red_data[index];  // Send it to next
      if (--sync == 0) {
        __syncthreads();
        if (tid == 0) {
          __threadfence_system();
          next_progress[blockIdx.x] = index + blockDim.x;
        }
        sync = SYNC_PERIOD;
      }
    } else {
      red_data[index] *= factor;
    }
  }
  if (next_progress != NULL) {
    // Don't forget the last update
    __syncthreads();
    if (tid == 0) {
      __threadfence_system();
      next_progress[blockIdx.x] = size;
    }
  }
}

template <typename Dtype>
__global__ void ring_sum_kernel(const int rank, const int nranks,
  int* my_progress, Dtype* red_data, Dtype* my_data, int* next_progress,
  Dtype* next_red_data, Dtype* next_data, Dtype factor, const int size) {
  const int tid = threadIdx.x, bid = blockIdx.x;

  // To keep good performance, align a bit. Align block/chunk sizes to a power
  // of two but not that big that it will break the algorithm.
  const int max_delta_size = size / (gridDim.x*gridDim.x*nranks*nranks);
  int align;
  for (align = 2; align < max_delta_size && align < (nranks*2048); \
      align <<= 1) {}
  align >>= 1;
  const int block_size = ((size+(gridDim.x*align)-1)/(gridDim.x*align)) * \
                         align;  // upper(size/dim)
  const int block_start = block_size*bid;
  const int b_end = block_size*(bid+1);
  const int block_end = b_end < size ? b_end : size;

  const int chunk_size = (block_size+nranks-1)/nranks;  // upper(b_size/nranks)
  const int start = block_start + rank * chunk_size;
  const int boundaries[4] = {
      // Phase 0 :   1 chunk  : send data to tmp buffer
      block_start + ((rank+1)%nranks) * chunk_size,
      // Phase 1 : n-2 chunks : add data + send to tmp buffer
      block_start + ((rank-1+nranks)%nranks) * chunk_size,
      // Phase 2 :   1 chunk  : add data + send to final buffer
      start,  // avoid to recompute block_start + rank * chunk_size
      // Phase 3 : n-2 chunks : send to final buffer
      block_start + ((rank-2+nranks)%nranks) * chunk_size
      // Phase 4 :   1 chunk  : wait data
  };

  // Do not wait to start sending first chunk : it is initial data
  int progress = boundaries[0];

  // Sync with receiver
  if (tid == 0) {
    while (((volatile int*)next_progress)[bid] != -1) {}
    __threadfence_system();
    next_progress[bid] = -2;
    while (((volatile int*)my_progress)[bid] != -2) {}
    __threadfence_system();
    my_progress[bid] = progress;
    while (((volatile int*)next_progress)[bid] != start) {}
    __threadfence_system();
  }
  __syncthreads();

  int phase = 0;
  Dtype* dest = next_data;  // first loop uses tmp buffer
  int chunk_start = start;

  while (phase < 4) {
    int chunk_end = chunk_start + chunk_size;
    if (chunk_end > block_end) chunk_end = block_end;

    for (int index = chunk_start + tid;
        index < chunk_end;
        index += blockDim.x) {
      // Do sums in phases 1 and 2
      if (phase == 1) {
        red_data[index] += my_data[index];
      } else if (phase == 2) {
        red_data[index] = (red_data[index]+my_data[index])*factor;
      }

      // Copy data to next peer in the ring (rank - 1)
      dest[index] = red_data[index];
    }

    chunk_start = chunk_end;
    if (chunk_start >= block_end) chunk_start = block_start;

    // Update at chunk boundaries
    __syncthreads();
    if (tid == 0) {
        __threadfence_system();
        next_progress[bid] = chunk_start;
    }

    // Update phase
    while (phase < 4 && chunk_start == boundaries[phase]) {
      phase++;
      // Phase 2 and 3 work on final buffer
      if (phase == 2) dest = next_red_data;
    }

    // Wait for prev to finish next chunk
    while (progress == chunk_start) {
      while (progress == ((volatile int*)my_progress)[bid]) {}
      progress = ((volatile int*)my_progress)[bid];
      __threadfence_system();
    }
  }
  // Reset progress for next call
  __syncthreads();
  if (tid == 0) my_progress[bid] = -1;
}

template <>
void multi_gpu_pipeline_bcast<float>(int *my_progress, float* my_data,
    int *next_progress, float* next_data, const int size, const int grid_dim,
    cudaStream_t stream) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  pipeline_bcast_kernel<float><<<grid_dim, COLL_NUM_THREADS, 0, stream>>>(
      my_progress, my_data, next_progress, next_data, size);
}

template <>
void multi_gpu_pipeline_bcast<double>(int *my_progress, double* my_data,
    int *next_progress, double* next_data, const int size, const int grid_dim,
    cudaStream_t stream) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  pipeline_bcast_kernel<double><<<grid_dim, COLL_NUM_THREADS, 0, stream>>>(
      my_progress, my_data, next_progress, next_data, size);
}

template<>
void multi_gpu_pipeline_sum<float>(int *my_progress, float* red_data,
    float* my_data, int *next_progress, float* next_data, float factor,
    const int size, const int grid_dim, cudaStream_t stream) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  pipeline_sum_kernel<float><<<grid_dim, COLL_NUM_THREADS, 0, stream>>>(
      my_progress, red_data, my_data, next_progress, next_data, factor, size);
}

template<>
void multi_gpu_pipeline_sum<double>(int *my_progress, double* red_data,
    double* my_data, int *next_progress, double* next_data, double factor,
    const int size, const int grid_dim, cudaStream_t stream) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  pipeline_sum_kernel<double><<<grid_dim, COLL_NUM_THREADS, 0, stream>>>(
      my_progress, red_data, my_data, next_progress, next_data, factor, size);
}

template<>
void multi_gpu_ring_sum<float>(const int rank, const int nranks,
    int *my_progress, float* red_data, float* my_data, int *next_progress,
    float* next_red_data, float* next_data, float factor, const int size,
    const int grid_dim, cudaStream_t stream) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  ring_sum_kernel<float><<<grid_dim, COLL_NUM_THREADS, 0, stream>>>(
      rank, nranks, my_progress, red_data, my_data, next_progress,
      next_red_data, next_data, factor, size);
}

template<>
void multi_gpu_ring_sum<double>(const int rank, const int nranks,
    int *my_progress, double* red_data, double* my_data, int *next_progress,
    double* next_red_data, double* next_data, double factor, const int size,
    const int grid_dim, cudaStream_t stream) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  ring_sum_kernel<double><<<grid_dim, COLL_NUM_THREADS, 0, stream>>>(
      rank, nranks, my_progress, red_data, my_data, next_progress,
      next_red_data, next_data, factor, size);
}
