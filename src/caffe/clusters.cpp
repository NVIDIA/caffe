#include "caffe/clusters.hpp"

namespace Clusters{
	
  int node_rank_;
  int node_count_;  
  int node_local_rank_;
  int node_local_count_;  

  void Init() {
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &node_rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &node_count_);
    MPI_Comm shmcomm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                        MPI_INFO_NULL, &shmcomm);
    MPI_Comm_rank(shmcomm, &node_local_rank_);
    MPI_Comm_size(shmcomm, &node_local_count_);
  }
  
  void Finalize() {
    MPI_Finalize();	
  }
/*  
  void ClusterAllreduce(int count, void* bucket, caffe::Type type) {
    if (caffe::is_type<float>(type)) {
      MPI_Allreduce(MPI_IN_PLACE, bucket, count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);  
    } else if (caffe::is_type<double>(type)) {
      MPI_Allreduce(MPI_IN_PLACE, bucket, count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);  
    } else {
      LOG(FATAL) << "Type " << caffe::Type_Name(type) << " is not supported by Caffe-MPI";
    }  
  }
*/
/*
  void ClusterBcast(int count, void* bucket, caffe::Type type, int root) {
    if (caffe::is_type<float>(type)) {
      MPI_Bcast(bucket, count, MPI_FLOAT, root, MPI_COMM_WORLD);  
    } else if (caffe::is_type<double>(type)) {
      MPI_Bcast(bucket, count, MPI_DOUBLE, root, MPI_COMM_WORLD);  
    } else {
      LOG(FATAL) << "Type " << caffe::Type_Name(type) << " is not supported by Caffe-MPI";
    }
  }
*/  
  int node_rank() {
    return node_rank_;
  }

  int node_count() {
    return node_count_;
  }
  
  int node_local_rank() {
    return node_local_rank_;
  }

  int node_local_count() {
    return node_local_count_;
  }

}
