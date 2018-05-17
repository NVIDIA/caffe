#include <mpi.h>
#include "caffe/proto/caffe.pb.h"
#include "caffe/type.hpp"

namespace Clusters{
	
  //int node_rank;
  //int node_count;
  
  void Init();
  
  void Finalize();
  
  //void ClusterAllreduce(int count, void* bucket, caffe::Type type);

  //void ClusterBcast(int count, void* bucket, caffe::Type type, int root);

  int node_rank();

  int node_count();

  int node_local_rank();

  int node_local_count();

}
