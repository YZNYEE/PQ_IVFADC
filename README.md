PQ algorithm and IVFADC structure
===============
  这是一个对FLANN软件包的扩展，实现了Product Quantization算法和IVFADC结构。以及相关测试程序。<br>

--------------
Precondition
--------------
  为了保证程序在电脑上正确运行，请确保在主机上正确安装flann软件包和HDF5接口。并且设置好相关的编译环境。

--------------
Introduction
--------------
`PQ_algorithm:`<br>
该软件包实现了一个简单的Product Quantization算法。实现该算法的主要文件为`PQSingle_index.h`。为了统一接口，所以额外地实现了两个文件`PQ_indices.h`,`PQIndex.h`。这两个文件保证了新添加的PQ算法与原FLANN软件包提供的kdtree，kmeans等算法相似的接口，但需要做些许调整。一个简单是实例：<br>
```c++
#include <flann/flann.hpp> 
#include <flann/io/hdf5.h> 
#include "PQIndex.h"

using namespace flann;

int main(int argc, char** argv)
{
  int nn = 100;
	//Logger::setLevel(5);
  Matrix<float> dataset;
	Matrix<float> query;
  
  load_from_file(dataset, "D:\\sift\\sift\\sift_min10.hdf5","dataset");
  load_from_file(query, "D:\\sift\\sift\\sift_min10.hdf5","dataset");
    
  Matrix<int> indices(new int[query.rows*nn], query.rows, nn);
	Matrix<float> dists(new float[query.rows*nn], query.rows, nn);
  
  flann::PQSingleIndexParams PQIP(256,8,0,12);
	PQIndex<L2<float>> index(dataset, PQIP);

	index.buildIndex();

	flann::PQSearchParams PQSP;
	index.knnSearch(query,indices,dists,nn,PQSP);
  	
  flann::save_to_file(indices,"result.hdf5","result");

  delete[] dataset.ptr();
  delete[] query.ptr();
	delete[] indices.ptr();
	delete[] dists.ptr();
    
    return 0;
}
```

