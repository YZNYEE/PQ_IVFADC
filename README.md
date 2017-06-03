PQ algorithm and IVFADC structure
===============
  这是一个对FLANN软件包的扩展，实现了Product Quantization算法和IVFADC结构。以及相关测试程序。<br>

--------------
Precondition
--------------
  为了保证程序在电脑上正确运行，请确保在主机上正确安装flann软件包和HDF5接口。并且设置好相关的编译环境。

--------------
PQ_algorithm
--------------
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
  
  load_from_file(dataset, "dataset.hdf5","dataset");
  load_from_file(query, "dataset.hdf5","query");
    
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

要使用PQ算法。需要在原先的基础上包含`PQIndex.h`头文件，并且需要初始化相应的PQ算法索引的建立参数。建立该算法的索引参数类为`PQSingleIndexParams`。建立该实例主要需要初始化四个参数:<br>
```c++
PQSingleIndexParams(int cluster_k = 32,int part_m = 8,int method = 0,int iterator = 11,flann_centers_init_t centers_init = FLANN_CENTERS_RANDOM)
```
这四个参数依次表示:每个部分聚类中心的个数`cluster_k`,分划个数`part_m`,分划方法`method`（注：目前只提供两种分划方法 0：连续分划，1：随机分划，指定分划方法还未实现）,迭代次数`iterator`(因为PQ算法建立时有大量的kmean操作，为了缩短建立索引结构所需的时间，设定迭代次数，为12时效果较好。-1为最佳聚类中心，无穷大)

另外一个需要改变的是搜索参数`PQSearchParams`:
```c++
PQSearchParams(int checks_num = 8,int smethod = 1,int checks_ = 32, float eps_ = 0.0, bool sorted_ = true)
```
主要需要初始化的为`checks_num`,`smethod`。checks_num为IVFADC结构提供，PQSingle算法可以忽略。smethod为选定搜索算法（0：SDC，1：ADC），默认为ADC。IVFADC算法可以忽略smethod。

其余的接口与原FLANN软件包相同。<br>

--------------------
IVFADC
--------------------

IVFADC只PQSingle有些许的不同，在清楚PQSingle算法的前提下，只需做以下改变就能正常使用IVFADC索引结构。

```c++
flann::PQSingleIndexParams PQIP(256,8,0,12);
PQIndex<L2<float>> index(dataset, PQIP);
```
```c++
flann::IVFADCIndexParams IVFIP(256,8,0,12,128);
PQIndex<L2<float>> index(dataset, IVFIP);
```
做如上替换，将PQIP替换为IVFIP。它们的参数前四个的含义完全相同，其中IVFADCIndexParams中第五个参数表示指定建立128个粗聚类中心。IVFADC搜索参数与PQSingle相同。<br>




