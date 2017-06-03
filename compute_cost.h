#ifndef FLANN_COM_COST
#define FLANN_COM_COST

#include"PQSingle_index.h"
#include"IVFADC_index.h"
#include"PQIndex.h"

#include "flann/general.h"
#include "flann/algorithms/nn_index.h"
#include "flann/nn/ground_truth.h"
#include "flann/nn/index_testing.h"
#include "flann/util/sampling.h"
#include "flann/algorithms/kdtree_index.h"
#include "flann/algorithms/kdtree_single_index.h"
#include "flann/algorithms/kmeans_index.h"
#include "flann/algorithms/composite_index.h"
#include "flann/algorithms/linear_index.h"
#include "flann/util/logger.h"



namespace flann
{



struct TestCostIndexParams : public IndexParams
{
	TestCostIndexParams( float build_weight = 0.01, float memory_weight = 0)
	{
		(*this)["build_weight"] = build_weight;
        (*this)["memory_weight"] = memory_weight;
	}
};



template <typename Distance>
class ComputeCostPQ
{
public:
	typedef typename Distance::ElementType ElementType;
    typedef typename Distance::ResultType DistanceType;


	ComputeCostPQ(const Matrix<ElementType> & inputData, const IndexParams & params, const IndexParams & cost_params)
		:index(inputData, params)
	{
		dataset_memory_ =  float(inputData.rows * inputData.cols * sizeof(float));
		build_weight_ = get_param<float>(cost_params, "build_weight");
		memory_weight_ = get_param<float>(cost_params, "memory_weight");
	}

	void start_test(const Matrix<ElementType>& queries,
    		Matrix<int>& indices,
			Matrix<int>& truth_indices,
    		Matrix<DistanceType>& dists,
    		int knn,
    		const SearchParams& params)
	{
		Logger::info("start building PQ Index\n");
		StartStopTimer t;
		t.start();
		index.buildIndex();
		t.stop();
		buildtime_ = (float)t.value;
		Logger::info("end building PQ Index\n");

		knn_ = knn;

		Logger::info("start searching\n");
		t.reset();
		t.start();
		index.knnSearch(queries,indices,dists,knn,params);
		t.stop();
		searchtime_ = (float)t.value;
		Logger::info("end searching\n");

		compute_precisions<int>(indices, truth_indices);
		
		memory_ = index.usedMemory() + dataset_memory_;

	}

	void start_test(const Matrix<ElementType>& queries,
    		Matrix<size_t>& indices,
			Matrix<size_t>& truth_indices,
    		Matrix<DistanceType>& dists,
    		size_t knn,
    		const SearchParams& params)
	{
		Logger::info("start building PQ Index\n");
		StartStopTimer t;
		t.start();
		index.buildIndex();
		t.stop();
		buildtime_ = (float)t.value;
		Logger::info("end building PQ Index\n");

		knn_ = knn;


		Logger::info("start searching\n");
		t.reset();
		t,start();
		index.knnSearch(queries,indices,dists,knn,params);
		t.stop();
		searchtime_ = (float)t.value;
		Logger::info("end searching\n");

		compute_precisions<size_t>(indices, truth_indices);

	}

	void show_result()
	{
		std::cout<<"==========================================="<<std::endl;
		std::cout<<"precison:_____"<<precision_<<"_____"<<std::endl
				<<"searchtime:_____"<<searchtime_<<"_____"<<std::endl
				<<"buildtime:_____"<<buildtime_<<"_____"<<std::endl;
		std::cout<<"==========================================="<<std::endl;
	}

public:
	PQIndex<L2<float>> index;

private:

	template<typename T>
	void compute_precisions(Matrix<T> &indices, Matrix<T> &truth_indices)
	{
		size_t rows = indices.rows;
		size_t trows = truth_indices.rows;
		assert( rows == trows);

		size_t obj;
		size_t count = 0;
		for(int i=0;i<rows;i++)
		{
			obj = truth_indices[i][0];
			for(int j=0;j<indices.cols;j++)
			{
				size_t index = indices[i][j];
				if( index == obj )
				{
					count++;
					break;
				}
			}
		}
		precision_ = (float)count/rows;
	}

public:

	float build_weight_;
	float memory_weight_;

	float precision_;
	float searchtime_;
	float buildtime_;
	float memory_;

	float dataset_memory_;

	size_t knn_;

};



}

#endif //FLANN_COM_COST