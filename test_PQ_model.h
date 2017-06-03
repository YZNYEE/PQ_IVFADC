#ifndef FLANN_TEST_PQ_MODEL
#define FLANN_TEST_PQ_MODEL

#include "compute_cost.h"
#include "select_kdtree.h"

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

template <typename Distance>
class TestPQModel
{
public:
	typedef typename Distance::ElementType ElementType;
	typedef typename Distance::ResultType DistanceType;

	TestPQModel(Matrix<ElementType> & inputData, const IndexParams & params, const IndexParams & cost_params,Distance d = Distance())
		:pq_params_(params),cost_params_(cost_params),kd_params_(kd_params),dataset_(inputData),distance_(d)
	{

		pq_model_ = new ComputeCostPQ<Distance>(dataset_,pq_params_,cost_params_);
		
	}

	void evaulate(const Matrix<ElementType>& queries,
    		Matrix<int>& pq_indices,
			Matrix<int>& kd_indices,
			Matrix<int>& truth_indices,
    		Matrix<DistanceType>& pq_dists,
			Matrix<DistanceType>& kd_dists,
    		int knn,
    		const SearchParams& pq_params)
	{
		Logger::info("start PQ model\n");
		pq_model_->start_test(queries,pq_indices,truth_indices,pq_dists,knn,pq_params);
		Logger::info("complete PQ model\n");
		/*
		kd_params_["target_precision"] = pq_model_->precision_;
		kdtree_ = new SelectBestKDTree<Distance>(dataset_,kd_params_,distance_);
		kdtree_->buildIndex();
		kdtree_->knnSearch<int>(queries,kd_indices,truth_indices,kd_dists,1);
		Logger::info("complete kdtree model\n");
		*/
	}

	void evaulate(const Matrix<ElementType>& queries,
    		Matrix<size_t>& pq_indices,
			Matrix<size_t>& kd_indices,
			Matrix<size_t>& truth_indices,
    		Matrix<DistanceType>& pq_dists,
			Matrix<DistanceType>& kd_dists,
    		size_t knn,
    		const SearchParams& pq_params)
	{
		pq_model_.start_test(queries,pq_indices,truth_indices,pq_dists,knn,params);
		/*
		kd_params_["target_precision"] = pq_model_->precision_;
		kdtree_ = new SelectBestKDTree(dataset_,kd_params_,distance_);
		kdtree_->buildIndex();
		kdtree_->knnSearch<size_t>(queries,kd_indices,truth_indices,kd_dists,1);
		*/
	}

	void show_result()
	{
		Logger::info("\nPQSingleIndexParams::\n");
		print_params(pq_params_);
		//Logger::info("\nKDTreeIndexParams::\n");
		//print_params(kdtree_->bestParams_);
		//Logger::info("\nKDTreeSearchParams::\n");
		//print_params(kdtree_->bestSearchParams_);
		Logger::info("| algorithm | precision | buildtime(s) | searchtime(s) | memory_use(b) |\n");
		std::cout<<"  PQSingle  "<<pq_model_->precision_<<"   "<<pq_model_->buildtime_<<"   "<<pq_model_->searchtime_
				<<"   "<<pq_model_->memory_<<std::endl;
	//	std::cout<<"  KDTreeIn  "<<kdtree_->truth_precision_<<"   "<<kdtree_->buildtimecost_<<"   "<<kdtree_->searchtimecost_
	//				<<"   "<<kdtree_->memory_<<std::endl;
	}

	void save_key_information(std::string filename)
	{
		FILE* fout = fopen(filename.c_str(), "wb");
        if (fout == NULL) {
            throw FLANNException("Cannot open file");
        }
		serialization::SaveArchive sa(fout);
		sa & (*this);
	}

	void save_pq_model(std::string filename)
	{
		(pq_model_->index).save(sfilename);
	}


	template<typename Archive>
	void serialize(Archive& ar)
	{
		ar & pq_model_->build_weight_;
		ar & pq_model_->memory_weight_;

		ar & pq_model_->precision_;
		
		ar & pq_model_->buildtime_;
		ar & pq_model_->searchtime_;
		ar & pq_model_->memory_;
		/*
		ar & kdtree_->buildtimecost_;
		ar & kdtree_->searchtimecost_;
		ar & kdtree_->memory_;
	*/
	}

private:

	ComputeCostPQ<Distance>* pq_model_;
	//SelectBestKDTree<Distance>* kdtree_;

	IndexParams pq_params_;
	IndexParams cost_params_;
	//IndexParams kd_params_;

	Distance distance_;
	Matrix<ElementType> dataset_;
};

}

#endif //FLANN_TEST_PQ_MODEL