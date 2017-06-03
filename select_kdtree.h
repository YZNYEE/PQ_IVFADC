
#ifndef FLANN_SELECT_KDTREE
#define FLANN_SELECT_KDTREE

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

struct SelectKDTreeIndexParams : public IndexParams
{
	SelectKDTreeIndexParams(float target_precision = 0.8, float build_weight = 0.01, float memory_weight = 0, float sample_fraction = 0.1)
	{
		(*this)["algorithm"] = FLANN_INDEX_AUTOTUNED;
        (*this)["target_precision"] = target_precision;
        (*this)["build_weight"] = build_weight;
        (*this)["memory_weight"] = memory_weight;
        (*this)["sample_fraction"] = sample_fraction;
	}
};

template <typename Distance>
class SelectBestKDTree
{
public:
	typedef typename Distance::ElementType ElementType;
    typedef typename Distance::ResultType DistanceType;

	SelectBestKDTree(const Matrix<ElementType> &inputData, const IndexParams& params = SelectKDtreeIndexParams(), Distance d = Distance()):
		bestIndex_(NULL), dataset_(inputData), distance_(d)
	{

		target_precision_ = get_param(params, "target_precision",0.8f);
		build_weight_ =  get_param(params,"build_weight", 0.01f);
		memory_weight_ = get_param(params, "memory_weight", 0.0f);
        sample_fraction_ = get_param(params,"sample_fraction", 0.1f);
	}

	void buildIndex()
	{
		
		Logger::info("estimateBuildParams ----------------------------\n");
		bestParams_ = estimateBuildParams();
		Logger::info("finish estimating BuildParams ----------------------------\n");
        
		Logger::info("----------------------------------------------------\n");
        Logger::info("Autotuned parameters:\n");
        if (Logger::getLevel()>=FLANN_LOG_INFO)
        	print_params(bestParams_);
        Logger::info("----------------------------------------------------\n");

		flann_algorithm_t index_type = get_param<flann_algorithm_t>(bestParams_,"algorithm");
        bestIndex_ = create_index_by_type(index_type, dataset_, bestParams_, distance_);

		Logger::info("=========================================================\n");

		Logger::info("start building kd-tree index\n");
		StartStopTimer t;
		t.start();
        bestIndex_->buildIndex();
		t.stop();
		buildtimecost_ = (float)t.value;
		Logger::info("finish building kd-tree index\n");

		Logger::info("=========================================================\n");

		float datasetMemory = float(dataset_.rows * dataset_.cols * sizeof(float));
		memory_ = (bestIndex_->usedMemory() + datasetMemory) ;

		estimateSearchParams(bestSearchParams_);
        Logger::info("----------------------------------------------------\n");
        Logger::info("Search parameters:\n");
        if (Logger::getLevel()>=FLANN_LOG_INFO)
        	print_params(bestSearchParams_);
        Logger::info("----------------------------------------------------\n");
        bestParams_["search_params"] = bestSearchParams_;
	
	}

	template<typename T>
	void knnSearch(const Matrix<ElementType>& queries,
    		Matrix<T>& indices,
			Matrix<T>& truth_indices,
    		Matrix<DistanceType>& dists,
    		T knn)
	{
		assert(bestIndex_ != NULL);

		StartStopTimer t;
		t.start();
        bestIndex_->knnSearch(queries,indices,dists,1,bestSearchParams_);
		t.stop();
		searchtimecost_ = (float)t.value;

		int cnt = 0;
		for(int i=0;i<queries.rows;i++)
			if(indices[i][0] == truth_indices[i][0])
				cnt++;
		truth_precision_ = float(cnt)/queries.rows;
	
	}
	
	void save_information(FILE* stream)
	{
		serialization::SaveArchive sa(stream);
    	sa & *this;
	}

	void show_result()
	{
		std::cout<<"================================================"<<std::endl
			<<"buildtimecost_____"<<buildtimecost_<<"_____"<<std::endl
			<<"searchtimecost_____"<<searchtimecost_<<"_____"<<std::endl
			<<"memorycost_____"<<memory_<<std::endl
			<<"truth_precision_____"<<truth_precision_<<std::endl
			<<"================================================"<<std::endl;
	}

	template<typename Archive>
	void serialize(Archive& ar)
	{
		ar & bestIndex_["trees"];
		ar & bestSearchParams_["checks"];

		ar & target_precision_;
		ar & build_weight_;
		ar & memory_weight_;
		ar & sample_fraction_;
		ar & buildtimecost_;
		ar & searchtimecost_;
		ar & memory_;
	}

private:

	struct CostData
    {
        float searchTimeCost;
        float buildTimeCost;
        float memoryCost;
        float totalCost;
        IndexParams params;
    };

	void evaluate_kdtree(CostData& cost)
	{
		StartStopTimer t;
        int checks;
        const int nn = 1;

        Logger::info("KDTree using params: trees=%d\n", get_param<int>(cost.params,"trees"));
        KDTreeIndex<Distance> kdtree(sampledDataset_, cost.params, distance_);

        t.start();
        kdtree.buildIndex();
        t.stop();
        float buildTime = (float)t.value;

        //measure search time
        float searchTime = test_index_precision(kdtree, sampledDataset_, testDataset_, gt_matches_, target_precision_, checks, distance_, nn);

        float datasetMemory = float(sampledDataset_.rows * sampledDataset_.cols * sizeof(float));
        cost.memoryCost = (kdtree.usedMemory() + datasetMemory) / datasetMemory;
        cost.searchTimeCost = searchTime;
        cost.buildTimeCost = buildTime;
        Logger::info("KDTree buildTime=%g, searchTime=%g\n", buildTime, searchTime);
	}

	void optimizeKDTree(std::vector<CostData>& costs)
    {
        Logger::info("KD-TREE, Step 1: Exploring parameter space\n");

        // explore kd-tree parameters space using the parameters below
        int testTrees[] = { 1, 4, 8, 16, 32 };

        // evaluate kdtree for all parameter combinations
        for (size_t i = 0; i < FLANN_ARRAY_LEN(testTrees); ++i) {
            CostData cost;
            cost.params["algorithm"] = FLANN_INDEX_KDTREE;
            cost.params["trees"] = testTrees[i];

            evaluate_kdtree(cost);
            costs.push_back(cost);
        }
	}

	IndexParams estimateBuildParams()
	{
		std::vector<CostData> costs;

		int sampleSize = int(sample_fraction_ * dataset_.rows);
		int testSampleSize = std::min(sampleSize / 10, 1000);

		sampledDataset_ = random_sample(dataset_, sampleSize);
		testDataset_ = random_sample(sampledDataset_, testSampleSize, true);

		gt_matches_ = Matrix<size_t>(new size_t[testDataset_.rows], testDataset_.rows, 1);
        StartStopTimer t;
        int repeats = 0;
        t.reset();
        while (t.value<0.2) {
        	repeats++;
            t.start();
        	compute_ground_truth<Distance>(sampledDataset_, testDataset_, gt_matches_, 0, distance_);
            t.stop();
        }

		CostData linear_cost;
        linear_cost.searchTimeCost = (float)t.value/repeats;
        linear_cost.buildTimeCost = 0;
        linear_cost.memoryCost = 0;
        linear_cost.params["algorithm"] = FLANN_INDEX_LINEAR;

		costs.push_back(linear_cost);

		float bestTimeCost = costs[0].buildTimeCost * build_weight_ + costs[0].searchTimeCost;
        for (size_t i = 0; i < costs.size(); ++i) {
            float timeCost = costs[i].buildTimeCost * build_weight_ + costs[i].searchTimeCost;
            Logger::debug("Time cost: %g\n", timeCost);
            if (timeCost < bestTimeCost) {
                bestTimeCost = timeCost;
            }
        }
		best_time_cost_ = bestTimeCost;

		Logger::debug("Best time cost: %g\n", bestTimeCost);

		IndexParams bestParams = costs[0].params;
        if (bestTimeCost > 0) {
        	float bestCost = (costs[0].buildTimeCost * build_weight_ + costs[0].searchTimeCost) / bestTimeCost;
        	for (size_t i = 0; i < costs.size(); ++i) {
        		float crtCost = (costs[i].buildTimeCost * build_weight_ + costs[i].searchTimeCost) / bestTimeCost +
        				memory_weight_ * costs[i].memoryCost;
        		Logger::debug("Cost: %g\n", crtCost);
        		if (crtCost < bestCost) {
        			bestCost = crtCost;
        			bestParams = costs[i].params;
        		}
        	}
            Logger::debug("Best cost: %g\n", bestCost);
        }

        delete[] gt_matches_.ptr();
        delete[] testDataset_.ptr();
        delete[] sampledDataset_.ptr();

        return bestParams;

		Logger::info("Autotuning parameters...\n");
	}

	void estimateSearchParams(SearchParams& searchParams)
	{
		const int nn = 1;
        const size_t SAMPLE_COUNT = 1000;

        assert(bestIndex_ != NULL); // must have a valid index

        float speedup = 0;
		int samples = (int)std::min(dataset_.rows / 10, SAMPLE_COUNT);

     	Matrix<ElementType> testDataset = random_sample(dataset_, samples);
	 	Logger::info("Computing ground truth\n");

            // we need to compute the ground truth first
        Matrix<size_t> gt_matches(new size_t[testDataset.rows], testDataset.rows, 1);
        StartStopTimer t;
        int repeats = 0;
        t.reset();
        while (t.value<0.2) {
           repeats++;
           t.start();
           compute_ground_truth<Distance>(dataset_, testDataset, gt_matches, 1, distance_);
           t.stop();
        }
		int checks;
        Logger::info("Estimating number of checks\n");

        float searchTime;
        float cb_index;
		searchTime = test_index_precision(*bestIndex_, dataset_, testDataset, gt_matches, target_precision_, checks, distance_, nn, 1);

		Logger::info("Required number of checks: %d \n", checks);
        searchParams.checks = checks;

		delete[] gt_matches.ptr();
        delete[] testDataset.ptr();

	}

public:
	NNIndex<Distance> * bestIndex_;

	Distance distance_;
	IndexParams bestParams_;
	SearchParams bestSearchParams_;

	Matrix<ElementType> sampledDataset_;
    Matrix<ElementType> testDataset_;
    Matrix<size_t> gt_matches_;

	Matrix<ElementType> dataset_;

	float truth_precision_;

    float target_precision_;
    float build_weight_;
    float memory_weight_;
    float sample_fraction_;

	float buildtimecost_;
	float searchtimecost_;
	float memory_;

	float best_time_cost_;


};

}


#endif //FLANN_SELECT_KDTREE