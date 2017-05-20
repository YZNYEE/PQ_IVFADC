/*

implement PQ algorithm based on FLANN software

created on 2017/5/20

*/

#ifndef FLANN_PQSINGLE_INDEX_H
#define FLANN_PQSINGLE_INDEX_H

#include <algorithm>
#include <string>
#include <map>
#include <cassert>
#include <limits>
#include <cmath>

#include "flann/general.h"
#include "flann/algorithms/nn_index.h"
#include "flann/algorithms/dist.h"
#include <flann/algorithms/center_chooser.h>
#include "flann/util/matrix.h"
#include "flann/util/result_set.h"
#include "flann/util/heap.h"
#include "flann/util/allocator.h"
#include "flann/util/random.h"
#include "flann/util/saving.h"
#include "flann/util/logger.h"

namespace flann{

enum part_method{
	NATURAL = 0,
	RANDOM = 1,
	STRUCTURED = 2

}; 

struct PQSingleIndexParams : public IndexParams{

	PQSingleIndexParams(int cluster_k = 32,int part_m = 8,int method = 0,int iterator = 11, 
						flann_centers_init_t centers_init = FLANN_CENTERS_RANDOM)
	{
	
		(*this)["algorithm"] = -100;
		(*this)["cluster_k"] = cluster_k;
		(*this)["splite_m"] = part_m;
		(*this)["method"] = method;
		(*this)["iterator"] = iterator;
		(*this)["center_init"] = centers_init;

	}
};


/*
	PQ index
*/


template <typename Distance>
class PQSingleIndex : public NNIndex<Distance>
{
public:
	typedef typename Distance::ElementType ElementType;
	typedef typename Distance::DistanceType DistanceType;

	typedef NNIndex<Distance> BaseClass;

	typedef bool needs_vector_space_distance;

	int getType() const{
		return 100;
	}

	/*
		Index constructor
	*/

	PQSingleIndex(const Matrix<ElementType>& inputData, const IndexParams& params = PQSingleIndexParams(),
                Distance d = Distance())
				: BaseClass(params,d), memoryCounter_(0)
	{
		cluster_k_ = get_params(params,"cluster_k",32);
		part_m_ = get_param(params,"part_m",8);
        iterations_ = get_param(params,"iterations",11);
		if (iterations_<0) {
            iterations_ = (std::numeric_limits<int>::max)();
        }

		method_ = get_params(params,"method",0);
		centers_init_  = get_param(params,"centers_init",FLANN_CENTERS_RANDOM);

		initCenterChooser();
        chooseCenters_->setDataset(inputData);

		setDataset(inputData);

	}

	PQSingleIndex(const IndexParams& params = PQSingleIndexParams(), Distance d = Distance())
        : BaseClass(params, d),  memoryCounter_(0)
    {
		cluster_k_ = get_params(params,"cluster_k",32);
		part_m_ = get_param(params,"part_m",8);
        iterations_ = get_param(params,"iterations",11);
		if (iterations_<0) {
            iterations_ = (std::numeric_limits<int>::max)();
        }

		method_ = get_params(params,"method",0);
		centers_init_  = get_param(params,"centers_init",FLANN_CENTERS_RANDOM);

		initCenterChooser();
    }

	/* not implementing */

	PQSingleIndex(const KMeansIndex& other) : BaseClass(other),
    	    (other.branching_),
    		iterations_(other.iterations_),
    		centers_init_(other.centers_init_),
    		cluster_k_(other.cluster_k_),
			part_m_(other.part_m_),
			method_(other.method_),
    		memoryCounter_(other.memoryCounter_)
    {
    	initCenterChooser();
    }

	PQSingleIndex& operator=(PQSingleIndex other)
    {
    	this->swap(other);
    	return *this;
    }

	void initCenterChooser()
    {
        switch(centers_init_) {
        case FLANN_CENTERS_RANDOM:
        	chooseCenters_ = new RandomCenterChooser<Distance>(distance_);
        	break;
        case FLANN_CENTERS_GONZALES:
        	chooseCenters_ = new GonzalesCenterChooser<Distance>(distance_);
        	break;
        case FLANN_CENTERS_KMEANSPP:
            chooseCenters_ = new KMeansppCenterChooser<Distance>(distance_);
        	break;
        default:
            throw FLANNException("Unknown algorithm for choosing initial centers.");
        }
    }

	    /**
     * Index destructor.
     *
     * Release the memory used by the index.
     */
    virtual ~PQSingleIndex()
    {
    	delete chooseCenters_;
    	freeIndex();
    }

	BaseClass* clone() const
    {
    	return new PQSingleIndex(*this);
    }

	/*
		it is temporary.
		it will be changed.
	*/

	int usedMemory() const
    {
        return pool_.usedMemory+pool_.wastedMemory+memoryCounter_;
    }

	using BaseClass::buildIndex;

	void addPoints(const Matrix<ElementType>& points, float rebuild_threshold = 2)
    {
        assert(points.cols==veclen_);
        size_t old_size = size_;

        extendDataset(points);
        
        if (rebuild_threshold>1 && size_at_build_*rebuild_threshold<size_) {
            buildIndex();
        }
        else {
			// undetermined    
        }
    }

	//undetermined

	template<typename Archive>
    void serialize(Archive& ar)
    {
    	ar.setObject(this);

    	ar & *static_cast<NNIndex<Distance>*>(this);

    	ar & cluster_k_;
		ar & part_m_;
		ar & method_;
    	ar & iteration_;
    	ar & memoryCounter_;
    	ar & centers_init_;

    	if (Archive::is_loading::value) {
    		// undetermined
    	}

    	if (Archive::is_loading::value) {
            index_params_["algorithm"] = -100;
            index_params_["cluster_k"] = cluster_k_;
			index_params_["part_m"] = part_m_;
            index_params_["iterations"] = iterations_;
            index_params_["centers_init"] = centers_init_;
            index_params_["method"] = method_;
    	}
    }

	void saveIndex(FILE* stream)
    {
    	serialization::SaveArchive sa(stream);
    	sa & *this;
    }

    void loadIndex(FILE* stream)
    {
    	freeIndex();
    	serialization::LoadArchive la(stream);
    	la & *this;
    }

	 /**
     * Find set of nearest neighbors to vec. Their indices are stored inside
     * the result object.
     *
     * Params:
     *     result = the result object in which the indices of the nearest-neighbors are stored
     *     vec = the vector for which to search the nearest neighbors
     *     searchParams = parameters that influence the search algorithm (search_method ADC|SDC)
     */

	/*
	* searchParams may be reconstructed.
	*/
	
    void findNeighbors(ResultSet<DistanceType>& result, const ElementType* vec, const SearchParams& searchParams) const
    {
    	if (removed_) {
    		findNeighborsWithRemoved<true>(result, vec, searchParams);
    	}
    	else {
    		findNeighborsWithRemoved<false>(result, vec, searchParams);
    	}

    }

	/*
		not implementing
	*/

	void getLookupTable(){};

protected:

	/*
		build the index
	*/

	void buildIndexImpl(){
	
    {
        if (cluster_k_<2) {
            throw FLANNException("cluster_k_ factor must be at least 2");
        }
	
		partMatrix(); // to part matrix based on method

		computeClustering(); // to compute clustering centor and initial lookup_table

	}

private:


	/*
		define some struct 
	*/ 


	void freeIndex()
	{
	}

	void compluteClustering();

	void compluteSubClustering();

	void partMatrix();

	void findNeighborsWithRemoved(ResultSet<DistanceType>& result, const ElementType* vec, const SearchParams& searchParams) const
	{
	
		int searchMethod = searchParams.searchMethod;
		if(searchMethod < 0)
			searchMethod = -1;

		/*
			searchMethod == 1 ,using ADC
			searchMethod == 0 ,using SDC
			searchMethod == -1,undefined
		*/

		if(searchMethod == 1)
		{
			findNNADC<with_removed>();
		}
		else{
			findNNSDC<with_removed>();
		}
	
	}

	template<bool with_removed>
	void findNNADC();

	template<bool with_removed>
	void findNNSDC();

	void swap(PQSingleIndex& other)
    {
    	std::swap(cluster_k, other.cluster_k);
    	std::swap(iterations_, other.iterations_);
    	std::swap(centers_init_, other.centers_init_);
    	std::swap(part_m_, other.part_m_);
    	std::swap(pool_, other.pool_);
    	std::swap(memoryCounter_, other.memoryCounter_);
    	std::swap(chooseCenters_, other.chooseCenters_);
    
		// etc
	}

	int cluster_k_;

	int part_m_;

	int method_;

	int iteration_s;

	int memoryCounters_;

	CenterChooser<Distance>* chooseCenters_;

};

}


#endif //FLANN_PQSINGLE_INDEX_H_/