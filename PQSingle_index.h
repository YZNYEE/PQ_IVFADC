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
#include <vector>

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
		(*this)["part_m"] = part_m;
		(*this)["method"] = method;
		(*this)["iterator"] = iterator;
		(*this)["center_init"] = centers_init;

		//(*this)["rmatrix"] = rmatrix;

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
	typedef typename Distance::ResultType DistanceType;

	typedef NNIndex<Distance> BaseClass;

	typedef bool needs_vector_space_distance;

	int getType_PQ() const{
		return -100;
	}

	flann_algorithm_t getType() const{
		return FLANN_INDEX_SAVED;
	}

	/*
		Index constructor
	*/

	PQSingleIndex(const Matrix<ElementType>& inputData, const IndexParams& params = PQSingleIndexParams(),
                Distance d = Distance())
				: BaseClass(params,d), memoryCounter_(0)
	{
		//std::cout<<"start"<<std::endl;
		part_m_ = get_param(params,"part_m",8);
        iterations_ = get_param(params,"iterator",11);
		//rmatrix_ = get_param(params,"rmatrix",NULL);
		if (iterations_<0) {
            iterations_ = (std::numeric_limits<int>::max)();
        }

		cluster_k_ = get_param(params,"cluster_k",32);
		method_ = get_param(params,"method",0);
		centers_init_  = get_param(params,"centers_init",FLANN_CENTERS_RANDOM);

		initCenterChooser();
        chooseCenters_->setDataset(inputData);

		setDataset(inputData);
		//std::cout<<"ending the init"<<std::endl;
	}

	PQSingleIndex(const IndexParams& params = PQSingleIndexParams(), Distance d = Distance())
        : BaseClass(params, d),  memoryCounter_(0)
    {
		cluster_k_ = get_param(params,"cluster_k",32);
		part_m_ = get_param(params,"part_m",8);
        iterations_ = get_param(params,"iterator",11);
		//rmatrix_ = get_param(params,"rmatrix",NULL);
		if (iterations_<0) {
            iterations_ = (std::numeric_limits<int>::max)();
        }

		method_ = get_param(params,"method",0);
		centers_init_  = get_param(params,"centers_init",FLANN_CENTERS_RANDOM);

		initCenterChooser();
    }

	/* not implementing */

	PQSingleIndex(const PQSingleIndex& other) : BaseClass(other),
    		iterations_(other.iterations_),
    		centers_init_(other.centers_init_),
    		cluster_k_(other.cluster_k_),
			part_m_(other.part_m_),
			method_(other.method_),
    		memoryCounter_(other.memoryCounter_),
			rmatrix_(other.rmatrix_)
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
		//std::cout<<"delete Index"<<std::endl;
    	freeIndex();
		//std::cout<<"finishing delete"<<std::endl;
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
    	ar & iterations_;
    	ar & memoryCounter_;
    	ar & centers_init_;

		typedef PQSingleIndex<Distance> Index;
		Index* obj = static_cast<Index*>(ar.getObject());

		if(method_ == 2)
			ar & serialization::make_binary_object(rmatrix_, obj->veclen_*sizeof(int));

    	if (Archive::is_loading::value) {
			lookup_table_.resize(part_m_);
			for(int i=0;i<part_m_;i++)
				lookup_table_[i] = new single_cluster(cluster_k_);
			for(int i=0;i<part_m_;i++){
				for(int j=0;j<cluster_k_;j++){
					(*lookup_table_[i])[j] = new(pool_) Node();
				}
			}
    	}

		for(int i=0;i<part_m_;i++)
			for(int j=0;j<cluster_k_;j++)
				ar & *((*lookup_table_[i])[j]);

		ar & part_indices_;
		ar & part_information_;

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
		//std::cout<<"serialization!!"<<std::endl;
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
		//if(search_cnt_%500 == 0)
		//	std::cout<<"has searched "<<search_cnt_<<" vectors!"<<std::endl;
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

	void getLookupTable(int id,std::vector<double *> & sublook)
	{
		if(sublook.size() != cluster_k_)
			sublook.resize(cluster_k_,NULL);
		for(int i=0;i<cluster_k_;i++)
		{
			sublook[i] = new double[veclen_/part_m_];
		}
		for(int i=0;i<cluster_k_;i++)
		{
			for(int j=0;j<veclen_/part_m_;j++)
			{
				//std::cout<<"i: "<<i<<" j: "<<j<<std::endl;
				sublook[i][j] = ((*lookup_table_[id])[i])->pivot[j];
			}
		}
	};

	void getIndices(int id,std::vector<int> & indices)
	{
		if(indices.size() != size_)
		{
			indices.resize(size_);
		}

		for(int i=0;i<size_;i++)
		{
			indices[i] = part_indices_[i][id];
		}
	}

	void getFeature(int id,std::vector<double *> & subfeature)
	{
		int stride = veclen_/part_m_;
		if(subfeature.size() != size_)
		{
			subfeature.resize(size_);
		}
		for(int i=0;i<size_;i++)
		{
			subfeature[i] = new double[stride];
		}

		for(int i=0;i<size_;i++)
		{
			for(int j=0;j<stride;j++)
			{
				subfeature[i][j] = points_[i][part_information_[j+id*stride]];
			}
		}
	}

	int getStride() const
	{
		return veclen_/part_m_;
	}

protected:

	/*
		build the index
	*/

	void buildIndexImpl()
    {
		//std::cout<<"hi!"<<std::endl;
		search_cnt_ = 0;
        if (cluster_k_<2) {
            throw FLANNException("cluster_k_ factor must be at least 2");
        }
	
		Logger::info("start parting\n");

		partMatrix(); // to part matrix based on method

		Logger::info("end parting\n");

		computeClustering(); // to compute clustering centor and initial lookup_table

		Logger::info("end computeClustering\n");

	}

private:


	/*
		define some struct 
	*/ 


	struct Node{
	
		DistanceType * pivot;

		DistanceType radius;

		DistanceType variance;
	
		~Node(){
			delete[] pivot;
		}

		template<typename Archive>
		void serialize(Archive& ar){
		
			//std::cout<<"ok!"<<std::endl;

			typedef PQSingleIndex<Distance> Index;
			Index* obj = static_cast<Index*>(ar.getObject());

    		if (Archive::is_loading::value) {
    			pivot = new DistanceType[obj->veclen_/obj->part_m_];
    		}

			ar & serialization::make_binary_object(pivot, obj->veclen_/obj->part_m_*sizeof(DistanceType));
    		ar & radius;
    		ar & variance;
		
		}
		friend struct serialization::access;

	};
	typedef Node* NodePtr;
	typedef std::vector<NodePtr> single_cluster;
	typedef std::vector<single_cluster*> LookupTable;

	void freeIndex()
	{
		//std::cout<<"hi"<<std::endl;
		if(lookup_table_.size() > 0){		
			for(int i=0;i<part_m_;i++)
				for(int j=0;j<cluster_k_;j++){
					//std::cout<<"i: "<<i<<"j: "<<j<<std::endl;
					(*lookup_table_[i])[j]->~Node();
					(*lookup_table_[i])[j] = NULL;
				}
		}
		pool_.free();
	}

	/*
		initize all subVectors
	*/

	void computeClustering()
	{
		Logger::info("init the same variance\n");
		lookup_table_.resize(part_m_);

		for(int i=0;i<part_m_;i++)
			lookup_table_[i] = new single_cluster(cluster_k_);

		for(int i=0;i<part_m_;i++){
			for(int j=0;j<cluster_k_;j++)
				(*lookup_table_[i])[j] = new(pool_) Node();
		}

		part_indices_.resize(size_);
		for(int i=0;i<size_;i++)
			part_indices_[i].resize(part_m_);

		std::vector<ElementType *> all_sub(size_);
		for(int i=0;i<size_;i++){
			all_sub[i] = new ElementType[veclen_/part_m_];
		}	

		int * indices;
		indices = new int[size_];
		for(int i=0 ;i<size_;i++)
			indices[i] = i;

		//std::cout<<"size: "<<size_<<std::endl;
		//std::cout<<"part_m_: "<<part_m_<<std::endl;
		//std::cout<<"cluster: "<<cluster_k_<<std::endl;
		//std::cout<<"len: "<<veclen_<<std::endl;
		for(int i=0;i<part_m_;i++){

			if(Logger::getLevel() >= FLANN_LOG_INFO)
				std::cout<<"computing "<<i<<"/"<<part_m_<<" cluster"<<std::endl;
			computeSubClustering(i,lookup_table_[i],all_sub,indices);

		}
		

		delete[] indices;
		//std::cout<<"go go"<<std::endl;
		for(int i=0;i<size_;i++){
			delete[] (all_sub[i]);
		}


	}

	void computeSubClustering(int id,single_cluster * save,std::vector<ElementType *>& allsub,int * indices)
	{
		
		Logger::info("initialize every variance\n");
		loadSubVector(id,allsub);
		//delete[] allsub[0];
		//std::cout<<id<<' '<<"delete "<<std::endl;

		std::vector<int> centers_idx(cluster_k_);
		int centers_length;
		(*chooseCenters_)(cluster_k_, indices, size_, &centers_idx[0], centers_length);

		int stride = veclen_/part_m_;
		Matrix<double> dcenters(new double[cluster_k_*stride],cluster_k_,stride);

		for(int i=0;i<centers_length;i++){
			for (size_t k=0; k<stride; ++k) {
                dcenters[i][k] = allsub[centers_idx[i]][k];
            }
		}

		std::vector<DistanceType> radiuses(cluster_k_,0);
        std::vector<int> count(cluster_k_,0);

		std::vector<int> belongs_to(size_);

		Logger::info("computing initial cluster_center\n");
        for (int i=0; i<size_; ++i) {

            DistanceType sq_dist = distance_(allsub[indices[i]], dcenters[0], stride);
            belongs_to[i] = 0;
            for (int j=1; j<cluster_k_; ++j) {
                DistanceType new_sq_dist = distance_(allsub[indices[i]], dcenters[j], stride);
                if (sq_dist>new_sq_dist) {
                    belongs_to[i] = j;
                    sq_dist = new_sq_dist;
                }
            }
            if (sq_dist>radiuses[belongs_to[i]]) {
                radiuses[belongs_to[i]] = sq_dist;
            }
            count[belongs_to[i]]++;
        }
		Logger::info("finish computing initial cluster_center\n");

		bool converged = false;
		int iteration = 0;
		while(!converged && iteration<iterations_){
			converged = true;
			iteration++;

			//compute the new cluster centers;
			if(Logger::getLevel() >= FLANN_LOG_INFO )
				std::cout<<"iterate "<<iteration<<"/"<<iterations_<<std::endl;

			for(int i=0;i<cluster_k_;i++){
			    memset(dcenters[i],0,sizeof(double)*stride);
                radiuses[i] = 0;
			}
			for (int i=0; i<size_; ++i) {
                ElementType* vec = allsub[indices[i]];
                double* center = dcenters[belongs_to[i]];
                for (size_t k=0; k<stride; ++k) {
                    center[k] += vec[k];
                }
            }
			for (int i=0; i<cluster_k_; ++i) {
                int cnt = count[i];
                double div_factor = 1.0/cnt;
                for (size_t k=0; k<stride; ++k) {
                    dcenters[i][k] *= div_factor;
                }
            }
			// reassign points to clusters

			//std::cout<<iteration<<" "<<size_<<std::endl;

            for (int i=0; i<size_; ++i) {
                DistanceType sq_dist = distance_(allsub[indices[i]], dcenters[0], stride);
                int new_centroid = 0;
                for (int j=1; j<cluster_k_; ++j) {
                    DistanceType new_sq_dist = distance_(allsub[indices[i]], dcenters[j], stride);
                    if (sq_dist>new_sq_dist) {
                        new_centroid = j;
                        sq_dist = new_sq_dist;
                    }
                }
                if (sq_dist>radiuses[new_centroid]) {
                    radiuses[new_centroid] = sq_dist;
                }
                if (new_centroid != belongs_to[i]) {
                    count[belongs_to[i]]--;
                    count[new_centroid]++;
                    belongs_to[i] = new_centroid;

                    converged = false;
                }
            }

			for (int i=0; i<cluster_k_; ++i) {
                // if one cluster converges to an empty cluster,
                // move an element into that cluster
                if (count[i]==0) {
					//std::cout<<"empty: "<<i<<std::endl;
                    int j = (i+1)%cluster_k_;
                    while (count[j]<=1) {
                        j = (j+1)%cluster_k_;
                    }
					//std::cout<<"j: "<<j<<std::endl;
                    for (int k=0; k<size_; ++k) {
                        if (belongs_to[k]==j) {
                            belongs_to[k] = i;
                            count[j]--;
                            count[i]++;
							//std::cout<<"k: "<<k<<std::endl;
                            break;
                        }
                    }
                    converged = false;
                }
            }
		}

		if(Logger::getLevel() >= FLANN_LOG_INFO)
			std::cout<<"finish iteration : "<<id<<"  "<<iteration<<"  iterations_: "<<iterations_<<std::endl;
		Logger::info("store result\n");

		for (int i=0; i<cluster_k_; ++i) {
            (*save)[i]->pivot = new DistanceType[stride];
            memoryCounter_ += stride*sizeof(DistanceType);
            for (size_t k=0; k<stride; ++k) {
                ((*save)[i]->pivot)[k] = (DistanceType)dcenters[i][k];
            }
			(*save)[i]->radius = radiuses[i];
        }

		//compute the variance
		for(int i=0; i<cluster_k_; i++){
			int s = count[i];
			DistanceType variance = 0;
            for (int j=0; j<size_; ++j) {
                if (belongs_to[j]==i) {
                    variance += distance_(dcenters[i], allsub[indices[j]], stride);
                }
            }
            variance /= s;
			(*save)[i]->variance = variance;
		}
		
		//construct part_indices_
		for(int i=0;i<size_;i++){
			part_indices_[i][id] = belongs_to[i];
		}

		delete[] dcenters.ptr();
	}

	void loadSubVector(int id,std::vector<ElementType *>& allsub){

		int stride = veclen_/part_m_;
		int first = id*stride;
		for(int i=0;i<size_;i++){
			//std::cout<<i+first<<std::endl;
			for(int j=0;j<stride;j++){
				allsub[i][j] = (ElementType)points_[i][part_information_[j+first]];
			}
			//std::cout<<allsub[0]<<std::endl;
			//delete[] allsub[0];
			//std::cout<<"delete loadsub"<<std::endl;
		}

	}

	void partMatrix()
	{
		part_information_.resize(veclen_);
		for(size_t i=0;i<veclen_;i++){
			part_information_[i] = i;		
		}

		if(method_ == 0)
			return;
		else if(method_ == 1){
			UniqueRandom r(veclen_);
			for(int i=0;i<veclen_;i++){
				int rnd = r.next();
				part_information_[i] = rnd;
			}
			return;
		}
		else if(method_ == 2){
			if(rmatrix_ == NULL){
				throw FLANNException("Dont find rmatrix!!");
				return;
			}
			for(int i=0;i<veclen_;i++){
				part_information_[i] = (*rmatrix_)[i];
			}
			return;
		}
		else{
			throw FLANNException("Unknown algorithm for parting original matrix.");
		}
	}

	/*
		params m:express gain mth part of vector;
				m>=0,m<part_m_
		params subVector:new subvector
	*/
	
	void getSubVector(int m,ElementType * subVector,const ElementType * Vector) const
	{
		int stride = veclen_/part_m_;
		int first = m*stride;
		for(int i=0;i<stride;i++){
			subVector[i] = Vector[part_information_[i+first]];
		}
	}

	template<bool with_removed>
	void findNeighborsWithRemoved(ResultSet<DistanceType>& result, const ElementType* vec, const SearchParams& searchParams) const
	{
		PQSearchParams * now = (PQSearchParams *)(&searchParams);
		int searchMethod = now->searchMethod;
		if(searchMethod < 0)
			searchMethod = 1;

		/*
			searchMethod == 1 ,using ADC
			searchMethod == 0 ,using SDC
			searchMethod == -1,undefined
		*/

		if(searchMethod == 1)
		{
			findNNADC<with_removed>(result,vec);
		}
		else{
			findNNSDC<with_removed>(result,vec);
		}
	
	}

	template<bool with_removed>
	void findNNADC(ResultSet<DistanceType>& result, const ElementType* vec) const
	{

		int stride = veclen_/part_m_;

		std::vector<ElementType*> subvec(part_m_);

		for(int i=0;i<part_m_;i++)
		{
			subvec[i] = new ElementType[stride];
			getSubVector(i,subvec[i],vec);
		}

		DistanceType wsq = result.worstDist();

		std::vector<std::vector<DistanceType>> distance_lookup_table;
		distance_lookup_table.resize(part_m_);
		for(int i=0;i<part_m_;i++)
			distance_lookup_table[i].resize(cluster_k_);

		for(int i=0;i<part_m_;i++)
		{
			for(int j=0;j<cluster_k_;j++)
			{
				DistanceType dis;
				DistanceType * cluster_head;
				cluster_head = (*lookup_table_[i])[j]->pivot;
				dis = distance_(cluster_head, subvec[i],stride);
				distance_lookup_table[i][j] = dis;
			}
		}

		DistanceType dis = 0;
		DistanceType * cluster_head;
		for(int i=0;i<size_;i++){
			dis = 0;
		    if (with_removed) {
				if (removed_points_.test(i)) continue;
            }
			for(int j=0;j<part_m_;j++){
				cluster_head = (*lookup_table_[j])[part_indices_[i][j]]->pivot;
				//dis += distance_(cluster_head, subvec[j], stride);
				dis += distance_lookup_table[j][part_indices_[i][j]];
			}
			result.addPoint(dis, i);

		}
		for(int i=0;i<part_m_;i++)
			delete[] subvec[i];
	}

	template<bool with_removed>
	void findNNSDC(ResultSet<DistanceType>& result, const ElementType* vec) const 
	{
	
		int stride = veclen_/part_m_;

		std::vector<int> product_id(part_m_,0);
		ElementType * subvec = new ElementType[stride];
		DistanceType wsq = result.worstDist();

		/*
			compute the cluster index of vec
		*/

		double * best_dist = new double[part_m_];
		memset(best_dist, 0, sizeof(double)*part_m_);
		double dist = 0;

		DistanceType * cluster_head;
		for(int i=0;i<part_m_;i++){
			dist = 0;
			best_dist[i] = wsq;
			getSubVector(i,subvec,vec);
			for(int j=0;j<cluster_k_;j++){
				cluster_head = (*lookup_table_[i])[j]->pivot;
				dist = distance_(cluster_head, subvec, stride);
				if(dist < best_dist[i]){
					best_dist[i] = dist;
					product_id[i] = j;
				}
			}
		}

		/*
			based on the paper,the distance should be saved in the lookup_table.
			here, compute dist between two clusters.
		*/

		DistanceType * cluster_head_sq;
		DistanceType * cluster_head_fe;
		for(int i=0;i<size_;i++){
			dist = 0;
		    if (with_removed) {
				if (removed_points_.test(i)) continue;
            }
			for(int j=0;j<part_m_;j++){
				cluster_head_sq = (*lookup_table_[j])[product_id[j]]->pivot;
				cluster_head_fe = (*lookup_table_[j])[part_indices_[i][j]]->pivot;
				dist += distance_(cluster_head_fe, cluster_head_sq, stride);
			}
			result.addPoint(dist, i);

		}

		delete[] subvec;
		delete[] best_dist;
	}

	void swap(PQSingleIndex& other)
    {
    	std::swap(cluster_k, other.cluster_k);
    	std::swap(iterations_, other.iterations_);
    	std::swap(centers_init_, other.centers_init_);
    	std::swap(part_m_, other.part_m_);
    	std::swap(pool_, other.pool_);
    	std::swap(memoryCounter_, other.memoryCounter_);
    	std::swap(chooseCenters_, other.chooseCenters_);
 
		std::swap(rmatrix_, other.rmatrix_);
		std::swap(lookup_table_, other.lookup_table_);
		std::swap(part_m_, other.part_m_);

		std::swap(part_information_, other.part_information_);
		std::swap(part_indices_, other.part_indices_);

	}

	PooledAllocator pool_;

	int search_cnt_;

	int cluster_k_;

	int part_m_;

	int method_;

	int iterations_;

	int memoryCounter_;

	CenterChooser<Distance>* chooseCenters_;

	// save the information of partation.

	std::vector<int> part_information_;

	//save every features' clusters

	std::vector<std::vector<int>> part_indices_;

	//part the matrix based on the rmatrix

	std::vector<size_t> * rmatrix_;

	// save the lookup_table used to find the clusters of the feature vector.

	LookupTable lookup_table_;

	flann_centers_init_t centers_init_;

	USING_BASECLASS_SYMBOLS

};

}


#endif /*FLANN_PQSINGLE_INDEX_H*/