/*

	implement the IVFADC based on the FLANN software.

*/

#ifndef FLANN_IVFADC_INDEX_H
#define FLANN_IVFADC_INDEX_H

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

#include "params_PQ.h"
//#include "check.h"

namespace flann{


struct IVFADCIndexParams : public IndexParams{

	IVFADCIndexParams(int cluster_k = 32,int part_m = 8,int method = 0,int iterator = 11,
						int coarse_cluster_k = 1000,
						flann_centers_init_t centers_init = FLANN_CENTERS_RANDOM)
	{
	
		(*this)["algorithm"] = -101;
		(*this)["cluster_k"] = cluster_k;
		(*this)["part_m"] = part_m;
		(*this)["method"] = method;
		(*this)["iterator"] = iterator;
		(*this)["center_init"] = cexnters_init;

		(*this)["coarse_cluster_k"] = coarse_cluster_k;
		//(*this)["rmatrix"] = rmatrix;

	}
};

/* IVFADC index */

template <typename Distance>
class IVFADCIndex : public NNIndex<Distance>
{
public:
	typedef typename Distance::ElementType ElementType;
	typedef typename Distance::ResultType DistanceType;

	typedef NNIndex<Distance> BaseClass;

	typedef bool needs_vector_space_distance;

	int getType_PQ() const{
		return -101;
	}

	flann_algorithm_t getType() const{
		return FLANN_INDEX_SAVED;
	}

	IVFADCIndex(const Matrix<ElementType>& inputData, const IndexParams& params = PQSingleIndexParams(),
                Distance d = Distance())
				: BaseClass(params,d), memoryCounter_(0)
	{
		cluster_k_ = get_param(params,"cluster_k",32);
		part_m_ = get_param(params,"part_m",8);
        iterations_ = get_param(params,"iterator",11);
		//rmatrix_ = get_param(params,"rmatrix",NULL);

		coarse_cluster_k_ = get_param(params,"coarse_cluster_k",1000);

		if (iterations_<0) {
            iterations_ = (std::numeric_limits<int>::max)();
        }

		method_ = get_param(params,"method",0);
		centers_init_  = get_param(params,"centers_init",FLANN_CENTERS_RANDOM);

		initCenterChooser();
        chooseCenters_->setDataset(inputData);

		setDataset(inputData);

	}

	IVFADCIndex(const IndexParams& params = IVFADCIndexParams(), Distance d = Distance())
        : BaseClass(params, d),  memoryCounter_(0)
    {
		cluster_k_ = get_params(params,"cluster_k",32);
		part_m_ = get_param(params,"part_m",8);
        iterations_ = get_param(params,"iterator",11);
		//rmatrix_ = get_param(params,"rmatrix",NULL);
		
		coarse_cluster_k_ = get_param(params,"coarse_cluster_k",1000);
		if (iterations_<0) {
            iterations_ = (std::numeric_limits<int>::max)();
        }

		method_ = get_param(params,"method",0);
		centers_init_  = get_param(params,"centers_init",FLANN_CENTERS_RANDOM);

		initCenterChooser();
    }

	IVFADCIndex(const IVFADCIndex& other) : BaseClass(other),
    		iterations_(other.iterations_),
    		centers_init_(other.centers_init_),
    		cluster_k_(other.cluster_k_),
			part_m_(other.part_m_),
			method_(other.method_),
    		memoryCounter_(other.memoryCounter_),
			rmatrix_(other.rmatrix_),
			coarse_cluster_k_(other.coarse_cluster_k_)
	{
    	initCenterChooser();
    }

	IVFADCIndex& operator=(IVFADCIndex other)
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

	int usedMemory() const
    {
        return pool_.usedMemory+pool_.wastedMemory+memoryCounter_;
    }

	BaseClass* clone() const
    {
    	return new IVFADCIndex(*this);
    }

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
		ar & coarse_cluster_k_;

		typedef IVFADCIndex<Distance> Index;
		Index* obj = static_cast<Index*>(ar.getObject());

		if(method_ == 2)
			ar & serialization::make_binary_object(rmatrix_, obj->veclen_*sizeof(int));

    	if (Archive::is_loading::value) {
			lookup_table_.resize(part_m_);
			for(int i=0;i<part_m_;i++){
				if(lookup_table_[i]!=NULL)
					lookup_table_[i]->resize(cluster_k_);
				else
					lookup_table_[i] = new single_cluster(cluster_k_);
				for(int j=0;j<cluster_k_;j++){
					(*lookup_table_[i])[j] = new(pool_) Node();
				}
			}

			for(int i=0;i<coarse_cluster_k_;i++)
				if(coarse_index_[i]==NULL)
					coarse_index_[i] = new(pool_) coarse_node();
    	}

		for(int i=0;i<part_m_;i++)
			for(int j=0;j<cluster_k_;j++)
				ar & *((*lookup_table_[i])[j]);

		for(int i=0;i<coarse_cluster_k_;i++)
			ar & coarse_index_[i];

		ar & part_indices_;
		ar & part_information_;

    	if (Archive::is_loading::value) {
            index_params_["algorithm"] = -100;
            index_params_["cluster_k"] = cluster_k_;
			index_params_["part_m"] = part_m_;
            index_params_["iterator"] = iterations_;
            index_params_["centers_init"] = centers_init_;
            index_params_["method"] = method_;
			index_params_["coarse_cluster_k"] = coarse_cluster_k_;
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

	void findNeighbors(ResultSet<DistanceType>& result, const ElementType* vec, const SearchParams& searchParams) const
	{
		if (removed_) {
    		findNeighborsWithRemoved<true>(result, vec, searchParams);
    	}
    	else {
    		findNeighborsWithRemoved<false>(result, vec, searchParams);
    	}
	}

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

	void getCoarseCenter(std::vector<double *> & ccenter)
	{
		ccenter.resize(coarse_index_.size());
		for(int i=0;i<coarse_index_.size();i++)
		{
			ccenter[i] = new double[veclen_];
			for(int j=0;j<veclen_;j++)
				ccenter[i][j] = coarse_index_[i]->pivot[j];
		}
	}

	void getAllFeature(std::vector<double *> & all)
	{
		int stride = veclen_;
		if(all.size() != size_)
		{
			all.resize(size_);
		}
		for(int i=0;i<size_;i++)
		{
			all[i] = new double[stride];
		}

		for(int i=0;i<size_;i++)
		{
			for(int j=0;j<stride;j++)
			{
				all[i][j] = points_[i][j];
			}
		}
	
	}

	void getCoarseIndices(std::vector<int> & cindices)
	{
		if(cindices.size()!=size_)
		{
			cindices.resize(size_);
		}
		for(int i=0;i<coarse_cluster_k_;i++)
		{
			for(int j=0;j<coarse_index_[i]->volume;j++)
			{
				cindices[coarse_index_[i]->container[j]] = i;
			}
		}
	}

	int getStride() const
	{
		return veclen_/part_m_;
	}

protected:

	void buildIndexImpl()
    {

		//std::cout<<"start building"<<std::endl;
        if (cluster_k_<2) {
            throw FLANNException("cluster_k_ factor must be at least 2");
        }
	
		partMatrix(); // to part matrix based on method

		Logger::info("===============================\n");
		Logger::info("start computing coarse clusters\n");
		computeCoarseClustering(); // to compute clustering centor and initial lookup_table
		Logger::info("start computing rest vector clusters\n");
		computeRestClustering();
		Logger::info("end up building index");

	}

private:

	/* 
		define some struct
	*/

	struct coarse_node{
	
		DistanceType * pivot;

		DistanceType radius;

		DistanceType variance;

		std::vector<int> container;

		int volume;

		coarse_node(){}

		coarse_node(int num):container(num),volume(num){}

		~coarse_node(){
		
			delete[] pivot;

		}

		template<typename Archive>
		void serialize(Archive& ar){
		
			typedef IVFADCIndex<Distance> Index;
			Index* obj = static_cast<Index*>(ar.getObject());

			ar & volume;

    		if (Archive::is_loading::value) {
    			pivot = new DistanceType[obj->veclen_];
    			contains.resize(volume)
			}

			ar & serialization::make_binary_object(pivot, obj->veclen_*sizeof(DistanceType));
    		ar & radius;
    		ar & variance;
			ar & container;
		
		}
		friend struct serialization::access;
	
	};

	struct Node{
	
		DistanceType * pivot;

		DistanceType radius;

		DistanceType variance;
	
		~Node(){
			delete[] pivot;
		}

		template<typename Archive>
		void serialize(Archive& ar){
		
			typedef IVFADCIndex<Distance> Index;
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

	typedef coarse_node* coarse_node_ptr;
	typedef std::vector<coarse_node_ptr> coarse_table;
	typedef Node* NodePtr;
	typedef std::vector<NodePtr> single_cluster;
	typedef std::vector<single_cluster*> LookupTable;

	void freeIndex(){
		if(lookup_table_.size()>0)
		{
			for(int i=0;i<size_;i++)
				for(int j=0;j<cluster_k_;j++){
					(*lookup_table_[i])[j]->~Node();
					(*lookup_table_[i])[j] = NULL;
				}
		}
		if(coarse_index_.size()>0)
		{
			for(int i=0;i<coarse_cluster_k_;i++){
				coarse_index_[i]->~coarse_node();
				coarse_index_[i] = NULL;
			}
		}
		pool_.free();
	}

	void computeCoarseClustering()
	{
	
		int * indices = new int[size_];
		for(int i=0;i<size_;i++)
			indices[i] = i;

		std::vector<ElementType *> pivots(coarse_cluster_k_);
		for(int i=0;i<coarse_cluster_k_;i++)
		{
			pivots[i] = new ElementType[veclen_];
		}
		std::vector<double> radius(coarse_cluster_k_);
		std::vector<double> variance(coarse_cluster_k_);
		std::vector<int> belongs(size_);
		std::vector<int> counts(coarse_cluster_k_);

		Logger::info("finish initializing and start computing\n");
		computeClustering(points_,veclen_,indices,coarse_cluster_k_,pivots,radius,variance,belongs,counts);
		Logger::info("end up computing and store the result\n");

		coarse_index_.resize(coarse_cluster_k_);
		for(int i=0;i<coarse_cluster_k_;i++){
			coarse_index_[i] = new(pool_) coarse_node(counts[i]);
			coarse_index_[i]->radius = radius[i];
			coarse_index_[i]->variance = variance[i];
			coarse_index_[i]->pivot = new DistanceType[veclen_];
			memoryCounter_ += veclen_*sizeof(DistanceType);
			for(int j=0;j<veclen_;j++)
				(coarse_index_[i]->pivot)[j] = (DistanceType)pivots[i][j];
		}

		std::vector<int> pos(coarse_cluster_k_,0);
		for(int i=0;i<size_;i++){
			(coarse_index_[belongs[i]]->container)[pos[belongs[i]]] = i;
			pos[belongs[i]]++;
		}

		/*check*/
		size_t sum = 0;
		for(int i=0;i<coarse_cluster_k_;i++){
			assert(coarse_index_[i]->volume == (coarse_index_[i]->container).size());
			assert(coarse_index_[i]->volume == pos[i]);
			sum+=coarse_index_[i]->volume;
		}

		for(int i=0;i<coarse_cluster_k_;i++)
			delete[] pivots[i];
		delete[] indices;

	}

	void computeRestClustering()
	{
		int * indices = new int[size_];
		for(int i=0;i<size_;i++)
			indices[i] = i;

		int stride = veclen_/part_m_;

		std::vector<ElementType *> restvec(size_);
		for(int i=0;i<size_;i++)
			restvec[i] = new ElementType[veclen_];
		for(int i=0;i<coarse_cluster_k_;i++)
			for(int j=0;j<coarse_index_[i]->volume;j++)
				for(int k=0;k<veclen_;k++)
					restvec[(coarse_index_[i]->container)[j]][k] = points_[(coarse_index_[i]->container)[j]][k]
																	- (coarse_index_[i]->pivot)[k];

		std::vector<ElementType *> allsub(size_);
		for(int i=0;i<size_;i++)
			allsub[i] = new ElementType[stride];

		std::vector<ElementType *> pivots(cluster_k_);
		for(int i=0;i<cluster_k_;i++)
			pivots[i] = new ElementType[stride];
		std::vector<double> radius(cluster_k_);
		std::vector<double> variance(cluster_k_);
		std::vector<int> belongs(size_);
		std::vector<int> counts(cluster_k_);

		/*
			intialize the container.
		*/
		part_indices_.resize(size_);
		for(int i=0;i<size_;i++)
			part_indices_[i].resize(part_m_);
		
		lookup_table_.resize(part_m_);
		for(int i=0;i<part_m_;i++){
			lookup_table_[i] = new single_cluster(cluster_k_);
			for(int j=0;j<cluster_k_;j++){
				(*lookup_table_[i])[j] = new(pool_) Node();
				(*lookup_table_[i])[j]->pivot = new DistanceType[stride];
				memoryCounter_ += stride*sizeof(DistanceType);
			}
		}

		for(int i=0;i<part_m_;i++)
		{
			loadSubVector(i,size_,allsub,restvec);
			if(Logger::getLevel() >= FLANN_LOG_INFO)
				std::cout<<"computing "<<i<<"/"<<part_m_<<" cluster"<<std::endl;
			computeClustering(allsub,stride,indices,cluster_k_,pivots,radius,variance,belongs,counts);

			Logger::info("store the result\n");
			// save part_indices
			for(int j=0;j<size_;j++)
				part_indices_[j][i] = belongs[j];

			for(int j=0;j<cluster_k_;j++)
			{
				(*lookup_table_[i])[j]->radius = radius[j];
				(*lookup_table_[i])[j]->variance = variance[j];
				for(int k=0;k<stride;k++)
				{
					((*lookup_table_[i])[j]->pivot)[k] = (DistanceType)pivots[j][k];
				}
			}

		}


		for(int i=0;i<cluster_k_;i++)
			delete[] pivots[i];
		for(int i=0;i<size_;i++){
			delete[] allsub[i];
			delete[] restvec[i];
		}
		delete[] indices;

	}



	/*
		it is universal functions to compute cluster.
		params[1]:dataset
		params[2]:width of the vector
		params[3]:
		params[4]:num of the clusters
		params[5]:save the pivot
		params[6]:save the radius of centers
		params[7]:save the variance of centers
		params[8]:save the owner
	*/

	void computeClustering(std::vector<ElementType*>& data,int cols,int * indices,int cluster_k, 
							std::vector<DistanceType*>& pivots_,std::vector<double>& radius_,
							std::vector<double>& variance_,std::vector<int>& belongs_,
							std::vector<int>& counts_)
	{
		std::vector<int> centers_idx(cluster_k);
		int centers_length;
		(*chooseCenters_)(cluster_k, indices, size_, &centers_idx[0], centers_length);

		int stride = cols;
		Matrix<double> dcenters(new double[cluster_k*stride],cluster_k,stride);

		for(int i=0;i<centers_length;i++){
			for (size_t k=0; k<stride; ++k) {
                dcenters[i][k] = data[centers_idx[i]][k];
            }
		}

		std::vector<DistanceType> radiuses(cluster_k,0);
        std::vector<int> count(cluster_k,0);

		std::vector<int> belongs_to(size_);
        for (int i=0; i<size_; ++i) {

            DistanceType sq_dist = distance_(data[indices[i]], dcenters[0], stride);
            belongs_to[i] = 0;
            for (int j=1; j<cluster_k; ++j) {
                DistanceType new_sq_dist = distance_(data[indices[i]], dcenters[j], stride);
                if (sq_dist>new_sq_dist) {
                    belongs_to[i] = j;
                    sq_dist = new_sq_dist;
                }
            }
            if (sq_dist>radiuses[belongs_to[i]]) {
                radiuses[belongs_to[i]] = sq_dist;
            }
            count[belongs_to[i]]++;

			//std::cout<<i<<" belongs to "<<belongs_to[i]<<" "<<count[belongs_to[i]]<<std::endl;;
        }

		bool converged = false;
		int iteration = 0;
		while(!converged && iteration<iterations_){
			converged = true;
			iteration++;

			//compute the new cluster centers;

			if(Logger::getLevel() >= FLANN_LOG_INFO )
				std::cout<<"iterate "<<iteration<<"/"<<iterations_<<std::endl;

			for(int i=0;i<cluster_k;i++){
			    memset(dcenters[i],0,sizeof(double)*stride);
                radiuses[i] = 0;
			}
			for (int i=0; i<size_; ++i) {
                ElementType* vec = data[indices[i]];
                double* center = dcenters[belongs_to[i]];
                for (size_t k=0; k<stride; ++k) {
                    center[k] += vec[k];
                }
            }
			for (int i=0; i<cluster_k; ++i) {
                int cnt = count[i];
                double div_factor = 1.0/cnt;
                for (size_t k=0; k<stride; ++k) {
                    dcenters[i][k] *= div_factor;
                }
            }
			// reassign points to clusters

            for (int i=0; i<size_; ++i) {
                DistanceType sq_dist = distance_(data[indices[i]], dcenters[0], stride);
                int new_centroid = 0;
                for (int j=1; j<cluster_k; ++j) {
                    DistanceType new_sq_dist = distance_(data[indices[i]], dcenters[j], stride);
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

			for (int i=0; i<cluster_k; ++i) {
                // if one cluster converges to an empty cluster,
                // move an element into that cluster
                if (count[i]==0) {
                    int j = (i+1)%cluster_k;
                    while (count[j]<=1) {
						//std::cout<<count[j]<<std::endl;
                        j = (j+1)%cluster_k;
                    }

                    for (int k=0; k<size_; ++k) {
                        if (belongs_to[k]==j) {
                            belongs_to[k] = i;
                            count[j]--;
                            count[i]++;
                            break;
                        }
                    }
                    converged = false;
                }
            }
		}

		//save the information

		for (int i=0; i<cluster_k; ++i) 
            for (size_t k=0; k<stride; ++k) 
                pivots_[i][k] = (DistanceType)dcenters[i][k];

		for (int i=0; i<cluster_k;i++)
			radius_[i] = radiuses[i];

		for(int i=0; i<cluster_k; i++){
			int s = count[i];
			DistanceType variance = 0;
            for (int j=0; j<size_; ++j) {
                if (belongs_to[j]==i) {
                    variance += distance_(dcenters[i], data[indices[j]], stride);
                }
            }
            variance /= s;
			variance_[i] = variance;
		}

		for(int i=0;i<size_;i++)
			belongs_[i] = belongs_to[i];

		for(int i=0;i<cluster_k;i++)
			counts_[i] = count[i];

		delete[] dcenters.ptr();

	}

	template<bool with_removed>
	void findNeighborsWithRemoved(ResultSet<DistanceType>& result, const ElementType* vec, const SearchParams& searchParams) const
	{
	
		PQSearchParams * now = (PQSearchParams *)(&searchParams);
		int checks = now->checks_coarse_cluster;
		findNN<with_removed>(result,vec,checks);
	
	}
	
	struct dist_index
	{
		int index;
		DistanceType dis;
		dist_index():index(0),dis(0){}
		dist_index(int id,DistanceType d):index(id),dis(d){}
		bool operator<(const dist_index & other) const
		{
			return dis<other.dis;
		}
		dist_index& operator=(const dist_index& other){
			index = other.index;
			dis = other.dis;
			return (*this);
		}
	};

	template<bool with_removed>
	void findNN(ResultSet<DistanceType>& result, const ElementType* vec,int checks_num ) const
	{

		int stride = veclen_/part_m_;

		Heap<dist_index> index_dist(coarse_cluster_k_);
		std::vector<dist_index> queue(coarse_cluster_k_);

		for(int i=0;i<coarse_cluster_k_;i++){
			queue[i].index = i;
			queue[i].dis = distance_(coarse_index_[i]->pivot, vec, veclen_);
			index_dist.insert(queue[i]);
		}

		ElementType * restvec;
		std::vector<ElementType *> restsubvec(part_m_);

		restvec = new ElementType[veclen_];
		for(int i=0;i<part_m_;i++)
		{
			restsubvec[i] = new ElementType[stride];
		}

		std::vector<std::vector<DistanceType>> distance_lookup_table;
		distance_lookup_table.resize(part_m_);
		for(int i=0;i<part_m_;i++)
			distance_lookup_table[i].resize(cluster_k_);

		for(int i=0;i<checks_num;i++)
		{
			dist_index now;
			bool flag = index_dist.popMin(now);
			if(!flag)
				;
			int id = now.index;
			
			for(int j=0;j<veclen_;j++)
				restvec[j] = vec[j] - (coarse_index_[id]->pivot)[j];

			for(int j=0;j<part_m_;j++)
				getSubVector(j, restsubvec[j], restvec);

			for(int j=0;j<part_m_;j++)
				for(int k=0;k<cluster_k_;k++)
					distance_lookup_table[j][k] = distance_(restsubvec[j], (*lookup_table_[j])[k]->pivot, stride);

			for(int j=0;j<coarse_index_[id]->volume;j++)
			{
				int y_index = (coarse_index_[id]->container)[j];
				if (with_removed) {
					if (removed_points_.test(y_index)) continue;
				}	
				DistanceType d = 0;
				for(int k=0;k<part_m_;k++)
				{
					//d += distance_(((*lookup_table_[k])[part_indices_[y_index][k]])->pivot,subvec[k],stride);
					d += distance_lookup_table[k][part_indices_[y_index][k]];
				}
				result.addPoint(d,y_index);
			}
		}

		delete[] restvec;
		for(int i=0;i<part_m_;i++)
			delete[] restsubvec[i];

	}

	void loadSubVector(int id,int len,std::vector<ElementType *>& subMatrix,std::vector<ElementType *>& Matrix)
	{
	
		for(int i=0;i<len;i++)
			getSubVector(id,subMatrix[i],Matrix[i]);

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

	void getSubVector(int m,ElementType * subVector,const ElementType * Vector) const 
	{
		int stride = veclen_/part_m_;
		int first = m*stride;
		for(int i=0;i<stride;i++){
			subVector[i] = Vector[part_information_[i+first]];
		}
	}

	void subVector(int len,ElementType * result,const ElementType * d1,const ElementType * d2){
	
		for(int i=0;i<len;i++)
			result[i] = d1[i]-d2[i];
	
	}

	void swap(IVFADCIndex& other)
    {
    	std::swap(cluster_k, other.cluster_k);
    	std::swap(iterations_, other.iterations_);
    	std::swap(centers_init_, other.centers_init_);
    	std::swap(part_m_, other.part_m_);
    	std::swap(pool_, other.pool_);
    	std::swap(memoryCounter_, other.memoryCounter_);
    	std::swap(chooseCenters_, other.chooseCenters_);
 
		std::swap(rmatrix_, other.rmatrix_);
		//std::swap(lookup_table_, other.lookup_table_);

		std::swap(part_information_, other.part_information_);
		//std::swap(part_indices_, other.part_indices_);

		std::swap(coarse_cluster_k_, other.coarse_cluster_k_);

	}

private:

	PooledAllocator pool_;

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

	coarse_table coarse_index_;

	int coarse_cluster_k_;

	flann_centers_init_t centers_init_;

	USING_BASECLASS_SYMBOLS

};
	


}

#endif /*FLANN_IVFADC_INDEX_H*/