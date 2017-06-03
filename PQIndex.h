
#ifndef FLANN_PQINDEX
#define FLANN_PQINDEX

#include <vector>
#include <string>
#include <cassert>
#include <cstdio>

#include "flann/general.h"
#include "flann/util/matrix.h"
#include "flann/util/params.h"
#include "flann/util/saving.h"

#include "flann/algorithms/all_indices.h"
#include "PQ_indices.h"
#include "PQSingle_index.h"
#include "IVFADC_index.h"

namespace flann
{

template<typename Distance>
class PQIndex
{
public:
    typedef typename Distance::ElementType ElementType;
    typedef typename Distance::ResultType DistanceType;
    typedef NNIndex<Distance> IndexType;

    PQIndex(const IndexParams& params, Distance distance = Distance() )
        : index_params_(params)
    {
        int mark = get_param<int>(params,"algorithm");
        loaded_ = false;

        Matrix<ElementType> features;
        if (mark == FLANN_INDEX_SAVED) {
            nnIndex_ = load_saved_index(features, get_param<std::string>(params,"filename"), distance);
            loaded_ = true;
        }
        else {
        	int mark = get_param<int>(params, "algorithm");
            nnIndex_ = create_PQ_index_by_mark<Distance>(mark, features, params, distance);
        }
    }


    PQIndex(const Matrix<ElementType>& features, const IndexParams& params, Distance distance = Distance() )
        : index_params_(params)
    {
        int mark = get_param<int>(params,"algorithm");
        loaded_ = false;
		//std::cout<<"inmark "<<mark<<std::endl;
        if (mark == FLANN_INDEX_SAVED) {
            nnIndex_ = load_saved_index(features, get_param<std::string>(params,"filename"), distance);
            loaded_ = true;
        }
        else {
			//int index = get_param<int>(params, "algorithm");
			//std::cout<<"creating PQ index"<<std::endl;
            nnIndex_ = create_PQ_index_by_mark<Distance>(mark, features, params, distance);
        }
    }


    PQIndex(const PQIndex& other) : loaded_(other.loaded_), index_params_(other.index_params_)
    {
    	nnIndex_ = other.nnIndex_->clone();
    }

    PQIndex& operator=(PQIndex other)
    {
    	this->swap(other);
    	return *this;
    }

    virtual ~PQIndex()
    {
        delete nnIndex_;
    }

    /**
     * Builds the index.
     */
    void buildIndex()
    {
        if (!loaded_) {
			//std::cout<<"buildindex"<<std::endl;
            nnIndex_->buildIndex();
        }
    }

	void getLookupTable(int id,std::vector<double *> &centers)
	{
		int mark = get_param<int>(index_params_,"algorithm");
		if(mark == -100)
			((PQSingleIndex<Distance> *)nnIndex_)->getLookupTable(id, centers);
		if(mark == -101)
			((IVFADCIndex<Distance> *)nnIndex_)->getLookupTable(id, centers);
	}

	void getIndices(int id,std::vector<int> &indices)
	{
		int mark = get_param<int>(index_params_,"algorithm");
		if(mark == -100)
			((PQSingleIndex<Distance> *)nnIndex_)->getIndices(id,indices);
		if(mark == -101)
			((IVFADCIndex<Distance> *)nnIndex_)->getIndices(id,indices);
	}

	void getFeature(int id,std::vector<double *> & feature)
	{
		int mark = get_param<int>(index_params_,"algorithm");
		if(mark == -100)
			((PQSingleIndex<Distance> *)nnIndex_)->getFeature(id,feature);
		if(mark == -101)
			((IVFADCIndex<Distance> *)nnIndex_)->getFeature(id,feature);
	}

    void buildIndex(const Matrix<ElementType>& points)
    {
    	nnIndex_->buildIndex(points);
    }

	int getStride() const
	{
		int mark = get_param<int>(index_params_,"algorithm");
		if(mark == -100)
			return ((PQSingleIndex<Distance> *)nnIndex_)->getStride();
		if(mark == -101)
			((IVFADCIndex<Distance> *)nnIndex_)->getStride();
	}

	void getCoarseCenter(std::vector<double *> & ccenter)
	{
		((IVFADCIndex<Distance> *)nnIndex_)->getCoarseCenter(ccenter);
	}

	void getAllFeature(std::vector<double *> & all)
	{
		((IVFADCIndex<Distance> *)nnIndex_)->getAllFeature(all);
	}

	void getCoarseIndices(std::vector<int> & cindices)
	{
		((IVFADCIndex<Distance> *)nnIndex_)->getCoarseIndices(cindices);
	}

    void addPoints(const Matrix<ElementType>& points, float rebuild_threshold = 2)
    {
        nnIndex_->addPoints(points, rebuild_threshold);
    }

    /**
     * Remove point from the index
     * @param index Index of point to be removed
     */
    void removePoint(size_t point_id)
    {
    	nnIndex_->removePoint(point_id);
    }

    /**
     * Returns pointer to a data point with the specified id.
     * @param point_id the id of point to retrieve
     * @return
     */
    ElementType* getPoint(size_t point_id)
    {
    	return nnIndex_->getPoint(point_id);
    }

    /**
     * Save index to file
     * @param filename
     */
    void save(std::string filename)
    {
        FILE* fout = fopen(filename.c_str(), "wb");
        if (fout == NULL) {
            throw FLANNException("Cannot open file");
        }
        nnIndex_->saveIndex(fout);
        fclose(fout);
    }

	void load(std::string filename)
	{
		FILE* fout = fopen(filename.c_str(), "rb");
        if (fout == NULL) {
            throw FLANNException("Cannot open file");
        }
        nnIndex_->loadIndex(fout);
        fclose(fout);
	}

    /**
     * \returns number of features in this index.
     */
    size_t veclen() const
    {
        return nnIndex_->veclen();
    }

    /**
     * \returns The dimensionality of the features in this index.
     */
    size_t size() const
    {
        return nnIndex_->size();
    }

    /**
     * \returns The index type (kdtree, kmeans,...)
     */
    int getType() const
    {
        return nnIndex_->getType_PQ();
    }

    /**
     * \returns The amount of memory (in bytes) used by the index.
     */
    int usedMemory() const
    {
        return nnIndex_->usedMemory();
    }


    /**
     * \returns The index parameters
     */
    IndexParams getParameters() const
    {
        return nnIndex_->getParameters();
    }

    /**
     * \brief Perform k-nearest neighbor search
     * \param[in] queries The query points for which to find the nearest neighbors
     * \param[out] indices The indices of the nearest neighbors found
     * \param[out] dists Distances to the nearest neighbors found
     * \param[in] knn Number of nearest neighbors to return
     * \param[in] params Search parameters
     */
    int knnSearch(const Matrix<ElementType>& queries,
                                 Matrix<size_t>& indices,
                                 Matrix<DistanceType>& dists,
                                 size_t knn,
                           const SearchParams& params) const
    {
    	return nnIndex_->knnSearch(queries, indices, dists, knn, params);
    }

    /**
     *
     * @param queries
     * @param indices
     * @param dists
     * @param knn
     * @param params
     * @return
     */
    int knnSearch(const Matrix<ElementType>& queries,
                                 Matrix<int>& indices,
                                 Matrix<DistanceType>& dists,
                                 size_t knn,
                           const SearchParams& params) const
    {
    	return nnIndex_->knnSearch(queries, indices, dists, knn, params);
    }

    /**
     * \brief Perform k-nearest neighbor search
     * \param[in] queries The query points for which to find the nearest neighbors
     * \param[out] indices The indices of the nearest neighbors found
     * \param[out] dists Distances to the nearest neighbors found
     * \param[in] knn Number of nearest neighbors to return
     * \param[in] params Search parameters
     */
    int knnSearch(const Matrix<ElementType>& queries,
                                 std::vector< std::vector<size_t> >& indices,
                                 std::vector<std::vector<DistanceType> >& dists,
                                 size_t knn,
                           const SearchParams& params)
    {
    	return nnIndex_->knnSearch(queries, indices, dists, knn, params);
    }

    /**
     *
     * @param queries
     * @param indices
     * @param dists
     * @param knn
     * @param params
     * @return
     */
    int knnSearch(const Matrix<ElementType>& queries,
                                 std::vector< std::vector<int> >& indices,
                                 std::vector<std::vector<DistanceType> >& dists,
                                 size_t knn,
                           const SearchParams& params) const
    {
    	return nnIndex_->knnSearch(queries, indices, dists, knn, params);
    }

    /**
     * \brief Perform radius search
     * \param[in] queries The query points
     * \param[out] indices The indinces of the neighbors found within the given radius
     * \param[out] dists The distances to the nearest neighbors found
     * \param[in] radius The radius used for search
     * \param[in] params Search parameters
     * \returns Number of neighbors found
     */
    int radiusSearch(const Matrix<ElementType>& queries,
                                    Matrix<size_t>& indices,
                                    Matrix<DistanceType>& dists,
                                    float radius,
                              const SearchParams& params) const
    {
    	return nnIndex_->radiusSearch(queries, indices, dists, radius, params);
    }

    /**
     *
     * @param queries
     * @param indices
     * @param dists
     * @param radius
     * @param params
     * @return
     */
    int radiusSearch(const Matrix<ElementType>& queries,
                                    Matrix<int>& indices,
                                    Matrix<DistanceType>& dists,
                                    float radius,
                              const SearchParams& params) const
    {
    	return nnIndex_->radiusSearch(queries, indices, dists, radius, params);
    }

    /**
     * \brief Perform radius search
     * \param[in] queries The query points
     * \param[out] indices The indinces of the neighbors found within the given radius
     * \param[out] dists The distances to the nearest neighbors found
     * \param[in] radius The radius used for search
     * \param[in] params Search parameters
     * \returns Number of neighbors found
     */
    int radiusSearch(const Matrix<ElementType>& queries,
                                    std::vector< std::vector<size_t> >& indices,
                                    std::vector<std::vector<DistanceType> >& dists,
                                    float radius,
                              const SearchParams& params) const
    {
    	return nnIndex_->radiusSearch(queries, indices, dists, radius, params);
    }

    /**
     *
     * @param queries
     * @param indices
     * @param dists
     * @param radius
     * @param params
     * @return
     */
    int radiusSearch(const Matrix<ElementType>& queries,
                                    std::vector< std::vector<int> >& indices,
                                    std::vector<std::vector<DistanceType> >& dists,
                                    float radius,
                              const SearchParams& params) const
    {
    	return nnIndex_->radiusSearch(queries, indices, dists, radius, params);
    }

private:
    IndexType* load_saved_index(const Matrix<ElementType>& dataset, const std::string& filename, Distance distance)
    {
        FILE* fin = fopen(filename.c_str(), "rb");
        if (fin == NULL) {
            return NULL;
        }
        IndexHeader header = load_header(fin);
        if (header.data_type != flann_datatype_value<ElementType>::value) {
            throw FLANNException("Datatype of saved index is different than of the one to be created.");
        }

        IndexParams params;
        params["algorithm"] = header.index_type;
        IndexType* nnIndex = create_PQ_index_by_mark<Distance>(header.index_type, dataset, params, distance);
        rewind(fin);
        nnIndex->loadIndex(fin);
        fclose(fin);

        return nnIndex;
    }

    void swap( PQIndex& other)
    {
    	std::swap(nnIndex_, other.nnIndex_);
    	std::swap(loaded_, other.loaded_);
    	std::swap(index_params_, other.index_params_);
    }

private:
    /** Pointer to actual index class */
    IndexType* nnIndex_;
    /** Indices if the index was loaded from a file */
    bool loaded_;
    /** Parameters passed to the index */
    IndexParams index_params_;
};

}

#endif