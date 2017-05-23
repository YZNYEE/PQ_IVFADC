#ifndef FLANN_PQ_INDICES_H
#define FLANN_PQ_INDICES_H

#include "flann/algorithms/nn_index.h"
#include "flann/algorithms/all_indices.h"
#include "PQSingle_index.h"
#include "IVFADC_index.h"

template<typename Distance>
inline NNIndex<Distance>*
	create_PQ_index_by_mark(const int mark,
	const Matrix<typename Distance::ElementType>& dataset, const IndexParams& params, const Distance& distance)
{
	typedef typename Distance::ElementType ElementType;

	NNIndex<Distance>* nnIndex;

	if(mark == -100){
		nnIndex = create_index_<PQSingleIndex,Distance,ElementType>(dataset, params, distance);
	}
	else if(mark == -101){
		nnIndex = create_index_<IVFADCIndex,Distance,ElementType>(dataset, params, distance);
	}
	return nnIndex;
}

#endif /* FLANN_PQ_INDICES_H */