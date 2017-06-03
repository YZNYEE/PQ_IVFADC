#ifndef FLANN_PARAMS_PQ_H
#define FLANN_PARAMS_PQ_H

#include "flann/util/params.h"

namespace flann
{

struct PQSearchParams : public SearchParams
{
	PQSearchParams(int checks_num = 8,int smethod = 1,int checks_ = 32, float eps_ = 0.0, bool sorted_ = true):
		SearchParams(checks_,eps_,sorted),checks_coarse_cluster(checks_num),searchMethod(smethod){}
	int checks_coarse_cluster;
	int searchMethod;
};

}

#endif	//FLANN_PARAMS_PQ_H