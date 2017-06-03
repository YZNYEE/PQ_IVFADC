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

    load_from_file(dataset, "data.hdf5","dataset");
    load_from_file(query, "data.hdf5","query");

	std::cout<<dataset.cols<<" "<<dataset.rows<<std::endl;
	std::cout<<query.cols<<" "<<query.rows<<std::endl;

    Matrix<int> indices(new int[query.rows*nn], query.rows, nn);
	Matrix<float> dists(new float[query.rows*nn], query.rows, nn);

	flann::IVFADCIndexParams IVFIP(256,8,0,12,128);

	PQIndex<L2<float>> index(dataset, IVFIP);

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