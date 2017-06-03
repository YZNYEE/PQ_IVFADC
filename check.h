/*

this is a check unit to check the clustering centers

*/

#include"flann/util/matrix.h"
#include<vector>
#include<cmath>

namespace flann{

	template<typename T>
	void show_vector(std::vector<T*> & m,int len);

	bool checks_clustering_centers(std::vector<double*> &centers,
									std::vector<double*> &feature,
									std::vector<int> &indices,int len)
	{
		for(int i=0;i<indices.size();i++)
		{
			int pos = 0;
			double dist = 0;
			for(int j=0;j<len;j++)
			{
				double sub = (centers[0][j]-feature[i][j]);
				dist += sub*sub;
			}

			//std::cout<<i<<" th"<<" pos: "<<indices[i]<<std::endl;
			//std::cout<<0<<" dist: "<<dist<<std::endl;

			for(int k=1;k<centers.size();k++)
			{
				double subdist = 0;
				for(int j=0;j<len;j++)
				{
					double sub = centers[k][j] - feature[i][j];
					subdist += sub*sub;
				}
				//std::cout<<k<<" dist: "<<subdist<<std::endl;
				if(subdist < dist)
				{
					dist = subdist;
					pos = k;
				}
			}

			//std::cout<<"new pos: "<<pos<<std::endl;
			if(pos != indices[i])
				return false;
		}
		return true;
	}

	bool is_perfect_cluster(std::vector<double*> &centers,
									std::vector<double*> &feature,
									std::vector<int> &indices,int len)
	{
		std::vector<double*> dcenters(centers.size());
		for(int i=0;i<centers.size();i++)
		{
			dcenters[i] = new double[len];
			memset(dcenters[i],0,sizeof(double)*len);
		}

		std::vector<int> count(dcenters.size(),0);

		for(int i=0;i<indices.size();i++)
		{
			int belong = indices[i];
			count[belong]++;

			for(int j=0;j<len;j++)
			{
				dcenters[belong][j] += feature[i][j];
			}
		}

		for(int i=0;i<centers.size();i++){
			double factor = 1.0/count[i];
			for(int j=0;j<len;j++)
				dcenters[i][j] *= factor;
		}

		
		//show_vector(centers, len);
		//std::cout<<"~~~~~~~~~~~~~~~~~~~~~~~~~`"<<std::endl;
		//show_vector(dcenters, len);

		for(int i=0;i<centers.size();i++)
			for(int j=0;j<len;j++)
				if(abs(dcenters[i][j]-centers[i][j])>1e-5)
					return false;

		for(int i=0;i<centers.size();i++)
			delete[i] dcenters[i];

		return true;

	}

	template<typename T>
	void show_matrix(Matrix<T> & m)
	{
		for(int i=0;i<m.rows;i++)
		{
			for(int j=0;j<m.cols;j++)
				std::cout<<m[i][j]<<' ';
			std::cout<<std::endl;
		}
	}

	template<typename T>
	void show_vector(std::vector<T*> & m,int len)
	{
		for(int i=0;i<m.size();i++)
		{
			for(int j=0;j<len;j++)
				std::cout<<m[i][j]<<' ';
			std::cout<<std::endl;
		}
	}

}