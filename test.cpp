#include "matrix.h"
#include "dmlpack.h"
#include "data_source.h"


#include <iostream>

void matrix_test()
{
	matrix<int> A(1,1, -2);	

	matrix<int> B(1,1, -9);	

	std::cout << A << B;

	matrix<int> C(A*B);

	std::cout << C << std::endl;	

	std::cout << (A * B) << std::endl;
	

}


int main()
{

	data_source a{};
	a.read_store_berkely_data(BRKLY_DATA::DIGIT , DATA_TYPE::TRAIN);

	matrix<int> data = a.get_features(); 	

	//std::cout << data.returnRow(5) << std::endl;
	//std::cout << data.returnRow(5000) << std::endl;

	
	matrix<int> label = a.get_labels();

	std::cout << label.returnRow(5) << std::endl;
	std::cout << label.returnRow(200) << std::endl;
	std::cout << label.returnRow(500) << std::endl;
		
	std::cout << "Mat SIZE row# : " << label.numRows() << std::endl;
	std::cout << "Mat SIZE col# : " << label.numCols() << std::endl;




	dmlpack<float> K{classifier_type::naive_bayes};
};



