#include "matrix.h"
#include "dmlpack.h"

#include <iostream>


int main()
{
	matrix<int> A(1,1, -2);	

	matrix<int> B(1,1, -9);	

	std::cout << A << B;

	matrix<int> C(A*B);

	std::cout << C << std::endl;	

	std::cout << (A * B) << std::endl;
	
	dmlpack<float> K{classifier_type::naive_bayes};
};

