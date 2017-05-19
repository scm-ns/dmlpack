// used to test the dmlpack collection of classes

#include <iostream>

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "dmlpack.hpp"
#include "single_layer_nn.hpp"
#include "matrix.hpp"
#include "data_source.h"


// For code timming.
#include <chrono>


void measure_exec_time(std::function<void(void)> lam)
{
	std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
	lam();
	std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

	std::cout << "time taken : " << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count()  << " ns || "
		  << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()  << " ms || "
		  << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()  << " mils || "
		  << std::endl;
}



TEST_CASE("testing the dmlpack::single_layer_nn")
{

	auto test1 =  "Create single_layer_nn ";
	SECTION(test1)
	{

		measure_exec_time([&]() ->void 
		{
			dmlpack::single_layer_nn<int>( 1000 , 0.001);
		});
	}


	auto test2  = " try to approximate xor funtion ";
	SECTION(test2)
	{
		// create artifical data set xor
		matrix_op::matrix<double> train_x(4,2,0); // 4 data point each with 2 features
		matrix_op::matrix<double> train_y(4,1,0); // 4 data point each with 1 class
		// 0 ^ 0 = 0 
		train_x(1,1)  = 0 ;
		train_x(1,2)  = 0 ;
		train_y(1,1) = 0 ;

		// 0 ^ 1 = 1
		train_x(2,1)  = 0 ;
		train_x(2,2)  = 1 ;
		train_y(2,1) =  1 ;

		// 1 ^ 0 = 1
		train_x(3,1)  = 1 ;
		train_x(3,2)  = 0 ;
		train_y(3,1) =  1 ;

		// 1 ^ 1 = 0 
		train_x(4,1)  = 1 ;
		train_x(4,2)  = 1 ;
		train_y(4,1) =  0 ;

		matrix_op::matrix<double> feature = { 0 , 0 };

		measure_exec_time([&]() ->void 
		{
			dmlpack::single_layer_nn<double> nn( 10000 , 0.001 );
			nn.set_training_data(train_x , train_y);
			nn.train();
			
			std::cout << "res : " << nn.infer_single_feature(feature) << std::endl;

		});

		
	}

}


