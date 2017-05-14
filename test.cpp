#include <iostream>

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "dmlpack.hpp"
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



TEST_CASE("testing the matrix class")
{
	using namespace matrix_op;

	auto test_simple_neuron = "create a simple neuron func ";
	SECTION(test_simple_neuron)
	{
		{
			auto dims = 100;
			auto inputs = matrix_op::rand_fill<float>(dims , 1 , 0 , 10);
			auto weights = matrix_op::rand_fill<float>(1 , dims , 0 , 10);
			float bias = 0.5;		
			
			auto res = (inputs * weights) + bias;

			auto firing = matrix_op::exp(res);
		}


		{
			auto dims = 100;
			auto inputs = matrix_op::rand_fill<float>(dims , 1 , 0 , 10);
			auto weights = matrix_op::rand_fill<float>(1 , dims , 0 , 10);
			float bias = 0.5;		
			
			auto res = (weights * inputs) + bias;

			auto firing = matrix_op::exp(res);
			std::cout << firing << std::endl; // produces inf
		}


		{
			auto dims = 10;
			auto inputs = matrix_op::rand_fill<float>(dims , 1 , 0 , 1);
			auto weights = matrix_op::rand_fill<float>(1 , dims , 0 , 1);
			float bias = 0.5;		
			
			auto res = (weights * inputs) + bias;

			auto firing = matrix_op::exp(res);
			std::cout << firing << std::endl;
		}

		
		{
			auto dims = 100;
			auto inputs = matrix_op::rand_fill<float>(dims , 1 , 0 , 10);
			auto weights = matrix_op::rand_fill<float>(1 , dims , 0 , 10);
			float bias = 0.5;		
			
			auto res = (weights * inputs) + bias;

			auto firing = matrix_op::sigmoid(res);
			std::cout << firing << std::endl; 
		}


		{
			auto dims = 100;
			auto inputs = matrix_op::rand_fill<float>(dims , 1 , 0 , 10);
			auto weights = matrix_op::rand_fill<float>(1 , dims , 0 , 10);
			float bias = 0.5;		
			
			auto res = (inputs * weights) + bias;

			auto firing = matrix_op::sigmoid(res);
			firing.for_each([](int r, int c , float val )
			{
				std::cout << r << c << std::endl;
				CHECK(val >= 0); CHECK( val <= 1);
			});	

		}

	}

}
