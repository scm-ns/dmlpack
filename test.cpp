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



	auto feed_forward= "create a feed forward network";
	SECTION(feed_forward)
	{
		auto x = matrix_op::rand_fill<float>(3, 1, 0 , 1);// 3,1
		auto w1 = matrix_op::rand_fill<float>(4,3,0,1);   // 4 , 3
		auto b1 = matrix_op::rand_fill<float>(4,1,0,1);   // 4 , 1
		auto h1 = matrix_op::sigmoid(w1.mul(x) + b1); // 4,1
		
		auto w2 = matrix_op::rand_fill<float>(4,4,0,1); // 4 , 4
		auto b2 = matrix_op::rand_fill<float>(4,1,0,1); // 4 , 1
		auto h2 = matrix_op::sigmoid(w2.mul(h1) + b2);  // 4 , 1

		auto w3 = matrix_op::rand_fill<float>(4,4,0,1); // 1 , 4
		auto b3 = matrix_op::rand_fill<float>(4,1,0,1); // 1 , 1
		auto out =  matrix_op::sigmoid(w3.mul(h2) + b3);
		
		std::cout << out ;

	}





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
			
			auto res = (weights.mul(inputs)) + bias;

			auto firing = matrix_op::sigmoid(res);
			std::cout << firing << std::endl; 
		}


		{
			auto dims = 10;
			auto inputs = matrix_op::rand_fill<float>(dims , 1 , 0 , 10);
			auto weights = matrix_op::rand_fill<float>(1 , dims , 0 , 10);
			float bias = 0.5;		
	
/* op* does not work, I need to complete its implementation. For now use mul
 *				 std::cout << inputs
				  << weights 
				  << inputs.mul(weights)
				  << weights.mul(inputs);
*/
			auto res = (inputs.mul(weights)) + bias;
			auto firing = matrix_op::sigmoid(res);

			firing.for_each([](int r, int c , float val )
			{
				CHECK(val >= 0); CHECK( val <= 1);
			});	

		}

	}

}
