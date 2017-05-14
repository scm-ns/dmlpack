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

}
