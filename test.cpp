#include <iostream>

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "dmlpack.h"
#include "matrix.hpp"
#include "data_source.h"


// For code timming.
#include <chrono>


void measure_exec_time(std::function<void(void)> lam)
{
	std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
	lam();
	std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
	std::cout << "time taken : " << duration << "\n";
}



TEST_CASE("testing the matrix class")
{
	SECTION("Create matrix ")
	{
		matrix<int> A(1,100);
		A.setAllNum(5);
		
		CHECK_THROWS( A(100,100));

		matrix<int> B(100,1);
		B.setAllNum(10);

		CHECK_THROWS( A + B );
		CHECK_THROWS( A - B );

		matrix<int> C(B * A);

		CHECK(C.size() == 100 * 100);

		CHECK_THROWS( C * A);

		matrix<int> D = C * (B * A);
		
		CHECK(D.size() == 100 * 100 );

		CHECK(D.isSquare());

		matrix<int> L(A*B);
		CHECK(L.size() == 1);


	}


	SECTION("test uniform rand fill")
	{
		matrix<int> rand_val(1,100);
		rand_val.randFillUniform(0,5);

		REQUIRE( rand_val.size() == 100);

		CHECK(rand_val.isRowVector());


		auto iter_beg = rand_val.begin();
		auto iter_end = rand_val.end();		

		for(auto itr = iter_beg ; itr != iter_end ; ++itr)
		{

			REQUIRE(*itr >= 0); 
		       	REQUIRE(*itr <= 5);
		}			

	}
	

	SECTION("test the linspace row and col")
	{
		matrix<double> K ; 
		K.resizeLinSpaceCol(1,100,0.1);		
		

	}

	
	SECTION("test the initialzier list ctor")
	{
		INFO("TESTING INIT LIST");
		matrix<double> K { 1 , 2 , 3 , 4 , 5 , 6};
		const int size = 6;
		CHECK(K.size() == size);
		CHECK_FALSE(K.isColVector());
		CHECK(K.isRowVector());

		SECTION("test vector size conversions")
		{
			K.resize(size , 1);
			CHECK(K.size() == size);

			CHECK_FALSE(K.isRowVector());
			CHECK(K.isColVector());
		}
	}
	
	SECTION("test scalar multiply")
	{

		measure_exec_time([]() ->void 
		{

			std::cout << __LINE__ << " : " ;
			matrix<double> K(100,100,5);

			K *= 0;
			CHECK(K.size() == 100*100);
			CHECK(K(1,1) == 0);
			K(1,1) = 5;	
			CHECK(K(1,1) == 5);
			K(1,1) *= 5;	
			CHECK(K(1,1) == 25);

		});

		measure_exec_time([]() ->void 
		{

			std::cout << __LINE__ << " : " ;
			matrix<double> K(100,100,5);
			K *= 0;

		});

		/* Profiling without the function call leads to 3 milisecond speed boost. 
			std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
			matrix<double> K(100,100,5);
				K *= 0;
			std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
			std::cout << "time taken : " << duration << std::endl;
		*/

	}

	SECTION("test : setAllZero func double ")
	{

		{
			matrix<double> K(100,100,5);

			measure_exec_time([&K]() ->void 
			{
				std::cout << __LINE__ << " : " ;
				K *= 0;
			});

			CHECK(K(20,20) == 0);

			measure_exec_time([&K]() ->void 
			{
				std::cout << __LINE__ << " : " ;
				K.setAllNum(5);
			});


			CHECK(K(20,20) == 5);

			measure_exec_time([&K]() ->void 
			{
				std::cout << __LINE__ << " : " ;
				K.setAllZero();
			});

			CHECK(K(20,20) == 0);
		}



	}

	SECTION("test : setAllZero func int ")
	{

		{
			matrix<int> K(101,103,5);

			measure_exec_time([&K]() ->void 
			{
				std::cout << __LINE__ << " : " ;
				K *= 0;
			});

			CHECK(K(20,20) == 0);

			measure_exec_time([&K]() ->void 
			{
				std::cout << __LINE__ << " : " ;
				K.setAllNum(5);
			});


			CHECK(K(20,20) == 5);

			measure_exec_time([&K]() ->void 
			{
				std::cout << __LINE__ << " : " ;
				K.setAllZero();
			});

			CHECK(K(20,20) == 0);
			CHECK(K(101,103) == 0);
			CHECK(K(101,102) == 0);
			CHECK(K(101,101) == 0);
			CHECK(K(101,99) == 0);
			CHECK(K(100,102) == 0);
			CHECK(K(100,102) == 0);
		}


	}


	SECTION("test : setAllZero func float")
	{

		{
			matrix<float> K(101,103,5);

			measure_exec_time([&K]() ->void 
			{
				std::cout << __LINE__ << " : " ;
				K *= 0;
			});

			CHECK(K(20,20) == 0);

			measure_exec_time([&K]() ->void 
			{
				std::cout << __LINE__ << " : " ;
				K.setAllNum(5);
			});


			CHECK(K(20,20) == 5);

			measure_exec_time([&K]() ->void 
			{
				std::cout << __LINE__ << " : " ;
				K.setAllZero();
			});

			CHECK(K(20,20) == 0);


			CHECK(K(20,20) == 0);
			CHECK(K(101,103) == 0);
			CHECK(K(101,102) == 0);
			CHECK(K(101,101) == 0);
			CHECK(K(101,99) == 0);
			CHECK(K(100,102) == 0);
			CHECK(K(100,102) == 0);

		}


	}


	SECTION("check allignment of std::vector memory")
	{
		matrix<int> K(97,97,5);
		
		CHECK(K.check_sse_allignment());

	}

}





TEST_CASE("testing different parts of dmlpack", "[dmlpack]")
{
	SECTION("testing using berkely data set")
	{
		SECTION("test using faces data set")
		{
			data_source data{};
			const BRKLY_DATA data_set = BRKLY_DATA::FACE;
			data.read_store_berkely_data( data_set , DATA_TYPE::TRAIN);

			matrix<double> featu = data.get_features();
			REQUIRE( featu(1,1) == 0 );

			matrix<double> label = data.get_labels();
			REQUIRE( label.numRows() == featu.numRows() );

			SECTION("train using data set")
			{
				dmlpack<double> K{CLASSIFIER_TYPE::PERCEPTRON};
				K.feed_train_data(std::move(featu) , std::move(label));
				K.train();

				
				SECTION("validate using faces")
				{

					data_source digit_valid{};
					digit_valid.read_store_berkely_data(data_set , DATA_TYPE::VALID);

					matrix<double> data_valid = digit_valid.get_features(); 	
					matrix<double> label_valid = digit_valid.get_labels();

					K.feed_test_data(std::move(data_valid), std::move(label_valid));
					auto res = K.inference();

					K.test_accuracy();

				}


			}



		}

	}	

}

