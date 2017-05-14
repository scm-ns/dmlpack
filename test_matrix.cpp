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

	auto test1 =  "Create matrix ";
	SECTION(test1)
	{
		std::cout << test1 << std::endl;
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

	auto test2 ="print matrix";
	SECTION(test2)
	{
		std::cout << test2 << std::endl;
		matrix<int> A(10 , 10);
		std::cout << A ;		
	}
	auto test3 ="test uniform rand fill" ;
	SECTION(test3)
	{
		std::cout << test3 << std::endl;
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
	
	auto test4 = "test the linspace row and col";
	SECTION(test4)
	{
		std::cout << test4 << std::endl;
		matrix<double> K ; 
		K.resizeLinSpaceCol(1,100,0.1);		
		

	}

 	auto test5 = "test the initialzier list ctor";
	SECTION(test5)
	{
		std::cout << test5 << std::endl;
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
	
	auto test6 = "test scalar multiply";
	SECTION(test6)
	{
		std::cout << test6 << std::endl;
		SCOPED_INFO("TESTING INIT LIST");
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
	auto test7 = "test : setAllZero func double ";
	SECTION(test7)
	{
		std::cout << test7 << std::endl;
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
	
	auto test8 = "test : setAllZero func int ";
	SECTION(test8)
	{
		std::cout << test8 << std::endl;
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

	auto test9 = "test : setAllZero func float";
	SECTION(test9)
	{
		std::cout << test9 << std::endl;
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

			measure_exec_time([&K]() ->void 
			{
				std::cout << __LINE__ << " : " ;
				K.setAllNum(5);
			});

			CHECK(K(20,20) == 5);

			measure_exec_time([&K]() ->void 
			{
				std::cout << __LINE__ << " : " ;
				K.setAllNum(0);
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

	auto test10 = "check allignment of std::vector memory";
	SECTION(test10)
	{
		matrix<int> K(97,97,5);
		std::cout << "SSE ALLIGNMENT " << K.check_sse_allignment() << "\n";
		CHECK(K.check_sse_allignment());

	}

	auto test11 = "check if new operator+ works ";
	SECTION(test11)
	{
		matrix<int> K(97,97,5);
		matrix<int> P(97,97,-5);
		auto L = K + P;

	 	CHECK(L(20,20) == 0);	
	 	CHECK(L(70,80) == 0);	
	 	CHECK(L(10,50) == 0);	
	 	CHECK(L(90,30) == 0);	
	}



	auto test12 = "check new operator+ vs old speed";
	SECTION(test12)
	{
		std::cout << test12 << std::endl;

		{
			matrix<int> K(97,97,5);
			matrix<int> P(97,97,-5);
			matrix<int> L;

			measure_exec_time([&]() ->void 
			{
				std::cout << __LINE__ << " : " ;
				L = K + P;
			});

			CHECK(L(20,20) == 0);	
			CHECK(L(70,80) == 0);	
			CHECK(L(10,50) == 0);	
			CHECK(L(90,30) == 0);	

			
		}

		{
			matrix<int> K(97,97,5);
			matrix<int> P(97,97,-5);
			matrix<int> L;

			measure_exec_time([&]() ->void 
			{
				std::cout << __LINE__ << " : " ;
				L = K.add(P);
			});

			CHECK(L(20,20) == 0);	
			CHECK(L(70,80) == 0);	
			CHECK(L(10,50) == 0);	
			CHECK(L(90,30) == 0);	

			
		}



	}

	auto test13 = "check new see op+ vs old";
	SECTION(test13)
	{
		std::cout << test13 << std::endl;

		{
			matrix<int> K(97,97,5);
			matrix<int> P(97,97,-5);
			matrix<int> L;

			measure_exec_time([&]() ->void 
			{
				std::cout << __LINE__ << " : " ;
				L = K * P;
			});
			
		}

		{
			matrix<int> K(97,97,5);
			matrix<int> P(97,97,-5);
			matrix<int> L;

			measure_exec_time([&]() ->void 
			{
				std::cout << __LINE__ << " : " ;
				L = K.mul(P);
			});

			
		}
		
		{
			matrix<int> K(97,97,5);
			matrix<int> P(97,97,-5);
			matrix<int> L;

			measure_exec_time([&]() ->void 
			{
				std::cout << __LINE__ << " : " ;
				L = K.mul_iter(P);
			});

			
		}


		{
			matrix<int> K(9,9,5);
			matrix<int> P(9,9,-5);
			matrix<int> L;

			measure_exec_time([&]() ->void 
			{
				std::cout << __LINE__ << " : " ;
				L = K * P;
			});
			
		}

		{
			matrix<int> K(9,9,5);
			matrix<int> P(9,9,-5);
			matrix<int> L;

			measure_exec_time([&]() ->void 
			{
				std::cout << __LINE__ << " : " ;
				L = K.mul(P);
			});

			
		}

		{
			matrix<int> K(9,9,5);
			matrix<int> P(9,9,-5);
			matrix<int> L;

			measure_exec_time([&]() ->void 
			{
				std::cout << __LINE__ << " : " ;
				L = K.mul_iter(P);
			});

			
		}


		{
			matrix<float> K(97,97,5);
			matrix<float> P(97,97,-5);
			matrix<float> L;

			measure_exec_time([&]() ->void 
			{
				std::cout << __LINE__ << " : " ;
				L = K * P;
			});
			
		}

		{
			matrix<float> K(97,97,5);
			matrix<float> P(97,97,-5);
			matrix<float> L;

			measure_exec_time([&]() ->void 
			{
				std::cout << __LINE__ << " : " ;
				L = K.mul(P);
			});

			
		}
		
		{
			matrix<float> K(97,97,5);
			matrix<float> P(97,97,-5);
			matrix<float> L;

			measure_exec_time([&]() ->void 
			{
				std::cout << __LINE__ << " : " ;
				L = K.mul_iter(P);
			});

			
		}


		{
			matrix<float> K(9,9,5);
			matrix<float> P(9,9,-5);
			matrix<float> L;

			measure_exec_time([&]() ->void 
			{
			std::cout << __LINE__ << " : " ;
				L = K * P;
			});
			
		}

		{
			matrix<float> K(9,9,5);
			matrix<float> P(9,9,-5);
			matrix<float> L;

			measure_exec_time([&]() ->void 
			{
				std::cout << __LINE__ << " : " ;
				L = K.mul(P);
			});

		}

		{
			matrix<float> K(9,9,5);
			matrix<float> P(9,9,-5);
			matrix<float> L;

			measure_exec_time([&]() ->void 
			{
				std::cout << __LINE__ << " : " ;
				L = K.mul_iter(P);
			});

			
		}



	}

	auto test_op_plus = "check if scalar perator+ works ";
	SECTION(test_op_plus)
	{
		std::cout << test_op_plus << std::endl;
		{
			matrix<int> K(97,97,5);
			int P = 10;
			auto L = K + P;

			CHECK(L(20,20) == 15);	
			CHECK(L(70,80) == 15);	
			CHECK(L(10,50) == 15);	
			CHECK(L(90,30) == 15);	
		}


		{
			matrix<float> K(1,1,5);
			float P = 10;
			auto L = K + P;
			CHECK(L(1,1) == 15);
		}

	}




}


TEST_CASE("testing the matrix_op namepsace ")
{

	auto test1 = "check new namepsace ";
	SECTION(test1)
	{
		std::cout << test1 << std::endl;
		matrix_op::matrix<double> mat = matrix_op::linspace_row<double>(0 , 5 , 1);
		std::cout << mat << std::endl;

	}



	auto test_sin = "test sin";
	SECTION(test_sin)
	{
		{
			int num_cols = 100000; 
			matrix_op::matrix<float> L; 
			L = matrix_op::rand_fill<float>(1 , num_cols , 1 ,10);	// row vector
			std::cout << __LINE__ << " : " ;
			measure_exec_time([&]() ->void 
			{
				L = matrix_op::sin<float>(L);
				L = matrix_op::cos<float>(L);
				L = matrix_op::tanh<float>(L);
				L = matrix_op::sigmoid<float>(L);
				L = matrix_op::relu<float>(L);
				L = matrix_op::leaky_relu<float>(L);
				L = matrix_op::leaky_relu<float>(L, 0.1);
			});
		}


		{
			int num_cols = 100000; 
			matrix_op::matrix<float> L; 
			L = matrix_op::rand_fill<float>(1 , num_cols , 1 ,10);	// row vector
			std::cout << __LINE__ << " : " ;
			measure_exec_time([&]() ->void 
			{
				L = matrix_op::sin<float>(L);
				L = matrix_op::cos<float>(L);
				L = matrix_op::tanh<float>(L);
				L = matrix_op::sigmoid<float>(L);
				L = matrix_op::relu<float>(L);
				L = matrix_op::leaky_relu<float>(L);
			});
		}


		{
			int num_cols = 1000; 
			int num_rows = 1000; 
			matrix_op::matrix<float> L; 
			L = matrix_op::rand_fill<float>(num_rows , num_cols , 1 ,10);	// row vector
			std::cout << __LINE__ << " : " ;
			measure_exec_time([&]() ->void 
			{
				L = matrix_op::sin<float>(L);
				L = matrix_op::cos<float>(L);
				L = matrix_op::tanh<float>(L);
				L = matrix_op::sigmoid<float>(L);
				L = matrix_op::relu<float>(L);
				L = matrix_op::leaky_relu<float>(L);
			});
		}

		{
			float L = 0.5;
			std::cout << __LINE__ << " : " ;
			measure_exec_time([&]() ->void 
			{
				L = matrix_op::sin<float>(L);
				L = matrix_op::cos<float>(L);
				L = matrix_op::tanh<float>(L);
				L = matrix_op::sigmoid<float>(L);
				L = matrix_op::relu<float>(L);
			});
		}



	}	


	auto test_sum = "test sum";
	SECTION(test_sum)
	{
		{
			int num_cols = 100000; 
			matrix_op::matrix<float> L(1,num_cols,5); // row vector
			
			long result = 0;
			measure_exec_time([&]() ->void 
			{
					result = matrix_op::sum(L);	
					std::cout << __LINE__ << " : " ;
			});			
			CHECK(result == (5 * num_cols));

		}



		{
			int num_rows = 100000; 
			matrix_op::matrix<float> L(num_rows,1,5); // row vector
			
			long result = 0;
			measure_exec_time([&]() ->void 
			{
					result = matrix_op::sum(L);	
					std::cout << __LINE__ << " : " ;
			});			
			CHECK(result == (5 * num_rows));

		}



		{

			int num_cols = 10000; 
			int num_rows = 10000; 
			matrix_op::matrix<long long> L(num_rows,num_cols,5); // row vector
			
			long result = 0;
			measure_exec_time([&]() ->void 
			{
					result = matrix_op::sum(L);	
					std::cout << __LINE__ << " : " ;
			});			
			CHECK(result == (5 * num_rows *  num_cols) );
		}



	}





	
	auto test_rand_fill = "check rand_fill namepsace ";
	SECTION(test_rand_fill)
	{
		std::cout << test_rand_fill << std::endl;

		{
			matrix_op::matrix<double> L;

			measure_exec_time([&]() ->void 
			{
					L = matrix_op::rand_fill<double>(100 , 100 , 0 , 1);
					std::cout << __LINE__ << " : " ;
			});			


			CHECK( (L(20,20) >= 0 && L(20,20) <= 1) );
			CHECK( (L(90,20) >= 0 && L(90,20) <= 1) );
			CHECK( (L(80,30) >= 0 && L(80,30) <= 1) );
			CHECK( (L(90,90) >= 0 && L(90,90) <= 1) );
			CHECK( (L(100,100) >= 0 && L(100,100) <= 1) );
		}



		{

			matrix_op::matrix<float> L;

			measure_exec_time([&]() ->void 
			{
					L = matrix_op::rand_fill<float>(100 , 100 , 0 , 1);
					std::cout << __LINE__ << " : " ;
			});			


			CHECK( (L(20,20) >= 0 && L(20,20) <= 1) );
			CHECK( (L(90,20) >= 0 && L(90,20) <= 1) );
			CHECK( (L(80,30) >= 0 && L(80,30) <= 1) );
			CHECK( (L(90,90) >= 0 && L(90,90) <= 1) );
			CHECK( (L(100,100) >= 0 && L(100,100) <= 1) );
		}


		{

			matrix_op::matrix<float> L;

			measure_exec_time([&]() ->void 
			{
					L = matrix_op::rand_fill<float>(10000 , 10000 , 0 , 1);
					std::cout << __LINE__ << " : " ;
			});			


			CHECK( (L(20,20) >= 0 && L(20,20) <= 1) );
			CHECK( (L(90,20) >= 0 && L(90,20) <= 1) );
			CHECK( (L(80,30) >= 0 && L(80,30) <= 1) );
			CHECK( (L(90,90) >= 0 && L(90,90) <= 1) );
			CHECK( (L(100,100) >= 0 && L(100,100) <= 1) );
		}


		{

			matrix_op::matrix<int> L;

			measure_exec_time([&]() ->void 
			{
					L = matrix_op::rand_fill<int>(100 , 100 , 0 , 1);
					std::cout << __LINE__ << " : " ;
			});			


			CHECK( (L(20,20) >= 0 && L(20,20) <= 1) );
			CHECK( (L(90,20) >= 0 && L(90,20) <= 1) );
			CHECK( (L(80,30) >= 0 && L(80,30) <= 1) );
			CHECK( (L(90,90) >= 0 && L(90,90) <= 1) );
			CHECK( (L(100,100) >= 0 && L(100,100) <= 1) );
		}




		{

			#include <stdint.h>
			matrix_op::matrix<std::uint8_t> L;

			measure_exec_time([&]() ->void 
			{
					L = matrix_op::rand_fill<std::uint8_t>(100 , 100 , 0 , 1);
					std::cout << __LINE__ << " : " ;
			});			


			CHECK( (L(20,20) >= 0 && L(20,20) <= 1) );
			CHECK( (L(90,20) >= 0 && L(90,20) <= 1) );
			CHECK( (L(80,30) >= 0 && L(80,30) <= 1) );
			CHECK( (L(90,90) >= 0 && L(90,90) <= 1) );
			CHECK( (L(100,100) >= 0 && L(100,100) <= 1) );
		}



		{

			#include <stdint.h>
			matrix_op::matrix<std::uint8_t> L;

			measure_exec_time([&]() ->void 
			{
					L = matrix_op::rand_fill<std::uint8_t>(10000 , 100000 , 0 , 1);
					std::cout << __LINE__ << " : " ;
			});			


			CHECK( (L(20,20) >= 0 && L(20,20) <= 1) );
			CHECK( (L(90,20) >= 0 && L(90,20) <= 1) );
			CHECK( (L(80,30) >= 0 && L(80,30) <= 1) );
			CHECK( (L(90,90) >= 0 && L(90,90) <= 1) );
			CHECK( (L(100,100) >= 0 && L(100,100) <= 1) );
		}



	}	

	
	auto test_inner_prod = "test inner product ";
	SECTION(test_inner_prod)
	{
		{

			matrix_op::matrix<float> L;

			measure_exec_time([&]() ->void 
			{
					L = matrix_op::rand_fill<float>(10000000 , 1 , 0 , 100);
					std::cout << __LINE__ << " : " ;
			});			

			int result = 0;
			measure_exec_time([&]() ->void 
			{
					result = matrix_op::inner_product<float>(L , L);
					std::cout << __LINE__ << " : " ;
			});			
			




		}


	}


}


/*




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
*/
