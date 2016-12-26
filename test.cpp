#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "dmlpack.h"
#include "data_source.h"


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


