#include "matrix.h"
#include "dmlpack.h"
#include "data_source.h"
#include <iostream>



void test(CLASSIFIER_TYPE type, BRKLY_DATA data_f , float train_percen = 1)
{	
	std::cout << " ####################################### TRAINING ##################################" << std::endl;
	data_source a{};
	a.read_store_berkely_data(data_f , DATA_TYPE::TRAIN);

	matrix<double> data = a.get_features(); 	
	matrix<double> label = a.get_labels();


	dmlpack<double> K{type};
	
	K.feed_train_data(std::move(data) , std::move(label));
		
	K.train(train_percen);

	std::cout << " ###################################### VALIDATION SET ACCURACY #######################" << std::endl;
	data_source digit_valid{};
	digit_valid.read_store_berkely_data(data_f , DATA_TYPE::VALID);

	matrix<double> data_valid = digit_valid.get_features(); 	
	matrix<double> label_valid = digit_valid.get_labels();

	K.feed_test_data(std::move(data_valid), std::move(label_valid));
	auto res = K.inference();

	K.test_accuracy();


	std::cout << " ##################################### TESTING SET ACCURACY ###########################" << std::endl;
	data_source digit_test{};
	digit_test.read_store_berkely_data(data_f , DATA_TYPE::TEST);

	matrix<double> data_test = digit_test.get_features(); 	
	matrix<double> label_test = digit_test.get_labels();

	K.feed_test_data(std::move(data_test), std::move(label_test));
	res = K.inference();

	K.test_accuracy();

}



int main()
{
//	std::cout << " RUN THE 3 CLASSIFIERS ON THE DIGIT DATA SET " << std::endl;	
	for(int i = 1 ; i <= 10 ; ++i)
	{
		std::cout << std::endl << std::endl << std::endl << " Run algo with percent data " << i * 0.1 << std::endl << std::endl ;
		test(CLASSIFIER_TYPE::PERCEPTRON_MIRA , BRKLY_DATA::FACE , i * 0.1);
	}
	//test(CLASSIFIER_TYPE::PERCEPTRON , BRKLY_DATA::DIGIT);
	//test(CLASSIFIER_TYPE::PERCEPTRON_MIRA , BRKLY_DATA::DIGIT);
/*'
	std::cout << " RUN THE 3 CLASSIFIERS ON THE FACES DATA SET " << std::endl;
	test(CLASSIFIER_TYPE::NAIVE_BAYES , BRKLY_DATA::FACE);
	test(CLASSIFIER_TYPE::PERCEPTRON , BRKLY_DATA::FACE);
	test(CLASSIFIER_TYPE::PERCEPTRON_MIRA , BRKLY_DATA::FACE);
	*/

};



