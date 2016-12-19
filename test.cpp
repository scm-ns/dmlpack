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

void test_digit_nb()
{
	data_source a{};
	a.read_store_berkely_data(BRKLY_DATA::DIGIT , DATA_TYPE::TRAIN);

	matrix<double> data = a.get_features(); 	
	matrix<double> label = a.get_labels();

	dmlpack<double> K{CLASSIFIER_TYPE::NAIVE_BAYES};
	
	K.feed_train_data(std::move(data) , std::move(label));
		
	K.train(1);

	data_source digit_valid{};
	digit_valid.read_store_berkely_data(BRKLY_DATA::DIGIT , DATA_TYPE::VALID);

	matrix<double> data_valid = digit_valid.get_features(); 	
	matrix<double> label_valid = digit_valid.get_labels();


	std::cout << data_valid.returnRow(1) << std::endl;
	std::cout << label_valid.returnRow(1) << std::endl;

	K.feed_test_data(std::move(data_valid), std::move(label_valid));
	auto res = K.inference();

	K.test_accuracy();


	data_source digit_test{};
	digit_test.read_store_berkely_data(BRKLY_DATA::DIGIT , DATA_TYPE::TEST);

	matrix<double> data_test = digit_test.get_features(); 	
	matrix<double> label_test = digit_test.get_labels();

	std::cout << data_test.returnRow(1) << std::endl;
	std::cout << label_test.returnRow(1) << std::endl;

	K.feed_test_data(std::move(data_test), std::move(label_test));
	res = K.inference();

	K.test_accuracy();


}

void test_perceptron()
{
	data_source a{};
	a.read_store_berkely_data(BRKLY_DATA::DIGIT , DATA_TYPE::TRAIN);

	matrix<double> data = a.get_features(); 	
	matrix<double> label = a.get_labels();

	dmlpack<double> K{CLASSIFIER_TYPE::PERCEPTRON};
	
	K.feed_train_data(std::move(data) , std::move(label));
		
	K.train();

	data_source digit_test{};
	digit_test.read_store_berkely_data(BRKLY_DATA::DIGIT , DATA_TYPE::TEST);

	matrix<double> data_test = digit_test.get_features(); 	
	matrix<double> label_test = digit_test.get_labels();

	std::cout << data_test.returnRow(1) << std::endl;
	std::cout << label_test.returnRow(1) << std::endl;

	K.feed_test_data(std::move(data_test), std::move(label_test));
	auto res = K.inference();

	K.test_accuracy();




}

void test_mira()
{
	data_source a{};
	a.read_store_berkely_data(BRKLY_DATA::DIGIT , DATA_TYPE::TRAIN);

	matrix<double> data = a.get_features(); 	
	matrix<double> label = a.get_labels();


	dmlpack<double> K{CLASSIFIER_TYPE::PERCEPTRON_MIRA};

	K.feed_train_data(std::move(data) , std::move(label));
		
	K.train();

	data_source digit_test{};
	digit_test.read_store_berkely_data(BRKLY_DATA::DIGIT , DATA_TYPE::TEST);

	matrix<double> data_test = digit_test.get_features(); 	
	matrix<double> label_test = digit_test.get_labels();

	K.feed_test_data(std::move(data_test), std::move(label_test));
	auto res = K.inference();

	K.test_accuracy();

}

void test_nb_faces()
{
	data_source a{};
	a.read_store_berkely_data(BRKLY_DATA::FACE , DATA_TYPE::TRAIN);

	matrix<double> data = a.get_features(); 	
	matrix<double> label = a.get_labels();

	dmlpack<double> K{CLASSIFIER_TYPE::NAIVE_BAYES};
	
	K.feed_train_data(std::move(data) , std::move(label));
		
	K.train(0.7,0);

	data_source digit_test{};
	digit_test.read_store_berkely_data(BRKLY_DATA::FACE , DATA_TYPE::TEST);

	matrix<double> data_test = digit_test.get_features(); 	
	matrix<double> label_test = digit_test.get_labels();

	std::cout << data_test.returnRow(1) << std::endl;
	std::cout << label_test.returnRow(1) << std::endl;

	K.feed_test_data(std::move(data_test), std::move(label_test));
	auto res = K.inference();

	K.test_accuracy();

}


int main()
{
	test_digit_nb();
};



