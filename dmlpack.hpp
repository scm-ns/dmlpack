/*
 * differenct objects for handling different algorithms
 * adding support to a single object to do everything was a bad architecture design causing a lot of pain
 * Each algorithm will have some function redrived from a base class. 
 */

#include "matrix.hpp"



namespace dmlpack
{
	// template base class for every algorithm
	template <typename T>	
	class ml
	{ 

		protected:
			// hold the training data
			matrix_op::matrix<T> _train_x;  // inputs
			matrix_op::matrix<T> _train_y;  // labels

			// hold the testing data
			matrix_op::matrix<T> _test_x;
			matrix_op::matrix<T> _test_y;

			// hold the validation data 
			matrix_op::matrix<T> _valid_x;
			matrix_op::matrix<T> _valid_y;	
	
			std::size_t num_classes;
			std::size_t num_features;
			std::size_t num_samples;

		public:
			ml()
			{
				
			}


			void set_training_data( matrix_op::matrix<T> train_x , matrix_op::matrix<T> train_y)
			{
				_train_x = train_x;
				_train_y = train_y;
				
				// set these values to be used in the training algorithms
				num_classes = _train_y.numCols();	
				num_features = _train_x.numCols();
				num_samples = _train_x.numRows();
			}

			void set_testing_data( matrix_op::matrix<T> test_x , matrix_op::matrix<T> test_y)
			{
				_test_x = test_x;
				_test_y = test_y;
			}

			virtual void train() = 0;
			virtual void test() = 0;
	};

	template <typename T>
	class single_layer_nn : public ml<T>
	{
		public:
			single_layer_nn() 
			{
				
			};
				
		private: 
			std::size_t iterations;
			double learning_rate;				
	};




}



