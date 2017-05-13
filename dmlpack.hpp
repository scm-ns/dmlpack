/*
 * differenct objects for handling different algorithms
 * adding support to a single object to do everything was a bad architecture design causing a lot of pain
 * Each algorithm will have some function redrived from a base class. 
 *
 * along column : features of a particular data point
 * along row : all the data point in the data set
 *	
 *	     *	*  *  *  *
 *	     *	*  *  *  *
 *	     *	*  *  *  *
 *	 num features : 5
 *	 num data points : 3
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
	
			std::size_t _num_classes;
			std::size_t _num_features;
			std::size_t _num_train_samples;

		public:
			ml()
			{
				
			}

			void set_training_data( matrix_op::matrix<T> train_x , matrix_op::matrix<T> train_y)
			{
				_train_x = train_x;
				_train_y = train_y;
				
				// set these values to be used in the training algorithms
				_num_classes = _train_y.numCols();	
				_num_features = _train_x.numCols();
				_num_train_samples = _train_x.numRows();
			}

			void set_testing_data( matrix_op::matrix<T> test_x , matrix_op::matrix<T> test_y)
			{
				_test_x = test_x;
				_test_y = test_y;
			}

			virtual void train() = 0;
			virtual void test() = 0;
			virtual matrix_op::matrix<T> infer_single_feature(matrix_op::matrix<T> feature_vec) = 0;
			virtual matrix_op::matrix<T> infer_batch(matrix_op::matrix<T> feature_vec) = 0;
	};

	template <typename T>
	class single_layer_nn : public ml<T>
	{
		public:
			// TODO : create a better way to pass in parameters
			single_layer_nn(std::size_t iterations , double learning_rate , std::size_t logging_iter = 0) 
			{
				_iterations = iterations;
				_learning_rate = learning_rate;	
				_logging_iter = logging_iter;

			};

			/*
			 * precondition :
			 * 	Only call after testing the training data
			 *
			 */
			void train()
			{

				_layer.resize(ml<T>::_num_classes , ml<T>::_num_features + 1 , 0);

				//matrix<T> delta_weight(num_classes , num_features + 1 , 0);		
				for(size_t iter = 0 ; iter < _iterations ; ++iter)
				{
					for(size_t train_sample = 1; train_sample <= ml<T>::_num_train_samples ; ++train_sample) // each row in the matrices
					{
						// get the feature vector 1 * n
						matrix_op::matrix<T> feature_vec = ml<T>::_train_x.returnRow(train_sample);	

						// Append the +1 towards its end. 
						feature_vec.resize(1 , feature_vec.numCols() + 1);
						feature_vec(1 , feature_vec.numCols()) = 1;
					
						// m * 1
						matrix_op::matrix<T> actual_output_vec  = ml<T>:: _train_y.returnRow(train_sample).transpose();
						
						// m * 1				// m * n 		// n * 1
						matrix_op::matrix<T> pred_output_vec = _layer * feature_vec.transpose();

						auto error = actual_output_vec - pred_output_vec;
						
						if( _logging_iter != 0 && iter % _logging_iter == 0)
						{
							std::cout << error << std::endl;
						}

						//incremental change
						_layer = _layer + ( ( error ) * feature_vec) * _learning_rate; 
					}
				}

			}


			void test()
			{


			}

			matrix_op::matrix<T> infer_single_feature(matrix_op::matrix<T> feature_vec)
			{
				// data size : (1 , num_features)
				
				// output : ( num_classes , 1)
				//
				std::cout << _layer << std::endl;
				std::cout << feature_vec << std::endl;


				// Append the +1 towards its end. 
				feature_vec.resize(1 , feature_vec.numCols() + 1);
				feature_vec(1 , feature_vec.numCols()) = 1;
					
				return _layer * feature_vec.transpose();
			}

			matrix_op::matrix<T> infer_batch(matrix_op::matrix<T> batch)
			{


			}


		private: 
			std::size_t _iterations;
			double _learning_rate;				
			matrix_op::matrix<T> _layer;
			std::size_t _logging_iter;
	};

}



