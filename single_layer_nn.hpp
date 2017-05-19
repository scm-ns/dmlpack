#include "dmlpack.hpp"


/*
 *
 * Add each algorithm into the same open namespace but add it in different files.
 *
 */

namespace  dmlpack
{

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
							std::cout << "Current Error Vector : " << error << std::endl;
							std::cout << "Current Error : " << matrix_op::inner_product<T>(error , error) << std::endl;
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
