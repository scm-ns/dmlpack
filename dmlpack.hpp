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

#include <stdexcept>
#include <unordered_map>
#include <cmath>
#include <algorithm>
#include <memory>
#include <queue>



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





	// Contains the operators used to create the hash functions
	struct hash_fctor // functor
	{
		template <class T1 , class T2>
		std::size_t operator() (const std::pair<T1 , T2> & p) const 
		{
			auto h1 = std::hash<T1>{}(p.first);
			auto h2 = std::hash<T2>{}(p.second);	
			// check out 
			// http://eternallyconfuzzled.com/tuts/algorithms/jsw_tut_hashing.aspx for good hash algorithms
			
			unsigned int hash = 0 ; 
		
			h1 = h1 * 2654435761 % (2^17);
			h2 = h2 * 2654435761 % (2^17);

			h1 >> 13;
			h2 >> 3;

			// SHIFT ADD XOR HASH
			hash ^= (hash << 5) + (hash >> 2) + h1 + h2;
			
			return hash;
		}


		template <class T>
		std::size_t operator() (const T& p) const 
		{
			auto h1 = std::hash<T>{}(p);

			
			unsigned int hash = 0 ; 
			
			// SHIFT ADD XOR HASH
			hash ^= (hash << 5) + (hash >> 2) + h1 ;
			
			return hash;
		}

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
	
	//TODO : Break out namespace into seperate files

	template <typename T>
	class  naive_bayes : public ml<T>
	{
		public:
			naive_bayes()
			{

			};

			void train()
			{


				/*
				 * 
				 * Basic Idea : 
				 * 	Go through the training set and count the occurance of each type of training sample and feature.
				 *	
				 *	The training can be continued with another batch. 
				 *	So do not compute the probabilities directly, instead just keep the numbers and when probabilites are needed, normalize them to 
				 *	get the required results.
				 */
				

				// Now go through the data set and fill in these values
				for(size_t train_sample = 1; train_sample <= ml<T>::_train_x.numRows() ; ++train_sample) // each row in the matrices
				{

					size_t class_idx = 1;
					
					dout << train_sample << std::endl;
					// first go over the y portion of the data set to find the class
					for(class_idx = 1  ; class_idx <= ml<T>::_num_classes ; ++class_idx)
					{
						T val = ml<T>::_train_y(train_sample , class_idx );		 // the matrix is 1 indexed, this is odd for cs. But is standard in math > what is better ? 
						dout << val << std::endl;
						
						/*
						 * Why keep templates ? 
						 * Because in a deep learning network, I will want to different data types.
						 * If I use a short instead of a double, then it could lead to descritization of the search space ? Faster convergence ? 
						 */

						if(val == 1)
						{
							map_class_occurance[class_idx] += 1; // this class has been seen agian in the data set
							break;
							// we know know that class_idx is the y value for the current training sample
						}	


					}


					// Go over all the features and count of occurances of a feature and feature given class
					for(size_t feature_idx = 1 ; feature_idx <= ml<T>::_num_features ; ++feature_idx)
					{
						T val = ml<T>::_train_x(train_sample , feature_idx); 
							
						if(val == 1)
						{
							map_feature_occurance[feature_idx] += 1; // occurance of a feature in the data set
							
							feature_in_class key(feature_idx , class_idx);

							map_feature_in_class_occurance[key] += 1;  // occurance of a feature together with the class

						}	
					}

				}


			}


			void test()
			{


			}

			matrix_op::matrix<T> infer_single_feature(matrix_op::matrix<T> feature_vec)
			{


			}

			matrix_op::matrix<T> infer_batch(matrix_op::matrix<T> feature_vec)
			{


			}


		private:

		
			using occurance = std::size_t;
			using class_index = std::size_t;
			using feature_index = std::size_t;

			// These objects should persist between multiple calls of the function
			// TO DO : Move them into class memebers 
			
			// to compute p(y)
			std::unordered_map<class_index,occurance , hash_fctor> map_class_occurance;  // maintain mapping between class and count of its occurances in the training set

			// to compute p(f_i)
			std::unordered_map<feature_index, occurance , hash_fctor> map_feature_occurance ; // maintain mapping between each feature and the number of times it appears in the data set

			using feature_in_class = std::pair<size_t , size_t>;

			// to compute p(f_i | y)
			std::unordered_map< feature_in_class , occurance , hash_fctor> map_feature_in_class_occurance ;  // mapping between the occurance of each feature in a particular class



	};













}



