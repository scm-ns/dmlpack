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

			// output of the running the algorithm on the test set is stored
			matrix_op::matrix<T> _test_prob_pred;
			matrix_op::matrix<T> _test_class_pred;

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

				matrix_op::matrix<T> res; // the matrix will be of size # test samples * num classes

				size_t num_test_samples = ml<T>::_test_x.numRows();

		// Now compute the class prediction for each test sample
				matrix_op::matrix<T> prediction(num_test_samples , 1);  
				ml<T>::_test_prob_pred.resize(num_test_samples , 1); // Store all the output probabilities

				// Go through each element in the features of x and from compuute the probabilites
				for(size_t test_sample = 1; test_sample <= num_test_samples; ++test_sample) // each row in the matrices
				{

					matrix_op::matrix<T> sub_mat(1, ml<T>::num_classes); // this row vector will be concatenated to the end of the res
					// fill them with P(Y)
					dout << test_sample << std::endl;

					for(size_t class_idx = 1 ; class_idx <= ml<T>::num_classes ; ++class_idx)
					{
						//dout << num_classes << " " << num_samples << " " << map_class_occurance[class_idx];

						sub_mat(1,class_idx) =  normalize_laplace(map_class_occurance[class_idx] , ml<T>::num_classes , ml<T>::_num_train_samples);  // number of occuracnes of the given clas

						//dout << sub_mat  << std::endl;

						//sub_mat(1 , class_idx) = std::log(sub_mat(1 , class_idx));
					}

					dout << sub_mat  << std::endl;
					size_t num_features = ml<T>::_test_x.numCols(); 
					/*
					 * p(y , f_i ) = p(y) * sigma{ p(f_i|y) }
					 */
					for(size_t feature_idx = 1 ; feature_idx <=  num_features; ++feature_idx)
					{
						T val = ml<T>::_test_x(test_sample , feature_idx); 

						if(val == 1) // The feature is present and we will have to compute the p(f_i | y)
						{
						// if the feature is not present, then it does not give up any way to update the probability of which class to choose from
						// we have to compute for each class. That is given this feature, what is the probability of seeeing a partucular class	
							
							for(size_t class_idx = 1 ; class_idx <= ml<T>::_num_classes ; ++class_idx)
							{
								// compute p(f_i / y) , by counting the occurance of a feature for the partucular class and dividing it by the total occurance of that feature
							
								dout << class_idx << " " << map_feature_in_class_occurance[std::make_pair(feature_idx , class_idx)]  << " " << num_features << " " << map_feature_occurance[feature_idx]  << std::endl;

								T temp = normalize_laplace( map_feature_in_class_occurance[std::make_pair(feature_idx , class_idx)] ,
												num_features , 
												map_feature_occurance[feature_idx] ); 

								sub_mat(1, class_idx) += temp ;// std::log(temp);
							}

						}

					}

					// Use softMax and select max to do prediction on what is the best class to be taken

					// normalize using softmax. squishes everything to lie in the 0,1 range and the total sum = 1. 
					// std::max_element + distance to find the index of with the largest probaility . 
					// Add one since the output of distance is 0 index, while the classes are 1 indexed 
				
					sub_mat = softmax(sub_mat);
					res.addRow(sub_mat); // add the probabilities over the different classes 

					prediction(test_sample , 1) = sub_mat.arg_max();

				}

				ml<T>::_test_prob_pred = prediction;
				ml<T>::_test_class_pred = res;
				
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



