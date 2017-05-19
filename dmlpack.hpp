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

#ifndef DMLPACK_HPP
#define DMLPACK_HPP

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

	// TODO : add this sepereat to the namespace when creating file for naive bayes
	enum class perceptron_type {simple, mira};

	template <typename T>
	class multi_class_perceptron : public ml<T>
	{

		
		public :


			multi_class_perceptron(perceptron_type type) : _type(type)
			{

			};




			void train()			
			{

				const size_t num_train_samples = ml<T>::_train_x.numRows();
				//dout << ml<T>::_train_x.numRows() << " " ;


				// Now go through the data set and fill in these values
				for(size_t train_sample = 1; train_sample <= num_train_samples ; ++train_sample) // each row in the matrices
				{
					//dout << train_sample << std::endl;
					// get the feature vector
					matrix_op::matrix<T> feature_vec = ml<T>::_train_x.returnRow(train_sample);	

					// Append the +1 towards its end. 
					feature_vec.resize(1 , feature_vec.numCols() + 1);
					feature_vec(1 , feature_vec.numCols()) = 1;

					//dout << "FEATURE VEC : " << feature_vec << std::endl;

					// Okay two seperate ways to implement this. 
					// Update the weight of all the classes. 
					// Update the weight of the class with the max weight
					// and also the class that was predicted .
				
					matrix_op::matrix<T> class_pred(1,ml<T>::_num_classes);

					//dout << " CLASS PRED : " << class_pred ; 

					size_t actual_class_id = 0 ; 

					// first go over the y portion of the data set to find the actual class
					// This for loop finds out what the predictied class is and also what the acutal class is
					for(size_t class_idx = 1  ; class_idx <= ml<T>::_num_classes ; ++class_idx)
					{
						//dout << class_idx << " " << ml<T>::_num_classes << std::endl;
						// get the weight vector for a particular class
						matrix_op::matrix<T> weight_vec = perceptron_weight_.returnRow(class_idx);			
						//dout << " weight_actual_feature " << weight_vec ;

						std::pair<bool, T> pred = single_preceptron(feature_vec , weight_vec);
					
						//dout << " single _percrption " << pred.second <<std::endl;	

						class_pred(1,class_idx) = pred.second;

						//dout << weight_vec << std::endl;

						bool actual = matrix_op::matrix<T>::_train_y(train_sample , class_idx); 	

						if(actual)
						{
							actual_class_id = class_idx - 1;
						}
						
					}

					//dout << "actual pred " << actual_class_id << std::endl;

					//dout << "class pred " << class_pred << std::endl;	

					auto predicted_class_idx = class_pred.arg_max();

					//dout << "class pred arg " << predicted_class_idx << std::endl;	

					// update the weight vectors
					// if the predicted class and the acutal class are not the same,
					// then we reduce the weight vector for the predicted class 
					// and increase the weight vector for the actual class .
					if(predicted_class_idx != actual_class_id)
					{

						//dout << "error in prediction updating the weight vectors " << std::endl;
						//dout << "actual perceptron weight " << perceptron_weight_ ;

						// reduce the weight vector for the predicted class 
						matrix_op::matrix<T> reduced_weight =_perceptron_weight.returnRow(predicted_class_idx + 1);		
					
						//dout << "reduced weight before " << reduced_weight ;
						
						// increase the weight vector for the actual class .
						matrix_op::matrix<T> increase_weight = _perceptron_weight.returnRow(actual_class_id + 1);		

						//dout << "increased weight before " << increase_weight ;

						// different update rules based on choice 	
						if(_type == perceptron_type::simple)
						{
							auto res = perceptron_update(reduced_weight , increase_weight , feature_vec);
							reduced_weight = res.first;
							increase_weight = res.second;	
						}
						else
						{
							auto res = mira_perceptron_update(reduced_weight , increase_weight , feature_vec);
							reduced_weight = res.first;
							increase_weight = res.second;	
						}

						//dout << "reduced weight after " << reduced_weight ;

						//dout << "increased weight afer " << increase_weight ;

						_perceptron_weight.replaceRow(reduced_weight , predicted_class_idx + 1);
						_perceptron_weight.replaceRow(increase_weight, actual_class_id + 1);
						
//						dout << "weigth matrix after " << perceptron_weight_ << std::endl;

					}	
				}	


			}
			void test()
j			{

			}
			matrix_op::matrix<T> infer_single_feature(matrix_op::matrix<T> feature_vec)
			{

			}
			matrix_op::matrix<T> infer_batch(matrix_op::matrix<T> feature_vec)
			{


			}
	

	
		private :
			perceptron_type _type;

			matrix_op::matrix<T> _perceptron_weight;

			std::pair<bool,T> single_preceptron(const matrix_op::matrix<T>& feature , const matrix_op::matrix<T>& weight , T threshold  = 0 ) const;

			std::pair<matrix_op::matrix<T> , matrix_op::matrix<T>> multi_class_perceptron_inference();

			void multi_class_perceptron_train(perceptron_type type = perceptron_type::simple , float percentage = 100);
			void multi_class_perceptron_train_iter(perceptron_type type , float percentage , int num_iter = 100);




	};
	



	





}

#endif 


