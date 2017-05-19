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


}

#endif 


