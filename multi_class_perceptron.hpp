#ifndef MULTI_CLASS_PERCEPTRON_HPP
#define MULTI_CLASS_PERCEPTRON_HPP

#include "dmlpack.hpp"

namespace dmlpack
{

	enum class perceptron_type {simple, mira};

	template <typename T>
	class multi_class_perceptron : public ml<T>
	{

		
		public :

			multi_class_perceptron(perceptron_type type)
			{
				_type = type;
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
						matrix_op::matrix<T> weight_vec = _perceptron_weight.returnRow(class_idx);			
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
			{
				using namespace matrix_op;

				matrix<T> res; // the matrix will be of size # test samples * num classes

				size_t num_test_samples = ml<T>::_test_x.numRows();

				// Now compute the class prediction for each test sample
				matrix<T> prediction(num_test_samples , 1);  

				// Go through each element in the features of x and from compuute the probabilites
				for(size_t test_sample = 1; test_sample <= num_test_samples; ++test_sample) // each row in the matrices
				{
					// get the feature vector
					matrix<T> feature_vec = ml<T>::_test_x.returnRow(test_sample);	

					// Append the +1 towards its end. 
					feature_vec.resize(1 , feature_vec.numCols() + 1);
					feature_vec(1 , feature_vec.numCols()) = 1;

					//dout << " inference " << feature_vec;

					matrix<T> sub_res(1 , ml<T>::num_classes);

					// compute the similiarty between the current feature and each of the classes
					for(int class_idx = 1 ; class_idx <= ml<T>::num_classes ; ++class_idx)
					{
						sub_res(1,class_idx)  = _perceptron_weight.returnRow(class_idx).innerProduct(feature_vec);
					}
					
					//dout << " result matrix " << sub_res;

					res.addRow(sub_res); // add the probabilities over the different classes 
				
					//dout << " total result " << res ;
					// Use softMax and select max to do prediction on what is the best class to be taken
					// std::max_element + distance to find the index of with the largest probaility . 
					// Add one since the output of distance is 0 index, while the classes are 1 indexed 
				
					prediction(test_sample , 1) = sub_res.arg_max();
					//dout << " prediction udpates " << prediction;

				}

			       	// keep track of the prediciton that was made to test the accuracy
				ml<T>::_test_prob_pred = prediction;
				ml<T>::_test_class_pred = res;

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
