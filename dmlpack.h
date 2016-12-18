#ifndef DMLPACK_H
#define DMLPACK_H

/*
 *
 *
 * Might end up being a header only class or namespace *
 * Provide functionality for : 
 * 	Naive Bayes
 * 	Perceptron
 * 	Simply Multi Layer Neural Network and Back Prop
 *
 */


#include <stdexcept>
#include <unordered_map>
#include <cmath>
#include <algorithm>
#include <memory>
#include <queue>


#include "matrix.h"
#include "debug.h"


/* * Start : Dec 10 th: 
 * Memory Strategy : As long as I am not doing any real threading or cuda issues. I am going to make the C++ lib handle all the memory. 
 * 	> That is I will not explicitly allocate memory, but use the start contrainers to do the dirty work for me.
 *
 */


enum class classifier_type { naive_bayes , perceptron , single_later_nn , multi_layer_nn };


enum class perceptron_type {simple, mira};


/*
 * HOW TO USE : 
 * 	init the class with a particular type of classiifer 
 * 	then call the functions related to that classifier on the object.
 * 	if non related functions are called, this will lead to an exception.	
 *	// TO DO : Once the basic implementation is done, imagine, better api for ther user.
 *		 : Implement an abstract base class 
 *
 *
 *	The template has to be a basic type for things to work out properly.
 *
 */



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
class dmlpack
{
	public:	
		dmlpack(classifier_type ml_type) : ml_type_(ml_type)
		{
			
		};	
		
		// diable the copy cstor and copy assignment cstor
		dmlpack(const dmlpack& ) = delete ;
		dmlpack& operator=(const dmlpack& ) = delete;


		// feed the entire training data
		// keep reference to the class, instead of copying the data set
		// This means that the data should outlive the class
		void feed_train_data(const matrix<T>& train_x , const matrix<T>& train_y) const
		{
			train_x_ = train_x; 
			train_y_ = train_y;	

			// Number of classes to classify the data into 
			num_classes = train_y_.numCols();

			num_features = train_x_.numCols();

			num_samples += train_x_.numRows();

		}
	

		// feed the entire test data 
		void feed_test_data(matrix<T>& test_x )
		{
			test_x_ = test_x;	
		}

		// train on the whole data set provided
		void train();	


		// set the test set on the model and give back the accuracy 		
		double test();	

		// run inference , take in data point and see what the model predicts
		matrix<T> inference(matrix<T>& test_x);
		
		// pass in the index of the test set where you want the inference to be done.
		matrix<T> inference(size_t test_set_idx);	


		// train on a batch of data 
		// Two days this can be done. 
		// 	Pass in a matrix with the data to be trained on 
		// 		OR 
		//  	Pass in some sort of index or identifier, which will help us decide which part of the matrix we want to train on	
		void train_on_batch();
	

	private:
		classifier_type ml_type_; // machine learning type

		// training data
		std::shared_ptr<matrix<T>> train_x_;			
		std::shared_ptr<matrix<T>> train_y_;			

		// testing data
		std::shared_ptr<matrix<T>> test_x_;			
//		matrix<T>& test_y_ ; While testing only the x labels are given and we have to predict the y values from that 


		// multi layer neural network internals

		// will hold the different types of neural layers	
		std::vector<matrix<T>> network_layers_;

		void multi_layer_nn_train(double learning_rate, size_t iterations);

		T sigmoid(T val);

		// single layer neural network 

		matrix<T> single_layer_nn_weigh_;			 // single layer neural network weights

		void single_layer_nn_train(double learning_rate, size_t iterations);



		//percetron internals
		std::pair<bool,T> single_preceptron(const matrix<T>& feature , const matrix<T>& weight , T threshold  = 0 ) const;

		std::pair<matrix<T> , matrix<T>> multi_class_perceptron_inference();

		void multi_class_perceptron_train(perceptron_type type);
	 	matrix<T> perceptron_weight_;




		// multi-layer perceptron : Internals.


		// naive bayes //internals	------------------------------------
		void naive_bayes_train();
		std::pair<matrix<T> , matrix<T> > naive_bayes_inference();
		std::size_t num_samples;  // in a multi batch train scenario ne need to keep track of the number of samples we have seen
					      //------------------------------------
		std::size_t num_classes;
		std::size_t num_features ;


		// Normalize with laplace smoothing
		T normalize_laplace(size_t class_occurance  , size_t total_classes , size_t total_occurances ,  size_t strength = 1)  // pretend there is a uniform distribution of the data. Then update it based on evidence
		{ 
			return (double ) ( (class_occurance + strength) / (double) (total_occurances + total_classes * strength) ) ;
		};	


		matrix<T> softmax(matrix<T>& prob) const ;// has to be applied on a column vector
		
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




template <typename T>
matrix<T> dmlpack<T>::softmax(matrix<T>& prob) const // has to be applied on a column vector
{
	matrix<T> res(prob.numRows , prob.numCols);	
	
	double norm_factor = 0 ; 	
	
	for(int idx = 0 ; idx < prob.size() ; ++idx)
	{
		if(prob.isColVector)
		{
			norm_factor += std::exp(prob(idx,1));
		}
		else
		{
			norm_factor += std::exp(prob(1,idx));
		}

	}

	// Now compute e^(x) / sigma e^(x_i)
	for(int idx = 0 ; idx < prob.size() ; ++idx)
	{
		if(prob.isColVector)
		{
			res(idx,1) =  std::exp(prob(idx,1)); 

			// normalize
			res(idx,1) /= norm_factor;
		}
		else
		{
			res(1,idx) =  std::exp(prob(idx,1)); 

			// normalize
			res(1,idx) /= norm_factor;
		}
	}

	return res;
}






/*
 *  Run the training data on the set training set
 *
 * This navie bayes will run at an abstracted level, just pass in the feature vector and the output and will train on them.
 *
 *	The training data, each row correponds to a new sample. 
 *	The training data y is encoded in a one shot algorithm. 	
 *		So if class at index 1 is the output, then the value at index 1 witll be 1 and 0 everywhere else.
 * 	
 */
template <typename T>
void dmlpack<T>::naive_bayes_train()
{

	// check number of traning samples are consistent
	if(train_x_.numRows() != train_y_.numRows())
	{
		std::invalid_argument(std::string("mismatch ; Make sure the training set has equal number of smaples in x and y ") + std::string( " in ") + std::string("naive_bayes_train() ") + std::string( __FILE__) + std::string(" : ") + std::to_string(__LINE__) );
	}

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
	for(size_t train_sample = 1; train_sample <= train_x_.numRows() ; ++train_sample) // each row in the matrices
	{

		size_t class_idx = 1;

		// first go over the y portion of the data set to find the class
		for(class_idx = 1  ; class_idx <= num_classes ; ++class_idx)
		{
			T val = train_y_(train_sample , class_idx );		 // the matrix is 1 indexed, this is odd for cs. But is standard in math > what is better ? 

			/*
			 * Why keep templates ? 
			 * Because in a deep learning network, I will want to different data types.
			 * 
			 * If I use a short instead of a double, then it could lead to descritization of the search space ? Faster convergence ? 
			 * 
			 */

			if(val == 1)
			{
				map_class_occurance[class_idx] += 1; // this class has been seen agian in the data set
				break;
		       		// we know know that class_idx is the y value for the current training sample
			}	


		}



		// Go over all the features and count of occurances of a feature and feature given class
		for(size_t feature_idx = 1 ; feature_idx < num_features ; ++feature_idx)
		{
			T val = train_x_(train_sample , feature_idx); 

			if(val == 1)
			{
				map_feature_occurance[feature_idx] += 1; // occurance of a feature in the data set
				
				feature_in_class key(feature_idx , class_idx);

				map_feature_in_class_occurance[key] += 1;  // occurance of a feature together with the class

			}	
		}


	}


}



/*
 * Returns both the final probabilites of each class for each training samples and the result from applying softmax
 */
template <typename T>
std::pair<matrix<T> , matrix<T> > dmlpack<T>::naive_bayes_inference()
{

	matrix<T> res; // the matrix will be of size # test samples * num classes

	size_t num_test_samples = test_x_.numRows();

	// Now compute the class prediction for each test sample
	matrix<T> prediction(num_test_samples , 1);  


	// Go through each element in the features of x and from compuute the probabilites
	for(size_t test_sample = 1; test_sample <= num_test_samples; ++test_sample) // each row in the matrices
	{

		matrix<T> sub_mat(1, num_classes); // this row vector will be concatenated to the end of the res

		// fill them with P(Y)
	
		for(size_t class_idx = 1 ; class_idx <= num_classes ; ++class_idx)
		{
			sub_mat(1,class_idx) =  normalize_laplace(map_class_occurance[class_idx] , num_classes , num_samples );  // number of occuracnes of the given clas
			// convert to probability by normalizing
		
			sub_mat(1 , class_idx) = std::log(sub_mat(1 , class_idx));

			/*
			 * Normalizing here, instead of when the insertion into the map happens, enables us to train in batches
			 * the num_samples is updated when a new batch is added
			 * and the occurances of each of the classes are also updated.
			 */
		}
	
		size_t num_features = test_x_.numCols(); 
		/*
		 * p(y , f_i ) = p(y) * sigma{ p(f_i|y) }
		 */
		for(size_t feature_idx = 1 ; feature_idx <=  num_features; ++feature_idx)
		{
			T val = test_x_(test_sample , feature_idx); 

			if(val == 1) // The feature is present and we will have to compute the p(f_i | y)
			{
			// if the feature is not present, then it does not give up any way to update the probability of which class to choose from
				// we have to compute for each class. That is given this feature, what is the probability of seeeing a partucular class	
				
				for(size_t class_idx = 1 ; class_idx <= num_classes ; ++num_classes)
				{
					// compute p(f_i / y) , by counting the occurance of a feature for the partucular class and dividing it by the total occurance of that feature
					T temp = normalize_laplace( map_feature_in_class_occurance[std::make_pair(feature_idx , class_idx)] ,
						       	        	num_features , 
									map_feature_occurance[feature_idx] ); 

					sub_mat(1, class_idx) += std::log(temp);
				}

			}


		}
		
		res.addRow(sub_mat); // add the probabilities over the different classes 
		
		// Use softMax and select max to do prediction on what is the best class to be taken

		// normalize using softmax. squishes everything to lie in the 0,1 range and the total sum = 1. 
		sub_mat = softmax(sub_mat);

		// std::max_element + distance to find the index of with the largest probaility . 
		// Add one since the output of distance is 0 index, while the classes are 1 indexed 
		prediction(test_sample , 1) = sub_mat.arg_max();
			

	}

	return make_pair( res , prediction );
}


/* 
 * RUn the tained model on the given test data and gives back the accuracyt . 
 * 	How to measure accuracy ? For now just count the number of times the prediction and the actual y matches. 
 */


/*
 * A single percetron model 
 * Plug this model into others to obtain better results.
 *
 * if feauture * weight >= threshold then 1 else 0
 *
 * intput : the feature (row vector), the weight vector (column vector) , double threshould
 * output : 0 or 1 representing whether the neuroing has fired or not 
 */
template <typename T>
std::pair<bool, T> dmlpack<T>::single_preceptron(const matrix<T>& feature , const matrix<T>& weight , T threshold ) const
{
	T res = feature.innerProduct(weight);
	if(res > threshold)
	{
		return std::make_pair(true , res);
	}
	else
	{
		return std::make_pair(false, res);
	}
}



/*
 * simple perceptron update rule
 */
template <typename T>
std::pair<matrix<T> , matrix<T>> perceptron_update(const matrix<T>& predicted_id_weight ,const matrix<T>& actual_id_weight ,const matrix<T>& feature_vec)
{
	matrix<T> predicted_id_weight_result = predicted_id_weight - feature_vec;
	matrix<T> actual_id_weight_result  = actual_id_weight + feature_vec;

	return std::make_pair(predicted_id_weight_result , actual_id_weight_result);	
}	

/*
 * perceptron update with mira
 */
template <typename T>
std::pair<matrix<T> , matrix<T>> mira_perceptron_update(const matrix<T>& predicted_id_weight , const matrix<T>& actual_id_weight , const matrix<T>& feature_vec)
{

	T tau = (predicted_id_weight - actual_id_weight).innerProduct(feature_vec.transpose());

	tau += 1;
	tau /= 2 * feature_vec.normEuclidean();

	matrix<T> predicted_id_weight_result = predicted_id_weight - tau * feature_vec;
	matrix<T> actual_id_weight_result  = actual_id_weight + tau * feature_vec;

	return std::make_pair(predicted_id_weight_result , actual_id_weight_result);	
}


/*
 * Perceptron algorithm
 * For each output class there should be a particular weight vector that can recognize it.
 * The number of items in the weight vector will be equal to the number of features that we choose.
 * 
 * So how will training work ? 
 * 	 The weight vector for each class will be modified for each training sample.
 *	 so a training sample has class 1
 *	 we will update the class 0 weight vector so that it will not be activated on seeing this feature vector
 * 	 we will update the class 1 weight vector so that it will be activated on the next occurance of a similar feature.
 *
 *
 * 	 For each feature a bias has to be added to ensure that the descision can move away from the origin 
 *	 Will the bias addition be done with when the data source does the conversion or will it added / simulated with in this function ? 
 *		Simulated within this function. 
 *		The bias is anyways going to be +1 in the feature set
 *		The weight vector will determine what the magnitude of the bias will be 
 *
 * Before calling the function set the training set 
 * The training set's y should be in one-shot encoding
 *
 */
template <typename T>
void dmlpack<T>::multi_class_perceptron_train(perceptron_type type)
{

	const size_t num_train_samples = train_x_.numRows();

	//Resize the weight vector to hold # classes rows and #features columns
	// A weight vector for each of the classes 
	// the weight vector will be used to determine how much the neuron will look at each feature
	// Initially all the weights are 0
	perceptron_weight_.resize(num_classes , num_features + 1 , 0 ); // + 1 for the biases 

	perceptron_weight_(1 , num_features + 1) = 1 ; // set the bias for the class to be 1, so the tie can be broken for arg_max, when the algorithm starts are the weight vector is filled with 0


	// Now go through the data set and fill in these values
	for(size_t train_sample = 1; train_sample <= num_train_samples ; ++train_sample) // each row in the matrices
	{

		// get the feature vector
		matrix<T> feature_vec = train_x_.returnRow(train_sample);	

		// Append the +1 towards its end. 
		feature_vec.resize(1 , feature_vec.numCols() + 1);
		feature_vec(1 , feature_vec.numCols()) = 1;


		// Okay two seperate ways to implement this. 
		// Update the weight of all the classes. 
		// Update the weight of the class with the max weight
		// and also the class that was predicted .
	
		matrix<T> class_pred(1,num_classes);

		size_t actual_class_id = 0 ; 

		// first go over the y portion of the data set to find the class
		for(size_t class_idx = 1  ; class_idx <= num_classes ; ++class_idx)
		{

			// get the weight vector for a particular class
			matrix<T> weight_vec = perceptron_weight_.returnRow(class_idx);			
			std::pair<bool, T> pred = single_preceptron(feature_vec , weight_vec);
		
			class_pred(1,class_idx) = pred.second;

			bool actual = train_y_(train_sample , class_idx); 	

			if(actual)
			{
				actual_class_id = class_idx;
			}
			
		}
		
		auto predicted_class_idx = class_pred.arg_max();

		// update the weight vectors
		// if the predicted class and the acutal class are not the same,
		// then we reduce the weight vector for the predicted class 
		// and increase the weight vector for the actual class .
		if(predicted_class_idx != actual_class_id)
		{
			// reduce the weight vector for the predicted class 
			matrix<T> reduced_weight = perceptron_weight_.returnRow(predicted_class_idx);		

			// increase the weight vector for the actual class .
			matrix<T> increase_weight = perceptron_weight_.returnRow(actual_class_id);		


			// different update rules based on choice 	
			if(type == perceptron_type::simple)
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
		}	
	}	
}




/*
 * Create a single layer neural network. 
 * assume all the neruons are of sigmoid type
 * Train the neural net using the back prop algorithm
 * 
 * Hold the weight vector within a single matrix for now. 
 * What determines the size of the neuron matrix ? 
 */
template <typename T>
void dmlpack<T>::single_layer_nn_train(double learning_rate, size_t iterations)
{
	const size_t num_train_samples = train_x_.numRows();

	//Resize the weight vector to hold # classes rows and #features columns
	// A weight vector for each of the classes 
	// the weight vector will be used to determine how much the neuron will look at each feature
	// Initially all the weights are 0
	single_layer_nn_weigh_.resize(num_classes , num_features + 1 , 0 ); // + 1 for the biases 
	// m * n

	//matrix<T> delta_weight(num_classes , num_features + 1 , 0);		
	for(size_t iter = 0 ; iter < iterations ; ++iter)
	{
		for(size_t train_sample = 1; train_sample <= num_train_samples ; ++train_sample) // each row in the matrices
		{
			// get the feature vector 1 * n
			matrix<T> feature_vec = train_x_.returnRow(train_sample);	

			// Append the +1 towards its end. 
			feature_vec.resize(1 , feature_vec.numCols() + 1);
			feature_vec(1 , feature_vec.numCols()) = 1;
		
			// m * 1
			matrix<T> actual_output_vec  = train_y_.returnRow(train_sample).transpose();
			
		 	// m * 1				// m * n 		// n * 1
			matrix<T> pred_output_vec = single_layer_nn_weigh_ * feature_vec.transpose();

		//	delta_weight = delta_weight + learning_rate * ( ( actual_output_vec - pred_output_vec ) * feature_vec); 
		
			//incremental change
			
			single_layer_nn_weigh_ = single_layer_nn_weigh_ + learning_rate * ( ( actual_output_vec - pred_output_vec ) * feature_vec); 

		}
	}

}


template <typename T>
T dmlpack<T>::sigmoid(T val)
{
	T ret= 1 ;
	ret /= (1 + std::exp(-val));
	return ret;
}


/*
 * multy layer neural network. 
 * Each layer is inserted into a std::vector
 * Each layer will be a matrix
 *
 */

template <typename T>
void dmlpack<T>::multi_layer_nn_train(double learning_rate, size_t iterations)
{

	const size_t num_train_samples = train_x_.numRows();



	// for now assume it is a 2 layer network
	// 1 hidden layer
	// 1 fully connected layer

	// How will layers be inserted into the vector ? 
	// the output layer will be at the end of the vector
	// while input layers will be at the front (index 0)

	// create the pipeline

	const size_t num_neuron_hidden_layer = 20;
	const size_t num_neuron_hidden_layer_2 = 15;

	matrix<T> hidden_layer(num_neuron_hidden_layer,num_features + 1); // m1 * n

	matrix<T> hidden_layer_2(num_neuron_hidden_layer_2,num_neuron_hidden_layer); // m2 * m1
	matrix<T> output_layer(num_classes, num_neuron_hidden_layer_2);

	// m1 * n    *    n * 1
	// m1 * 1
	// m2 * m1   *    m1 * 1
	// m2 * 1
	// o * m2    *    m2 * 1
	// o * 1


	//simple for now. Copy into the pipeline. Later, look into creating the layers in place, or about moving it
	network_layers_.push_back(hidden_layer);
	network_layers_.push_back(hidden_layer_2);
	network_layers_.push_back(output_layer);



	// copy the values into a queue, where the output from each layer is stored during the forward pass
	std::queue<matrix<T>> qu;

	/*
	 * do the gradients have to be stored in a queue. No, the grad from the previous layer (i) is used in the current layer ( i - 1), 
	 * and in the next layer ( i - 2 ), the grad from the current layer (i -1 ) is only required, we will not need the grad from the previous layer ( i )
	 * But the gradients will of different sizes in different layers. 
	 */
	matrix<T> gradiant_store;


		//matrix<T> delta_weight(num_classes , num_features + 1 , 0);		
	for(size_t iter = 0 ; iter < iterations ; ++iter)
	{
		for(size_t train_sample = 1; train_sample <= num_train_samples ; ++train_sample) // each row in the matrices
		{
			// get the feature vector 1 * n
			matrix<T> feature_vec = train_x_.returnRow(train_sample);	

			// Append the +1 towards its end. 
			feature_vec.resize(1 , feature_vec.numCols() + 1);
			feature_vec(1 , feature_vec.numCols()) = 1;
	
			// forward propogate the signal thorugh the network and store in a queue. while updating the weigh. read from it .
			for(auto itr = network_layers_.begin() ; itr != network_layers_.end() ; ++itr)
			{
				matrix<T> output ;	
				if(itr == network_layers_.begin()) // if the first layer, feed in the actual input
				{
					output = feature_vec.transpose();	
				}		
				else
				{
					output = qu.top();
				}

				std::for_each(output.begin() , output.end() , [this](T& val)
						{
							val = sigmoid(val);
						});


				qu.push( (*itr) * output);
				
			}

	
			// o * 1
			matrix<T> actual_output_vec  = train_y_.returnRow(train_sample).transpose();


			// propogate down from the last layer backward . And update the weights accroding to the differing rules.
			// output layer has a different rule
			for(auto itr = --network_layers_.end() ; itr >= network_layers_.begin() ; --itr)
			{
		
				auto p_itr = itr + 1;


				// update the weights for the current layer.
				// 	TO DO : VEctorize the code 

				auto current_layer_activation = qu.top() ; qu.pop();
				auto prev_layer_activation = qu.top() ;
				
				// stores the gradient for the current layer which is fed back into lower layers	
				matrix<T> current_layer_grad(current_layer_activation.numRows() , current_layer_activation.numCols());

				bool output_layer_flag = (itr == --network_layers_.end());
					

				itr->transform_inplace([&](std::size_t row , std::size_t col)
						{
							T w_delta = 0 ;

							if(output_layer_flag)
							{
								// error will be calculated between the training output and the results from forward pass in the final layer
							
								// go over each of the neurons in the current layer
								w_delta = current_layer_activation(col , 1);
								w_delta *= (actual_output_vec(1,col) - w_delta);
								w_delta *= (1 - w_delta); 


							}
							else
							{

								w_delta = current_layer_activation(col , 1);
//								w_delta *= current_layer_grad * (*p_itr)(

								

								// in w(current_num_neur , prev_num_neur) each row represents the connnection of the a single nueron in the current layer 
								// with all the neurons in the previous layer-
								// 
								// So, inorder to determine the connets of a neuron in the lower layer with the neurons in the upper layer, we 
								// actually have to look at a particular column of each row of the weight matrix. 
								// 	The column index is decided by the nueron in the lower layer .




							}
	

							current_layer_grad(col ,1 ) = w_delta;	 // store the current gradients for the next layers

							return learning_rate * w_delta * prev_layer_activation(col , 1);

						});

				

				gradiant_store = std::move(current_layer_grad);


				


			}



		 	// m * 1				// m * n 		// n * 1
			matrix<T> pred_output_vec = single_layer_nn_weigh_ * feature_vec.transpose();

		//	delta_weight = delta_weight + learning_rate * ( ( actual_output_vec - pred_output_vec ) * feature_vec); 
		
			//incremental change
			
			single_layer_nn_weigh_ = single_layer_nn_weigh_ + learning_rate * ( ( actual_output_vec - pred_output_vec ) * feature_vec); 

		}
	}



}





#endif
