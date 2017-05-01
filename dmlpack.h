#ifndef DMLPACK_H
#define DMLPACK_H

/*
 * Might end up being a header only class or namespace *
 * Provide functionality for : 
 * 	Naive Bayes
 * 	Perceptron
 * 	Simply Multi Layer Neural Network and Back Prop
 */

//#define DEBUG_D

#ifdef DEBUG_D
#define dout std::cout << __FILE__<< " (" << __LINE__ << ") " << "DEBUG : "
#else
#define dout 0 && std::cout
#endif



#include <stdexcept>
#include <unordered_map>
#include <cmath>
#include <algorithm>
#include <memory>
#include <queue>


#include "matrix.hpp"
#include "debug.h"


/* * Start : Dec 10 th: 
 * Memory Strategy : As long as I am not doing any real threading or cuda issues. I am going to make the C++ lib handle all the memory. 
 * 	> That is I will not explicitly allocate memory, but use the start contrainers to do the dirty work for me.
 */

enum class CLASSIFIER_TYPE { NAIVE_BAYES , PERCEPTRON , PERCEPTRON_MIRA , SINGLE_LAYER_NN , MULTI_LAYER_NN };

enum class perceptron_type {simple, mira};

const double MIRA_CAP = 0.001;


/*
 * HOW TO USE : 
 * 	init the class with a particular type of classiifer 
 * 	then call the functions related to that classifier on the object.
 * 	if non related functions are called, this will lead to an exception.	
 *	// TO DO : Once the basic implementation is done, imagine, better api for ther user.
 *		 : Implement an abstract base class 
 *
 *
 *	Set the Parameters in a struct : Not in a single contrcutor as then the function becomes terrible
 *	Each classfier will have its own param_struct. Which has to be passed in with the train command.
 *
 *	The template has to be a basic type for things to work out properly.
 *
 */

/*
 * Jan 20th : Design Descisions : How to offer a better interface. 
 *      Factory Pattern :
 *  	Base Class Inheritance : 
 *	
 *	Avoid Pointers In the interface.
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

struct param_base
{


};

// NOTES : cpp function overriding
/*

class base
{
	void show() 
	{
		std::cout << "base" ;
	}

	void test()
	{
		std::cout << "base-test" 
	}
}

class derived : public base
{
	void show() // now show called on the derived class will be routed here
	{
		std::cout << "derived" ;
	}
	
	void test()
	{
		std::cout << " base-test";
	}

}


void func(base& b1 , base& b2);
// suppose there are two subclass derived1 and derived2


so the calls can be called as 
func(derived1() , derived2())


 */



template <typename T>
class classifier_base
{
	public:
		virtual void train() = 0 ;
		virtual void test() = 0;
		
		// Set the parameters for the algorithm to run
		// Each algorithm will have its own parameter struct
		// This is much better than algorithms having many many parameters in the function
		virtual void set_param( param_base& x) = 0;

		virtual void train(float percentage) = 0 ; // override this is subclass

		// train on the percentage of the data set provided
		virtual void train(float percentage = 1, int iter = 4) = 0 ;	 // do all class have iterations. I need to specialize this function on the type of the algorithm
			// I cannot ask that all the subclass implement this fuinction as they might not need the iter. How to handle this ? 


		// run inference , take in data point and see what the model predicts
		virtual matrix<T> inference(matrix<T>& test_x) = 0;

		// test set already set
		virtual std::pair<matrix<T>,matrix<T>> inference() = 0 ;


		// feed the entire training data
		// keep reference to the class, instead of copying the data set
		// This means that the data should outlive the class
		void feed_train_data( matrix<T> train_x ,  matrix<T> train_y);
		
		// feed the entire test data 
		void feed_test_data(matrix<T> test_x , matrix<T> test_y);
		void test_accuracy();

	protected: // need to share with base classes for their usage
		// TO DO : Make sure the initilazations are done properly

		// training data
		matrix<T> train_x_;			
		matrix<T> train_y_;			

		// testing data
		matrix<T> test_x_;			
		matrix<T> test_y_;			

		matrix<T> prediction_;

		std::size_t num_samples;  // in a multi batch train scenario ne need to keep track of the number of samples we have seen
		std::size_t num_classes;
		std::size_t num_features ;


};

template <typename T>
void classifier_base<T>::feed_train_data( matrix<T> train_x ,  matrix<T> train_y) 
{
	train_x_ = train_x; 
	train_y_ = train_y;	

	num_classes = train_y_.numCols();
	num_features = train_x_.numCols();
	num_samples = train_x_.numRows();

}

template <typename T>
void classifier_base<T>::feed_test_data(matrix<T> test_x , matrix<T> test_y)
{
	test_x_ = test_x;	
	test_y_ = test_y;
}



template <typename T>
void classifier_base<T>::test_accuracy()
{
	const int num_test_samples = test_y_.numRows();
	if(prediction_.empty())
	{
		throw std::runtime_error("The prediction matrix has not been built. Run inference first");
	}


	dout << prediction_ << std::endl;

	if(prediction_.numRows() != num_test_samples)
	{
		dout << "error" ;
		throw std::invalid_argument("number of sampels do not match"); }
		
	int correct  = 0 ;
	if(num_classes == 1)
	{
		for(size_t idx = 1 ; idx <= num_test_samples ; ++idx)
		{
			if(prediction_(idx, 1) == test_y_(idx,1))
			{
				correct++;
			}
		}

	}		
	else
	{
		for(size_t idx = 1 ; idx <= num_test_samples ; ++idx)
		{
			int actual_val = 0 ; 
			for(size_t class_idx = 1 ; class_idx <= num_classes ; ++class_idx)
			{
				if(test_y_(idx , class_idx) == 1)
				{
					actual_val = class_idx - 1;
					break;
				}
			}
			dout << actual_val << std::endl;	
			if(prediction_(idx, 1) == actual_val)
			{
				correct++;
			}
		}

	}


	std::cout << " NUMBER OF CORRECT PREDICTIONS = " << correct << " OUT OF " << num_test_samples << std::endl;
	std::cout << " Accurary % " << ( (correct) / (double) num_test_samples ) * 100 << std::endl;

}


template <typename T>
class naive_bayes : public classifier_base<T> // Should the base class also have a template specialization or will the super class do ? 
{
	public:
		naive_bayes() : classifier_base<T>()
		{

		}

		void train(float percentage) override final;
		std::pair<matrix<T> , matrix<T> > inference() override final;

	private : 
		using occurance = std::size_t;
		using class_index = std::size_t;
		using feature_index = std::size_t;

		// These objects should persist between multiple calls of the function ??? 
		
		// to compute p(y)
		std::unordered_map<class_index,occurance , hash_fctor> map_class_occurance;  // maintain mapping between class and count of its occurances in the training set

		// to compute p(f_i)
		std::unordered_map<feature_index, occurance , hash_fctor> map_feature_occurance ; // maintain mapping between each feature and the number of times it appears in the data set

		using feature_in_class = std::pair<size_t , size_t>;

		// to compute p(f_i | y)
		std::unordered_map< feature_in_class , occurance , hash_fctor> map_feature_in_class_occurance ;  // mapping between the occurance of each feature in a particular class
};


/*
 *  Run the training data on the set training set
 * This navie bayes will run at an abstracted level, just pass in the feature vector and the output and will train on them.
 *	The training data, each row correponds to a new sample. 
 *	The training data y is encoded in a one shot algorithm. 	
 *		So if class at index 1 is the output, then the value at index 1 witll be 1 and 0 everywhere else.
 */
template <typename T>
void naive_bayes<T>::train(float percentage)
{

	// check number of traning samples are consistent
	if(classifier_base<T>::train_x_.numRows() != classifier_base<T>::train_y_.numRows())
	{
		std::invalid_argument(std::string("ex is dead : mismatch ; Make sure the training set has equal number of smaples in x and y ") + std::string( " in ") + std::string("naive_bayes_train() ") + std::string( __FILE__) + std::string(" : ") + std::to_string(__LINE__) );
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
	for(size_t train_sample = 1; train_sample <= classifier_base<T>::train_x_.numRows() ; ++train_sample) // each row in the matrices
	{

		if(percentage * classifier_base<T>::train_x_.numRows() < train_sample)
		{
			classifier_base<T>::num_samples = percentage * classifier_base<T>::train_x_.numRows();
			break;
		}

		size_t class_idx = 1;
		
		dout << train_sample << std::endl;
		// first go over the y portion of the data set to find the class
		for(class_idx = 1  ; class_idx <= classifier_base<T>::num_classes ; ++class_idx)
		{
			T val = classifier_base<T>::train_y_(train_sample , class_idx );		 // the matrix is 1 indexed, this is odd for cs. But is standard in math > what is better ? 
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
		for(size_t feature_idx = 1 ; feature_idx <= classifier_base<T>::num_features ; ++feature_idx)
		{
			T val = classifier_base<T>::train_x_(train_sample , feature_idx); 
				
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
std::pair<matrix<T> , matrix<T> > naive_bayes<T>::inference()
{

	matrix<T> res; // the matrix will be of size # test samples * num classes

	size_t num_test_samples = classifier_base<T>::test_x_.numRows();

	// Now compute the class prediction for each test sample
	matrix<T> prediction(num_test_samples , 1);  
	classifier_base<T>::prediction_.resize(num_test_samples , 1);

	// Go through each element in the features of x and from compuute the probabilites
	for(size_t test_sample = 1; test_sample <= num_test_samples; ++test_sample) // each row in the matrices
	{

		matrix<T> sub_mat(1, classifier_base<T>::num_classes); // this row vector will be concatenated to the end of the res
		// fill them with P(Y)
		dout << test_sample << std::endl;

		for(size_t class_idx = 1 ; class_idx <= classifier_base<T>::num_classes ; ++class_idx)
		{
			dout << classifier_base<T>::num_classes << " " << classifier_base<T>::num_samples << " " << map_class_occurance[class_idx];

			sub_mat(1,class_idx) =  normalize_laplace(map_class_occurance[class_idx] , classifier_base<T>::num_classes , classifier_base<T>::num_samples );  // number of occuracnes of the given clas

			dout << sub_mat  << std::endl;

			//sub_mat(1 , class_idx) = std::log(sub_mat(1 , class_idx));
		}

		dout << sub_mat  << std::endl;
		size_t num_features = classifier_base<T>::test_x_.numCols(); 
		/*
		 * p(y , f_i ) = p(y) * sigma{ p(f_i|y) }
		 */
		for(size_t feature_idx = 1 ; feature_idx <=  num_features; ++feature_idx)
		{
			T val = classifier_base<T>::test_x_(test_sample , feature_idx); 

			if(val == 1) // The feature is present and we will have to compute the p(f_i | y)
			{
			// if the feature is not present, then it does not give up any way to update the probability of which class to choose from
			// we have to compute for each class. That is given this feature, what is the probability of seeeing a partucular class	
				
				for(size_t class_idx = 1 ; class_idx <= classifier_base<T>::num_classes ; ++class_idx)
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
		
		dout << sub_mat  << std::endl;


		// Use softMax and select max to do prediction on what is the best class to be taken

		// normalize using softmax. squishes everything to lie in the 0,1 range and the total sum = 1. 
		// std::max_element + distance to find the index of with the largest probaility . 
		// Add one since the output of distance is 0 index, while the classes are 1 indexed 
	
		sub_mat = softmax(sub_mat);

		dout << sub_mat  << std::endl;

		res.addRow(sub_mat); // add the probabilities over the different classes 


		prediction(test_sample , 1) = sub_mat.arg_max();

		dout << sub_mat << std::endl;
		dout << prediction << std::endl;
	}

	classifier_base<T>::prediction_ = prediction;
	return std::make_pair( res , prediction );
}

/*
	How to architect this  ?
	pass in the paramter struct. call train on object and the parameters will be set and 
	calls will be routed in the proper manner ? 


*/



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
class perceptron : public classifier_base<T>
{

	public : 

		void train(float percentage) override final;
		void train(float percentage , int iter) override final;

		void inference() override final;




	private : 

		//percetron internals
	 	matrix<T> perceptron_weight_;

		std::pair<bool,T> single_preceptron(const matrix<T>& feature , const matrix<T>& weight , T threshold  = 0 ) const;

		std::pair<matrix<T> , matrix<T>> multi_class_perceptron_inference();

		std::pair<matrix<T> , matrix<T>> perceptron_update(const matrix<T>& predicted_id_weight ,const matrix<T>& actual_id_weight ,const matrix<T>& feature_vec);
		std::pair<matrix<T> , matrix<T>> mira_perceptron_update(const matrix<T>& predicted_id_weight , const matrix<T>& actual_id_weight , matrix<T>& feature_vec , double cap = MIRA_CAP);

		void train(perceptron_type type = perceptron_type::simple , float percentage = 100);
		void multi_class_perceptron_train_iter(perceptron_type type , float percentage , int num_iter = 100);



};




/*
 * A single percetron model 
 * Plug this model into others to obtain better results.
 *
 * if feauture * weight >= threshold then 1 else 0
 *
 * intput : the feature (row vector), the weight vector (column vector) , double threshould
 * output : 0 or 1 representing whether the neuroing has fired or not 
##### */
template <typename T>
std::pair<bool, T> perceptron<T>::single_preceptron(const matrix<T>& feature , const matrix<T>& weight , T threshold ) const
{
	dout << "features give to single perc " << feature ;

	dout << "weight to  single perc " << weight ;

	T res = feature.innerProduct(weight);

	dout << "value of inner prod in single perc " << res << std::endl;
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
std::pair<matrix<T> , matrix<T>> perceptron<T>::perceptron_update(const matrix<T>& predicted_id_weight ,const matrix<T>& actual_id_weight ,const matrix<T>& feature_vec)
{
	dout << "features in perc update " << feature_vec;

	dout << "weight in class predicted for the feature || perc update " << predicted_id_weight ;

	dout << "weight of actual class in training set || perc update" << actual_id_weight ;

	matrix<T> predicted_id_weight_result = predicted_id_weight - feature_vec;

	dout << "weight obtained after updated the predicted weight by subtracting features  || prec weight update " << predicted_id_weight_result;

	matrix<T> actual_id_weight_result  = actual_id_weight + feature_vec;

	dout << "weight obtained after udpated the actualy wegiht by adding the feature vector || actual id result " << actual_id_weight_result; 

	return std::make_pair(predicted_id_weight_result , actual_id_weight_result);	
}	

/*
 * perceptron update with mira
 */
template <typename T>
std::pair<matrix<T> , matrix<T>> perceptron<T>::mira_perceptron_update(const matrix<T>& predicted_id_weight , const matrix<T>& actual_id_weight , matrix<T>& feature_vec , double cap )
{

	T tau = (predicted_id_weight - actual_id_weight).innerProduct(feature_vec.transpose());

	tau += 1;
	tau /= 2 * feature_vec.normEuclidean();

	tau = std::min(cap , tau);

	matrix<T> predicted_id_weight_result = predicted_id_weight - feature_vec * tau;
	matrix<T> actual_id_weight_result  = actual_id_weight + feature_vec * tau;

	return std::make_pair(predicted_id_weight_result , actual_id_weight_result);	
}

template <typename T>
void perceptron<T>::multi_class_perceptron_train_iter(perceptron_type type , float percentage , int num_iter)
{
	//Resize the weight vector to hold # classes rows and #features columns
	// A weight vector for each of the classes 
	// the weight vector will be used to determine how much the neuron will look at each feature
	// Initially all the weights are 0
	perceptron_weight_.resize(classifier_base<T>::num_classes , classifier_base<T>::num_features + 1 , 0 ); // + 1 for the biases 

	perceptron_weight_(1 , classifier_base<T>::num_features + 1) = 1 ; // set the bias for the class to be 1, so the tie can be broken for arg_max, when the algorithm starts are the weight vector is filled with 0


	dout << "WEIGHT MATRIX " <<  perceptron_weight_ << std::cout ; 
	for(int i = 0; i < num_iter ; ++i)
	{
		dout << "WEIGHT MATRIX " <<  perceptron_weight_ << std::cout ; 
		std::cout << " ITERATION : # " << i << std::endl;
		train(type , percentage);
	}
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
void dmlpack<T>::multi_class_perceptron_train(perceptron_type type , float percentage)
{

	const size_t num_train_samples = train_x_.numRows();
	dout << train_x_.numRows() << " " ;


	// Now go through the data set and fill in these values
	for(size_t train_sample = 1; train_sample <= num_train_samples ; ++train_sample) // each row in the matrices
	{
		dout << train_sample << std::endl;

		if(percentage * train_x_.numRows() < train_sample)
		{
			break;
		}

		// get the feature vector
		matrix<T> feature_vec = train_x_.returnRow(train_sample);	

		// Append the +1 towards its end. 
		feature_vec.resize(1 , feature_vec.numCols() + 1);
		feature_vec(1 , feature_vec.numCols()) = 1;

		dout << "FEATURE VEC : " << feature_vec << std::endl;

		// Okay two seperate ways to implement this. 
		// Update the weight of all the classes. 
		// Update the weight of the class with the max weight
		// and also the class that was predicted .
	
		matrix<T> class_pred(1,num_classes);

		dout << " CLASS PRED : " << class_pred ; 

		size_t actual_class_id = 0 ; 

		// first go over the y portion of the data set to find the actual class
		// This for loop finds out what the predictied class is and also what the acutal class is
		for(size_t class_idx = 1  ; class_idx <= num_classes ; ++class_idx)
		{
			dout << class_idx << " " << num_classes << std::endl;
			// get the weight vector for a particular class
			matrix<T> weight_vec = perceptron_weight_.returnRow(class_idx);			

			dout << " weight_actual_feature " << weight_vec ;

			std::pair<bool, T> pred = single_preceptron(feature_vec , weight_vec);
		
			dout << " single _percrption " << pred.second <<std::endl;	

			class_pred(1,class_idx) = pred.second;


			dout << weight_vec << std::endl;

			bool actual = train_y_(train_sample , class_idx); 	

			if(actual)
			{
				actual_class_id = class_idx - 1;
			}
			
		}

		dout << "actual pred " << actual_class_id << std::endl;

		dout << "class pred " << class_pred << std::endl;	

		auto predicted_class_idx = class_pred.arg_max();

		dout << "class pred arg " << predicted_class_idx << std::endl;	

		// update the weight vectors
		// if the predicted class and the acutal class are not the same,
		// then we reduce the weight vector for the predicted class 
		// and increase the weight vector for the actual class .
		if(predicted_class_idx != actual_class_id)
		{

			dout << "error in prediction updating the weight vectors " << std::endl;

			dout << "actual perceptron weight " << perceptron_weight_ ;

			// reduce the weight vector for the predicted class 
			matrix<T> reduced_weight = perceptron_weight_.returnRow(predicted_class_idx + 1);		
		
			dout << "reduced weight before " << reduced_weight ;
			
			// increase the weight vector for the actual class .
			matrix<T> increase_weight = perceptron_weight_.returnRow(actual_class_id + 1);		

			dout << "increased weight before " << increase_weight ;

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

			dout << "reduced weight after " << reduced_weight ;

			dout << "increased weight afer " << increase_weight ;

			perceptron_weight_.replaceRow(reduced_weight , predicted_class_idx + 1);
			perceptron_weight_.replaceRow(increase_weight, actual_class_id + 1);
			
			dout << "weigth matrix after " << perceptron_weight_ << std::endl;

		}	
	}	
}




template <typename T>
std::pair<matrix<T> , matrix<T>> dmlpack<T>::multi_class_perceptron_inference()
{
	matrix<T> res; // the matrix will be of size # test samples * num classes

	size_t num_test_samples = test_x_.numRows();

	// Now compute the class prediction for each test sample
	matrix<T> prediction(num_test_samples , 1);  

	// Go through each element in the features of x and from compuute the probabilites
	for(size_t test_sample = 1; test_sample <= num_test_samples; ++test_sample) // each row in the matrices
	{
		// get the feature vector
		matrix<T> feature_vec = test_x_.returnRow(test_sample);	

		// Append the +1 towards its end. 
		feature_vec.resize(1 , feature_vec.numCols() + 1);
		feature_vec(1 , feature_vec.numCols()) = 1;

		dout << " inference " << feature_vec;

		matrix<T> sub_res(1 , num_classes);

		// compute the similiarty between the current feature and each of the classes
		for(int class_idx = 1 ; class_idx <= num_classes ; ++class_idx)
		{
			sub_res(1,class_idx)  = perceptron_weight_.returnRow(class_idx).innerProduct(feature_vec);
		}
		
		dout << " result matrix " << sub_res;

		

		res.addRow(sub_res); // add the probabilities over the different classes 
	
		dout << " total result " << res ;
		// Use softMax and select max to do prediction on what is the best class to be taken

		// std::max_element + distance to find the index of with the largest probaility . 
		// Add one since the output of distance is 0 index, while the classes are 1 indexed 
	
		prediction(test_sample , 1) = sub_res.arg_max();

		dout << " prediction udpates " << prediction;

	}

	prediction_ = prediction; // keep track of the prediciton that was made to test the accuracy

	return std::make_pair( res , prediction );
}









template <typename T>
class perceptron<T>::train
{

emplate <typename T>
void dmlpack<T>::multi_class_perceptron_train(perceptron_type type , float percentage)
{

	const size_t num_train_samples = train_x_.numRows();
	dout << train_x_.numRows() << " " ;

// Now go through the data set and fill in these values
	for(size_t train_sample = 1; train_sample <= num_train_samples ; ++train_sample) // each row in the matrices
	{
		dout << train_sample << std::endl;

		if(percentage * train_x_.numRows() < train_sample)
		{
			break;
		}

		// get the feature vector
		matrix<T> feature_vec = train_x_.returnRow(train_sample);	

		// Append the +1 towards its end. 
		feature_vec.resize(1 , feature_vec.numCols() + 1);
		feature_vec(1 , feature_vec.numCols()) = 1;

		dout << "FEATURE VEC : " << feature_vec << std::endl;

		// Okay two seperate ways to implement this. 
		// Update the weight of all the classes. 
		// Update the weight of the class with the max weight
		// and also the class that was predicted .
	
		matrix<T> class_pred(1,num_classes);

		dout << " CLASS PRED : " << class_pred ; 

		size_t actual_class_id = 0 ; 

		// first go over the y portion of the data set to find the actual class
		// This for loop finds out what the predictied class is and also what the acutal class is
		for(size_t class_idx = 1  ; class_idx <= num_classes ; ++class_idx)
		{
			dout << class_idx << " " << num_classes << std::endl;
			// get the weight vector for a particular class
			matrix<T> weight_vec = perceptron_weight_.returnRow(class_idx);			

			dout << " weight_actual_feature " << weight_vec ;

			std::pair<bool, T> pred = single_preceptron(feature_vec , weight_vec);
		
			dout << " single _percrption " << pred.second <<std::endl;	

			class_pred(1,class_idx) = pred.second;


			dout << weight_vec << std::endl;

			bool actual = train_y_(train_sample , class_idx); 	

			if(actual)
			{
				actual_class_id = class_idx - 1;
			}
			
		}

		dout << "actual pred " << actual_class_id << std::endl;

		dout << "class pred " << class_pred << std::endl;	

		auto predicted_class_idx = class_pred.arg_max();

		dout << "class pred arg " << predicted_class_idx << std::endl;	

		// update the weight vectors
		// if the predicted class and the acutal class are not the same,
		// then we reduce the weight vector for the predicted class 
		// and increase the weight vector for the actual class .
		if(predicted_class_idx != actual_class_id)
		{

			dout << "error in prediction updating the weight vectors " << std::endl;

			dout << "actual perceptron weight " << perceptron_weight_ ;

			// reduce the weight vector for the predicted class 
			matrix<T> reduced_weight = perceptron_weight_.returnRow(predicted_class_idx + 1);		
		
			dout << "reduced weight before " << reduced_weight ;
			// increase the weight vector for the actual class .
			matrix<T> increase_weight = perceptron_weight_.returnRow(actual_class_id + 1);		

			dout << "increased weight before " << increase_weight ;

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

			dout << "reduced weight after " << reduced_weight ;

			dout << "increased weight afer " << increase_weight ;

			perceptron_weight_.replaceRow(reduced_weight , predicted_class_idx + 1);
			perceptron_weight_.replaceRow(increase_weight, actual_class_id + 1);
			
			dout << "weigth matrix after " << perceptron_weight_ << std::endl;

		}	
	}	
}










}







template <typename T>
class perceptron<T>::inference
{



















}






























































template <typename T>
class dmlpack
{
	public:	
		dmlpack(CLASSIFIER_TYPE ml_type) : ml_type_(ml_type) , num_samples(0)
		{
			
		};	
		
		// diable the copy cstor and copy assignment cstor
		dmlpack(const dmlpack& ) = delete ;
		dmlpack& operator=(const dmlpack& ) = delete;

		~dmlpack()
		{
			std::cout << "dmlpack destroyed " << std::endl;
		}


		// feed the entire training data
		// keep reference to the class, instead of copying the data set
		// This means that the data should outlive the class
		void feed_train_data( matrix<T> train_x ,  matrix<T> train_y) 
		{
			train_x_ = train_x; 
			train_y_ = train_y;	

			// Number of classes to classify the data into 
			num_classes = train_y_.numCols();

			num_features = train_x_.numCols();

			num_samples = train_x_.numRows();

		}
	

		// feed the entire test data 
		void feed_test_data(matrix<T> test_x , matrix<T> test_y)
		{
			test_x_ = test_x;	
			test_y_ = test_y;
		}

		// train on the percentage of the data set provided
		void train(float percentage = 1, int iter = 4);	


		// set the test set on the model and give back the accuracy 		
		double test();	

		void test_accuracy();

		// run inference , take in data point and see what the model predicts
		matrix<T> inference(matrix<T>& test_x);

		// test set already set
		std::pair<matrix<T>,matrix<T>> inference();
		
		// pass in the index of the test set where you want the inference to be done.
//		matrix<T> inference(float percent_test);	


		// train on a batch of data 
		// Two days this can be done. 
		// 	Pass in a matrix with the data to be trained on 
		// 		OR 
		//  	Pass in some sort of index or identifier, which will help us decide which part of the matrix we want to train on	
		void train_on_batch();
	

	private:
		CLASSIFIER_TYPE ml_type_; // machine learning type

		// training data
		matrix<T> train_x_;			
		matrix<T> train_y_;			

		// testing data
		matrix<T> test_x_;			
		matrix<T> test_y_ ; //While testing only the x labels are given and we have to predict the y values from that 

		matrix<T> prediction_;


		// in a multi batch train scenario ne need to keep track of the number of samples we have seen
		std::size_t num_samples;  
		std::size_t num_classes;
		std::size_t num_features ;


		// multi layer neural network internals

		// will hold the different types of neural layers	
		std::vector<matrix<T>> network_layers_;

		void multi_layer_nn_train(double learning_rate = 0.0001, size_t iterations = 100);

		T sigmoid(T val);

		// single layer neural network 

		matrix<T> single_layer_nn_weigh_;			 // single layer neural network weights

		void single_layer_nn_train(double learning_rate = 0.0001, size_t iterations = 1);

		
		// Normalize with laplace smoothing
		T normalize_laplace(size_t class_occurance  , size_t total_classes , size_t total_occurances ,  size_t strength = 1)  // pretend there is a uniform distribution of the data. Then update it based on evidence
		{ 
			return (double ) ( (class_occurance + strength) / (double) (total_occurances + total_classes * strength) ) ;
		};	


		matrix<T> softmax(matrix<T>& prob) const ;// has to be applied on a column vector


};


template <typename T>
void dmlpack<T>::train(float percentage , int iter)
{
	switch(ml_type_)
	{
		case(CLASSIFIER_TYPE::NAIVE_BAYES):
			naive_bayes_train(percentage);
			break;

		case(CLASSIFIER_TYPE::PERCEPTRON):
			multi_class_perceptron_train_iter(perceptron_type::simple , percentage , iter);
			break;

		case(CLASSIFIER_TYPE::PERCEPTRON_MIRA):
			multi_class_perceptron_train_iter(perceptron_type::mira , percentage, iter);
			break;

		case(CLASSIFIER_TYPE::SINGLE_LAYER_NN):
			single_layer_nn_train();
			break;

		case(CLASSIFIER_TYPE::MULTI_LAYER_NN):
			multi_layer_nn_train();
			break;	

	}


}


template <typename T>
std::pair<matrix<T>,matrix<T>> dmlpack<T>::inference()
{
	switch(ml_type_)
	{
		case(CLASSIFIER_TYPE::NAIVE_BAYES):
			return naive_bayes_inference();
			break;

		case(CLASSIFIER_TYPE::PERCEPTRON):
		case(CLASSIFIER_TYPE::PERCEPTRON_MIRA):
			return multi_class_perceptron_inference();
			break;
	
		case(CLASSIFIER_TYPE::SINGLE_LAYER_NN):
			single_layer_nn_train();
			break;

		case(CLASSIFIER_TYPE::MULTI_LAYER_NN):
			multi_layer_nn_train();
			break;	

	}


}

template <typename T>
matrix<T> dmlpack<T>::softmax(matrix<T>& prob) const // has to be applied on a column vector
{
	matrix<T> res(prob.numRows() , prob.numCols());	
	dout << "softmax" ;

	double norm_factor = 0 ; 	
	
	for(int idx = 1 ; idx <= prob.size() ; ++idx)
	{
		if(prob.isColVector())
		{
			norm_factor += std::exp(prob(idx,1));
		}
		else
		{
			norm_factor += std::exp(prob(1,idx));
		}

	}

	// Now compute e^(x) / sigma e^(x_i)
	for(int idx = 1 ; idx <= prob.size() ; ++idx)
	{
		if(prob.isColVector())
		{
			res(idx,1) =  std::exp(prob(idx,1)); 

			// normalize
			res(idx,1) /= norm_factor;
		}
		else
		{
			res(1,idx) =  std::exp(prob(1,idx)); 

			// normalize
			res(1,idx) /= norm_factor;
		}
	}
	dout << res << std::endl;
	return res;
}


/* 
 * RUn the tained model on the given test data and gives back the accuracyt . 
 * 	How to measure accuracy ? For now just count the number of times the prediction and the actual y matches. 
 */

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

			//incremental change

			single_layer_nn_weigh_ = single_layer_nn_weigh_ + ( ( actual_output_vec - pred_output_vec ) * feature_vec) * learning_rate; 

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
					output = qu.front();
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

				auto current_layer_activation = qu.front() ; qu.pop();
				auto prev_layer_activation = qu.front() ;
				
				// stores the gradient for the current layer which is fed back into lower layers	
				matrix<T> current_layer_grad(current_layer_activation.numRows() , current_layer_activation.numCols());

				bool output_layer_flag = (itr == --network_layers_.end());
					

				itr->transform_inplace([&](std::size_t row , std::size_t col, int val)
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
			
			single_layer_nn_weigh_ = single_layer_nn_weigh_ +  ( ( actual_output_vec - pred_output_vec ) * feature_vec) * learning_rate; 

		}
	}



}





#endif
