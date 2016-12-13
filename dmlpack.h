/*
 *
 * Might end up being a header only class or namespace 
 *
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


#include "matrix.h"
#include "debug.h"


/*
 * Start : Dec 10 th: 
 * Memory Strategy : As long as I am not doing any real threading or cuda issues. I am going to make the C++ lib handle all the memory. 
 * 	> That is I will not explicitly allocate memory, but use the start contrainers to do the dirty work for me.
 *
 */


enum class classifier_type { naive_bayes , perceptron , multi_layer_nn };
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
struct hash_fctor; // functor

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
		matrix<T>& train_x_;			
		matrix<T>& train_y_;	

		// testing data
		matrix<T>& test_x_ ; 
//		matrix<T>& test_y_ ; While testing only the x labels are given and we have to predict the y values from that 




		//percetron internals
		std::pair<bool,T> single_preceptron(const matrix<T>& feature , const matrix<T>& weight , T threshold  = 0 ) const;

		std::pair<matrix<T> , matrix<T>> multi_class_perceptron_inference();

		void multi_class_perceptron_train();

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
 *
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
void dmlpack<T>::multi_class_perceptron_train()
{

	const size_t num_train_samples = train_x_.numRows();

	//Resize the weight vector to hold # classes rows and #features columns
	// A weight vector for each of the classes 
	// the weight vector will be used to determine how much the neuron will look at each feature
	// Initially all the weights are 0
	perceptron_weight_.resize(num_classes , num_features + 1 , 0 ); // + 1 for the biases 


	// Now go through the data set and fill in these values
	for(size_t train_sample = 1; train_sample <= num_train_samples ; ++train_sample) // each row in the matrices
	{

		size_t class_idx = 1;

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

		// first go over the y portion of the data set to find the class
		for(class_idx = 1  ; class_idx <= num_classes ; ++class_idx)
		{

			// get the weight vector for a particular class
			matrix<T> weight_vec = perceptron_weight_.returnRow(class_idx);			
			bool pred = single_preceptron(feature_vec , weight_vec);
			
			bool actual = train_y_(train_sample , class_idx); 	

			// check if the classification is correct
			// if correct do nothing.
			// else update the weight vector for this particular class
			if(pred != actual)
			{
				weight_vec = weight_vec + actual * feature_vec;
				perceptron_weight_.replaceRow(weight_vec);
			}

		}

	}	


}





