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


		// naive bayes //internals	------------------------------------
		void naive_bayes_train();
		std::size_t num_samples;  // in a multi batch train scenario ne need to keep track of the number of samples we have seen
					      //------------------------------------
		std::size_t num_classes;
		std::size_t num_features ;


		// Normalize with laplace smoothing
		T normalize_laplace(T val) 
		{ 
			return (double ) ( (val + 1) / (num_classes + 1) ) ;
		};	


		matrix<T> softmax(matrix<T> prob) // has to be applied on a column vector
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



		using occurance = std::size_t;

		using class_index = std::size_t;

		using feature_index = std::size_t;

		// These objects should persist between multiple calls of the function
		// TO DO : Move them into class memebers 
		


		// to compute p(y)
		static std::unordered_map<class_index,occurance , hash_fctor> map_class_occurance;  // maintain mapping between class and count of its occurances in the training set

		// to compute p(f_i)
		static std::unordered_map<feature_index, occurance , hash_fctor> map_feature_occurance ; // maintain mapping between each feature and the number of times it appears in the data set

		using feature_in_class = std::pair<size_t , size_t>;

		// to compute p(f_i | y)
		static std::unordered_map< feature_in_class , occurance , hash_fctor> map_feature_in_class_occurance ;  // mapping between the occurance of each feature in a particular class



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
 *
 */

template <typename T>
std::pair<matrix<T> , matrix<T> > dmlpack<T>::naive_bayes_inference()
{


	matrix<T> res; // the matrix will be of size # test samples * num classes



	// Go through each element in the features of x and from compuute the probabilites
	for(size_t test_sample = 1; test_sample <= test_x_.numRows() ; ++test_sample) // each row in the matrices
	{

		matrix<T> sub_mat(1, num_classes); // this row vector will be concatenated to the end of the res

		// fill them with P(Y)
	
		for(size_t class_idx = 1 ; class_idx < num_classes ; ++class_idx)
		{
			sub_mat(1,class_idx) = map_class_occurance[class_idx];  // number of occuracnes of the given clas
			
			// convert to probability by normalizing
			sub_mat(1 , class_idx) /= num_samples;   	

			/*
			 * Normalizing here, instead of when the insertion into the map happens, enables us to train in batches
			 * the num_samples is updated when a new batch is added
			 * and the occurances of each of the classes are also updated.
			 */
		}
	

		/*
		 * p(y , f_i ) = p(y) * sigma{ p(f_i|y) }
		 */


		for(size_t feature_idx = 1 ; feature_idx < test_x_.numCols() ; ++feature_idx)
		{
			T val = test_x_(test_sample , feature_idx); 

			if(val == 1) // The feature is present and we will have to compute the p(f_i | y)
			{
			// if the feature is not present, then it does not give up any way to update the probability of which class to choose from
				// we have to compute for each class. That is given this feature, what is the probability of seeeing a partucular class	
				
				for(size_t class_idx = 1 ; class_idx < num_classes ; ++num_classes)
				{
					// compute p(f_i / y) , by counting the occurance of a feature for the partucular class and dividing it by the total occurance of that feature
					sub_mat(1,class_idx) *= map_feature_in_class_occurance[std::make_pair(feature_idx , class_idx)]; 

					sub_mat(1,class_idx) /= map_feature_occurance[feature_idx];
				}

			}


		}


	}


}

