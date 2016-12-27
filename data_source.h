#ifndef DATA_SOURCE_H
#define DATA_SOURCE_H

/*
 * Reads in the data and create matrixes out of it. 
 *
 * Requried format of the matrixes 
 *
 * Each row represents a different traning examples. 
 * 	So each column will represent a different freature
 *
 * In case of the output, labals, There are one hot encoded.
 *
 *  Dec 17, 2016
 *  	For now specialize for handing the data set from berkley, 
 *  	The library will be generalized later on
 */

#include "matrix.hpp"
#include <Python.h>
#include <string>
#include <stdexcept>
#include <iostream>

//#define DEBUG_P

#ifdef DEBUG_P
#define dout std::cout << __FILE__<< " (" << __LINE__ << ") " << "DEBUG : "
#else
#define dout 0 && std::cout
#endif

enum class BRKLY_DATA { DIGIT , FACE };
enum class DATA_TYPE { TRAIN , VALID , TEST };


class data_source
{
	public:
		/*
		 * Read the berkely data set and store it in the classes
		 */
		void read_store_berkely_data(BRKLY_DATA data , DATA_TYPE type);	
	
		data_source() 
		{

		}

		// defaul constructor and destructor will do 	
		// prevent copies being mate
		data_source(const data_source& ) = delete;
		data_source& operator=(const data_source& ) = delete;	

		matrix<double> get_features()
		{
			return x_data_;
		}

		matrix<double> get_labels()
		{
			return y_data_;
		}

	private:
		matrix<double> x_data_;
		matrix<double> y_data_;	
		void add_py_feature_list(PyObject * container);
		// one hot encoding
		void add_py_label_list(PyObject * container, const int num_classes);

};

#endif

