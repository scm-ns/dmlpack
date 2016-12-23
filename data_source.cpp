#include "data_source.h"
#include <stdlib.h>


const std::string DATA_LOADER_PYTHON_CLASS = "samples";

// Pass in a py obj that is pointing to a list or typle in python and a std::vec will be 
// returned containing the data
void data_source::add_py_feature_list(PyObject * container)
{
	if(PyList_Check(container))
	{

		// Get the heigt and size data
		PyObject* datum = PyList_GetItem(container, 10);
		PyObject* height = PyObject_GetAttrString(datum,"height");
		PyObject* width = PyObject_GetAttrString(datum,"width");

		
		const Py_ssize_t sample_size = PyList_Size(container);
		dout << "SAMPLE SIZE " << sample_size << std::endl;		

		// expand the data storage to increase in size for the data
		x_data_.resize(sample_size , PyInt_AsLong(height) * PyInt_AsLong(width));   // the number of features is #rows * #cols

		for(Py_ssize_t sample = 0 ; sample < sample_size; ++sample)
		{
			PyObject* datum_obj = PyList_GetItem(container, sample);
			// by reading the samples.py, we see that this is going to be a Datum class. So we extract the required information for the class	

			PyObject* get_pixl = PyObject_GetAttrString(datum_obj,"getPixel");
			
			int idx = 0 ;
			for(int row  = 0 ; row < PyInt_AsLong(height) ; ++row)
			{

				for(int col = 0 ; col < PyInt_AsLong(width) ; ++col)
				{
					PyObject * temp = PyObject_CallFunction(get_pixl,(char *)"(ii)" , col,row); // they have col first, then row odering for some reason in berkely code

					float temp_val = PyFloat_AsDouble(temp);

					if(temp_val > 0)
					{
						temp_val = 1;
					}	

					x_data_(sample + 1 , idx + 1 ) = temp_val;

					++idx; // idx will hold the column to which the current feature is being added for this specific training examples (row = samples)
				}
			}	

		}

	}
	else
	{
		throw std::logic_error("object ptr not a list or tuple");
	}
}


// adds the data stored in the list into the y_data in the class. Used to get the labels from the python side)
// num of classes is needed as an easy way to determine the size of the storage
void data_source::add_py_label_list(PyObject * container, const int num_classes)
{
	if(PyList_Check(container))
	{

		const Py_ssize_t sample_size = PyList_Size(container);
		// expand the data storage to increase in size for the ata
		y_data_.resize(sample_size ,num_classes , 0);  

		dout << "SAMPLE SIZE " << sample_size << std::endl;		

		for(Py_ssize_t sample = 0 ; sample < sample_size ; ++sample)
		{
			// here val is an int
			PyObject* int_obj = PyList_GetItem(container, sample);

			// from the value of int. determine which index should we marked as 1
			int val_int = PyInt_AsLong(int_obj);

			// val_int lies between 0 and 9 for digits, so mark the 1 to 10th position as 1 based on the result
			y_data_(sample + 1, val_int + 1) = 1;

			dout << "VAL :" << val_int << "SAMPLE # : " << sample + 1 << "STORED :" << y_data_.returnRow(sample + 1) <<  std::endl;

		}

	}
	else
	{
		throw std::logic_error("object ptr not a list or tuple");
	}
}

void data_source::read_store_berkely_data(BRKLY_DATA data , DATA_TYPE type)
{
	// required otherwise current dir will not be added to working dir and the interpreter will not be able to load modules in the current dir
	setenv("PYTHONPATH",".",1);

	Py_Initialize();

	PyObject* data_loader_module_name = PyString_FromString(DATA_LOADER_PYTHON_CLASS.c_str());

	PyObject* data_loader_module = PyImport_Import(data_loader_module_name);

	if(!data_loader_module)
		throw std::invalid_argument("module not found");

	// hold the function names. assign it dynamically based on function params
	std::string feature_func_to_call_str; 
	std::string label_func_to_call_str ;

	int num_classes = 0 ;

	if(data == BRKLY_DATA::DIGIT)
	{
		num_classes = 10;
		switch(type)
		{
			case(DATA_TYPE::TRAIN):
				feature_func_to_call_str = "load_digit_train_x";
				label_func_to_call_str = "load_digit_train_y";
				break;
			case(DATA_TYPE::TEST):
				feature_func_to_call_str = "load_digit_test_x";
				label_func_to_call_str = "load_digit_test_y";
				break;
			case(DATA_TYPE::VALID):
				feature_func_to_call_str = "load_digit_valid_x";
				label_func_to_call_str = "load_digit_valid_y";
				break;

		};


	}
	else if(data == BRKLY_DATA::FACE)
	{
		num_classes = 2;
		switch(type)
		{
			case(DATA_TYPE::TRAIN):
				feature_func_to_call_str = "load_face_train_x";
				label_func_to_call_str = "load_face_train_y";
				break;
			case(DATA_TYPE::TEST):
				feature_func_to_call_str = "load_face_test_x";
				label_func_to_call_str = "load_face_test_y";
				break;
			case(DATA_TYPE::VALID):
				feature_func_to_call_str = "load_face_valid_x";
				label_func_to_call_str = "load_face_valid_y";
				break;

		};


	}
	else
	{
		throw std::logic_error("The given data type does not exist. DONT BE STUPID");
	}
	
	// Load the features
	PyObject* load_data_func_x = PyObject_GetAttrString(data_loader_module , feature_func_to_call_str.c_str());
	if(!load_data_func_x || !PyCallable_Check(load_data_func_x))
		throw std::invalid_argument("function not found");
	// obtain the list from the python code
	PyObject* feature_list = PyObject_CallObject(load_data_func_x , NULL);
	// Convert it into vectors
	add_py_feature_list(feature_list);


	// Load the labels
	PyObject* load_data_func_y = PyObject_GetAttrString(data_loader_module , label_func_to_call_str.c_str());

	if(!load_data_func_y || !PyCallable_Check(load_data_func_y))
		throw std::invalid_argument("function not found");

	PyObject* label_list = PyObject_CallObject(load_data_func_y , NULL);

	add_py_label_list(label_list, num_classes);	


	// decrement all the reference counters
	Py_XDECREF(data_loader_module_name);
	Py_XDECREF(data_loader_module);
	Py_XDECREF(load_data_func_x);
	Py_XDECREF(load_data_func_y);
	Py_XDECREF(label_list);
	Py_XDECREF(feature_list);
	Py_Finalize();

}

