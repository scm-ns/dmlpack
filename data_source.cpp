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
		PyObject* val = PyList_GetItem(container, 10);
		PyObject* height = PyObject_GetAttrString(val,"height");
		PyObject* width = PyObject_GetAttrString(val,"width");

		// expand the data storage to increase in size for the data
		x_data_.resize(PyList_Size(container) , PyInt_AsLong(height) * PyInt_AsLong(width));  

		for(Py_ssize_t sample = 0 ; sample < PyList_Size(container) ; ++sample)
		{
			PyObject* val = PyList_GetItem(container, sample);

			PyObject* get_pixl = PyObject_GetAttrString(val,"getPixel");
			
			int idx = 0 ;
			for(int row  = 0 ; row < PyInt_AsLong(height) ; ++row)
			{

				for(int col = 0 ; col < PyInt_AsLong(width) ; ++col)
				{
					PyObject * temp = PyObject_CallFunction(get_pixl,(char *)"(ii)" , col,row);

					x_data_(sample + 1, idx  + 1) = PyFloat_AsDouble(temp);
					++idx;

				}
			}	

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


	//PyRun_SimpleString("import sys");
	//PyRun_SimpleString("sys.path.append(\".\")");

	PyObject* data_loader_module_name = PyString_FromString(DATA_LOADER_PYTHON_CLASS.c_str());

	PyObject* data_loader_module = PyImport_Import(data_loader_module_name);

	if(!data_loader_module)
		throw std::invalid_argument("module not found");

	std::string feature_func_to_call_str; 
	std::string label_func_to_call_str ;

	if(data == BRKLY_DATA::DIGIT)
	{
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
	
	// Load the features
	PyObject* load_data_func_x = PyObject_GetAttrString(data_loader_module , feature_func_to_call_str.c_str());
	if(!load_data_func_x || !PyCallable_Check(load_data_func_x))
		throw std::invalid_argument("function not found");
	PyObject* feature_list = PyObject_CallObject(load_data_func_x , NULL);

	add_py_feature_list(feature_list);


	// Load the labels
	PyObject* load_data_func_y = PyObject_GetAttrString(data_loader_module , label_func_to_call_str.c_str());

	if(!load_data_func_y || !PyCallable_Check(load_data_func_y))
		throw std::invalid_argument("function not found");

	PyObject* label_list = PyObject_CallObject(load_data_func_y , NULL);

//	add_py_label_list(label_list);	


	// decrement all the reference counters
	Py_XDECREF(data_loader_module_name);
	Py_XDECREF(data_loader_module);
	Py_XDECREF(load_data_func_x);
	Py_XDECREF(load_data_func_y);
	Py_XDECREF(label_list);
	Py_XDECREF(feature_list);
	Py_Finalize();

}

