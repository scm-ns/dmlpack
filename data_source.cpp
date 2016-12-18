#include "data_source.h"
#include <stdlib.h>


char * DATA_LOADER_PYTHON_CLASS = (char*)"samples\0";


// Pass in a py obj that is pointing to a list or typle in python and a std::vec will be 
// returned containing the data
std::vector<double> data_source::py_list_tuple_to_vec(PyObject * container)
{

	if(PyTuple_Check(container))
	{
		for(Py_ssize_t i = 0 ; i < PyTuple_Size(container) ; ++i)
		{
			PyObject* val = PyTuple_GetItem(container, i);

			PyObject* get_pixl = PyObject_GetAttrString(val,"getPixel");

			PyObject* height = PyObject_GetAttrString(val,"height");
			PyObject* width = PyObject_GetAttrString(val,"width");

			for(int i = 0 ; i < PyInt_AsLong(height) ; ++i)
			{

				for(int j = 0 ; j < PyInt_AsLong(width) ; ++j)
				{
					PyObject * temp = PyObject_CallFunction(get_pixl, (char *)"(ii)" , j,i);
					//data.push_back( PyFloat_AsDouble(temp));

					//std::cout << data.back() << std::endl;	
				}
			}	


		}

	}
	else if(PyList_Check(container))
	{

		// Get the heigt and size data
		PyObject* val = PyList_GetItem(container, 10);
		PyObject* height = PyObject_GetAttrString(val,"height");
		PyObject* width = PyObject_GetAttrString(val,"width");


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
					std::cout << x_data_(sample + 1 , idx + 1) ;
					++idx;

				}
			}	

			std::cout << std::endl;
		}

	}
	else
	{
		throw std::logic_error("object ptr not a list or tuple");
	}
	return std::vector<double>{};
}



void data_source::read_store_berkely_data(BRKLY_DATA data , DATA_TYPE type)
{
	setenv("PYTHONPATH",".",1);

	Py_Initialize();


	//PyRun_SimpleString("import sys");
	//PyRun_SimpleString("sys.path.append(\".\")");

	PyObject* data_loader_module_name = PyString_FromString(DATA_LOADER_PYTHON_CLASS);

	PyObject* data_loader_module = PyImport_Import(data_loader_module_name);

	if(!data_loader_module)
		throw std::invalid_argument("module not found");

	PyObject* load_data_func = PyObject_GetAttrString(data_loader_module , (char*)"load_digit_train_x");
	if(!load_data_func || !PyCallable_Check(load_data_func))
		throw std::invalid_argument("function not found");
	PyObject* train_list = PyObject_CallObject(load_data_func , NULL);


	py_list_tuple_to_vec(train_list);	



	Py_XDECREF(train_list);
	Py_XDECREF(data_loader_module);
	Py_Finalize();

}

