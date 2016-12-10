#pragma once

#include <functional>
#include "matrix.h"
#include <cmath>
#include "setMath.h"

typedef  std::function < double(double) > scalarFn_1;
typedef  std::function < double(double, double) > scalarFn_2;
typedef  std::function < double(double,double,double) > scalarFn_3;
typedef matrix<std::function<double(double)>> vectorFn_1;
typedef matrix<scalarFn_3> vectorFn_3; // Will be a column Vector containg scalar Function :: Otherwise Fn's will break 
typedef matrix<std::function<double(matrix<double>)>> vectorFn_N;

typedef std::function<double(matrix<double>)> scalarFn_N; 


// SET TYPEDEF 

typedef setMath<std::string> stringSet;
typedef setMath<int> intSet;
typedef setMath<double> doubleSet;


//typedef  std::function < double(double) > fn_ND;


class func
{
public:
	func();
	~func();

	// Function to be used an inputs . 
	static double testFN(double x){ return x*x*x + std::sin(x); }
	static double testFN_2(double x){ return std::cos(x); }
	static double waveEqn(double t, double A, double phi, double omega){ return A* std::sin(omega*t - phi); }
	static double waveEqnDrv2(double t, double A, double phi, double omega){ return -A*(omega*omega)* std::sin(omega*t - phi); }
	static double identity(double x){ return x; }
	static double testFn_3D(double x, double y, double z) { return x * x + y* y + z*z; }
	static double testFn_3D_2(double x, double y, double z) { return x + y + z; }
	static double testFn_3D_3(double x, double y, double z) { return x * y * z; }

	static double FN__N_2x_y(matrix<double> N)
	{
		return 2*N(1,1) + N(2,1); 
	}

	static double FN__N_100x_y(matrix<double> N)
	{
		return 100* N(1, 1) + N(2, 1);
	}
	static double FN__N_2x2_y2(matrix<double> N)
	{
		return 2 * N(1, 1) *N(1,1) + N(2, 1)*N(2,1);
	}

	static double FN__N_x3_y3(matrix<double> N)
	{
		return  N(1, 1)*N(1, 1)*N(1, 1) + N(2, 1)*N(2, 1)*N(2, 1);
	}

	static double FN__N_x3_y3_xy(matrix<double> N)
	{
		return  N(1, 1)*N(1, 1)*N(1, 1) + N(2, 1)*N(2, 1)*N(2, 1) + N(1,1)*N(2,1);
	}

	static double FN__N_xy2_x4_y4_x2y(matrix<double> N)
	{
		return  N(1, 1)*N(2, 1)*N(2, 1) + N(1, 1)*N(1, 1)*N(1, 1)*N(1, 1) + N(2, 1)*N(2, 1)*N(2, 1)*N(2,1)+ N(1, 1)*N(1, 1)*N(2, 1);
	}

	static double FN__N_x3_y3_x2y(matrix<double> N)
	{
		return  N(1, 1)*N(1, 1)*N(1, 1) + N(2, 1)*N(2, 1)*N(2, 1) + N(1, 1)*N(1, 1)*N(2, 1);
	}

	static double FN__N_cos_y(matrix<double> N)
	{
		return std::cos(N(2,1));
	}

	static double FN__N_cos_xy(matrix<double> N)
	{
		return std::cos(N(2, 1)*N(1,1));
	}



	// N is a Column Vector . 
	static double testFn_5_N(matrix<double> N)
	{
		return (N(1, 1)) * N(2, 1) * (N(3, 1)) + N(4, 1) + N(5, 1) * 3 - N(1, 1);
	}

	static double testFn_3_N(matrix<double> N)
	{
		return testFn_3D(N(1, 1), N(2, 1), N(3, 1));
	}

	//REAL WORLD FUNCTIONS 
	static double neg_log(double x) { return -std::log(x); };
	static double relativeEntropy(double x, double y) { return x * std::log(y) - y *log(x); };
	static double sin(double x) { return x - (x * x * x) / 6; };
	static double exp(double x) { return (1 + x + (x * x / 2) ); };

	static double abs(double x) { return (x > 0) ? x : -x; }; 



	// VECTOR FUNCTIONS: 

	// A Function containing multiple functions 
	// And the inner Function has multiple parameters 


	/*
	
	IMPLEMENTATION : 
		these functions will return the std::function 
		so when you call this function you will get back a matrix of std::function , and then within the function the std::function is called 


	MESS UP : 
	Initially I passed in a matDouble to the function which returns the matrix of std::function 
	But that matDouble made no difference as it is not used. 
	The matDouble that is used to evalutate the inner inner inner function is passed in by the inner caller ? 
	Just use it 

	static vectorFn_N FN_2_2__xy__x2y(matDouble N)
	{											^^ No need for this 
	vectorFn_N fn(2, 1);
	fn(1, 1) = [](matDouble N){ return N(1, 1) * N(2, 1); };
							^^ this will do the job ! 
	fn(2, 1) = [](matDouble N){ return N(1, 1) * N(2, 1) * N(1, 1); };
	return fn;
	}
	
	*/

	/*
		(x*y , x*x*y ) 2 function each with 2 parameters
	*/
	static vectorFn_N get_FN_2_2__xy__x2y() 
	{
		vectorFn_N fn(2, 1);
		fn(1, 1) = [](matDouble N){ return N(1, 1) * N(2, 1); };

		fn(2, 1) = [](matDouble N){ return N(1, 1) * N(2, 1) * N(1, 1); };
		return fn; 
	}

	/*
	(x*y , x*x*y , x*x*x) 3 function each with 2 parameters
	*/
	static vectorFn_N get_FN_3_2__xy__x2y__x3() // (x*y , x*x*y ) 
	{
		vectorFn_N fn(3, 1);
		fn(1, 1) = [](matDouble N){ return N(1, 1) * N(2, 1); };

		fn(2, 1) = [](matDouble N){ return N(1, 1) * N(2, 1) * N(1, 1); };

		fn(3, 1) = [](matDouble N){ return N(1, 1) * N(1, 1) * N(1, 1); };
		
		return fn;
	}

	/*
	(x*y*z , x*x*y*z  , x*x*x*z ) 3 function each with 3 parameters
	*/
	static vectorFn_N get_FN_3_3__xyz__x2yz__x3z()
	{
		vectorFn_N fn(3, 1);
		fn(1, 1) = [](matDouble N){ return N(1, 1) * N(2, 1) * N(3,1); };

		fn(2, 1) = [](matDouble N){ return N(1, 1) * N(2, 1) * N(1, 1)* N(3, 1); };

		fn(3, 1) = [](matDouble N){ return N(1, 1) * N(1, 1) * N(1, 1)* N(3, 1); };
		return fn;
	}


	/*
	(x*y*z*9 + x*y*z ,x*x*y*z*7 + x*y*z,x*x*x*z*7 + x*y*z 3 function each with 3 parameters
	*/
	static vectorFn_N get_FN_3_3__xyz_9_xyz__x2yz_7_xyz__x3z_7_xyz()
	{
		vectorFn_N fn(3, 1);
		fn(1, 1) = [](matDouble N){ return N(1, 1) * N(2, 1) * N(3, 1) * 9 + N(1, 1)*N(3,1)*N(3,1); };

		fn(2, 1) = [](matDouble N){ return N(1, 1) * N(2, 1) * N(1, 1)* N(3, 1) * 7 + N(1, 1)*N(2, 1)*N(3, 1); };

		fn(3, 1) = [](matDouble N){ return N(1, 1) * N(1, 1) * N(1, 1)* N(3, 1) * 7 + N(1,1)*N(2,1)*N(3,1); };
		return fn;
	}





};

