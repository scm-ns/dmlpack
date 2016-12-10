#pragma once

//System
#include <cmath> // For defining functions
#include <vector> // rootNewtonMultiple
#include <string>  // Error Handling 
#include <functional>
#include <initializer_list>

//Personal 
#include "matrix.h"
#include "la_pack.h"
#include "func.h" // For shorter math function declerations 


/*
	Pass in a function to the operation . 
	REASON
	Pass in function is redundant. Copy and Paste. 

	Begin : April 9th 
	April 24 th ; -> Not much Progress. Bulid is going very slow 



	Implementation Details
	Loops will have to use doubles as small values have to be added 

*/




class analysis
{
public:
	analysis(void);
	~analysis(void);
	double getStep()const { return _stepSize; }
	void setStep(double aStep){ _stepSize = aStep; }




	//--------------------------------------------INTEGRATION -----------------------------------------------------------------//
	double intergalSimple(std::function<double(double)> fn, double begin , double end );


	// Take Integral by splitting region into multiple Regions. High Accuracy , more time required 
	double integralTrapazoidalComposite(std::function<double(double)> fn, double begin, double end, double  precisionReduction);
	// Take Integral over a single region . Faster , lower Accuracy 
	double integralTrapazoidal(std::function<double(double)> fn, double begin, double end);
	double integralRomberg(std::function<double(double)> fn, double begin, double end);
	// Take Integral by splitting region into multiple Regions. High Accuracy , more time required 
	double integralSimpsonComposite(std::function<double(double)> fn, double begin, double end, double precisionReduction = 1000);
	double integralSimpson(std::function<double(double)> fn, double begin, double end, double precisionReduction = 1000);

	double integralSimpson3_8(std::function<double(double)> fn, double begin, double end);
	double integralSimpson3_8Composite(std::function<double(double)> fn, double begin, double end, double  precisionReduction = 1000);

	double fixedPointBrouwer(std::function<double(double)> fn, double begin = -100000, double end = 100000);



//--------------------------------------SIMPLE - DIFFERENTIATION----------------------------------------------------//


	// HIGHER ACCURACY -> H 8 order or 6 order 
	double derivative_1_H(std::function<double(double)> fn, double x);
	double derivative_2_H(std::function<double(double)> fn, double x);
	double derivative_3_H(std::function<double(double)> fn, double x);
	double derivative_4_H(std::function<double(double)> fn, double x);
	double derivative_5_H(std::function<double(double)> fn, double x);  // Only 2nd Order

	// FIVE POINT STENSIL -> S  
	double derivative_1_S(std::function<double(double)> fn, double x);
	double derivative_2_S(std::function<double(double)> fn, double x);
	double derivative_3_S(std::function<double(double)> fn, double x);
	double derivative_4_S(std::function<double(double)> fn, double x);



	double derivativeCenter(std::function<double(double)> fn, double point);
	double derivativeBacward(std::function<double(double)> fn, double x);
	double derivativeForward(std::function<double(double)> fn, double x);
	double derivativeAverage(std::function<double(double)> fn, double x);
	double derivativeNash(std::function<double(double)> fn, double x);
	double secondDerivative(std::function<double(double)> fn, double x);
	double derivative_1_Richardson_4O(std::function<double(double)> fn, double x, int n1 = 10000, int n2 = 20000);
	double derivative_1_Richardson_6O(std::function<double(double)> fn, double x, int n1 = 10000, int n2 = 20000);
	double thirdDerivative_unsafe(std::function<double(double)> fn, double x);
	double fourthDerivative_unsafe(std::function<double(double)> fn, double x);
	double fifthDerivative_unsafe(std::function<double(double)> fn, double x);


	//---------------------------------------------HIGER DIFFERENTIATION ------------------------------------------------//
	
	// STENSIL -> S ;  
	double laplacian_2D_S(std::function<double(double, double)> fn, double x, double y);



	//-------------------------------------------------ODE------------------------------------------------------------///
	//Solve ODE of first Order , y' = f(x,y) . input y' as the function 

	double analysis::eulerODE_1(std::function<double(double, double)> fn, double requiredPoint, double initalValue, double initialPoint);
	double analysis::heunODE_1(std::function<double(double, double)> fn, double requiredPoint, double initalValue, double initialPoint);


	// OutPut will be a matrix which contains values of x from -value to value as index and corresponding y in first column 
	matrix<double> eulerODE_1_range(std::function<double(double, double)> fn, double initalValue, double initialPoint ,int value = 1000,int step = 10);
	matrix<double> huenODE_1_range(std::function<double(double, double)> fn, double initalValue, double initialPoint, int value = 1000, int step = 10);

	double rungekuttaODE(std::function<double(double, double)> fn, double requiredPoint, double initalValue, double initialPoint);
	double dormandPrinceODE(std::function<double(double, double)> fn, double finalPoint, double initalValue, double initialPoint);
	double dormandPrinceODE_adapt(std::function<double(double, double)> fn, double finalPoint, double initalValue, double initialPoint);
	double lagrangeInterpolationPoly(std::function<double(double)> fn, double requiredPoint, int degree, double initialX, double finalX);







	//-------------------------------------------------------ROOT FINDING-----------------------------------------------------------------//
	/*
		Returns a column vector with 2 elements , first Root and Second Root 
	*/
	matrix<double>  rootQuadratic( double a, double b, double c);
	double rootBisection(std::function<double(double)> fn, double bracketOpen = -100000, double bracketEnd = 100000);
	double rootNewtonSingle(std::function<double(double)> fn, double begin = -100000, double end = 100000);
	double rootNewtonMultiplicity(std::function<double(double)> fn, double begin = -100000, double end = 100000);

	double rootSecant(std::function<double(double)> fn); // DOES NOT WORK

	double rootRegula_Falsi(scalarFn_1 fn, double begin = -100000, double end = 100000); // DOES NOT WORK 
	
	// Second Order HouseHolder Root Finding Algorithm 
	double rootHalley(scalarFn_1 fn, double begin = -100000, double end = 100000);


	// Iteration based on second degree eqn for sol
	/*
		Ignore h^3 terms not h^2 terms ; 
		3 order method 
	*/
	double rootChebyshev(scalarFn_1 fn, double guess = 0.1);
	/*
	 Requires 3 approximations in random 
	*/
	double rootMuller(scalarFn_1 fn, double guess_1 = 0, double guess_2 = 1, double guess_3 = 2); 


	//-------------------------------------------------------ROOT FINDING-----------------------------------------------------------------//



	//-------------------------------------------------------GENERAL-----------------------------------------------------------------------//
	double kroneckerDelta(double i, double j){ return (i == j) ? 1 : 0; };





	//--------------------------------------------------SPEICAL INTEGRATIONS------------------------------------------------------//

	double convolution(std::function<double(double)> fn, std::function<double(double)> fn2, double finalT);



	//----------------------------------------------------BEST----------------------------------------------------------------------------------//

	double differentiateBest(std::function<double(double)> fn, double x);
	double integralBest(std::function<double(double)> fn, double begin, double end);

	//----------------------------------------------------MULTIVARIABLE --------------------------------------------------------------------------//


	// Initial Guess Zero Vector 
	matrix<double> gradient_2D(scalarFn_2 fn, double x, double y);
	matrix<double> gradient_3D(scalarFn_3 fn, double x, double y , double z);
	
	// GeneralizedN Dimensional Gradient 


	
	matrix<double> gradient_ND(scalarFn_N  fn, matrix<double> N);
	


	// Hessian 

	matrix<double> analysis::hessian_ND(scalarFn_N fn, matDouble N);

	matrix<double> jacobian_3D(vectorFn_3 fn, double x, double y, double z);
	matrix<double> jacobian_ND(vectorFn_N fn, matDouble N);

	/*
	
	//matrix<double> hessian_3D(scalarFn_3 fn, double x, double y, double z);   // DOES NOT WORK

	
	
	*/



	// curl // Divergence // laplacian 

	/*
	General Representation of a vector function with parameter t ; 
	*/

	


	/*
	Representation of a Vector Function 
	The Vector Function represented as a column vector with 1st dimenasion is 1st position , 2nd dimension is 2nd position
	*/


	vectorFn_1 analysis::get_func_2D_paramT(scalarFn_1 fn1, scalarFn_1 fn2);
	vectorFn_1 analysis::get_func_3D_paramT(scalarFn_1 fn1, scalarFn_1 fn2, scalarFn_1 fn3);
	

	vectorFn_3 analysis::get_func_3D_param3(scalarFn_3 fn1, scalarFn_3 fn2, scalarFn_3 fn3);

	vectorFn_N analysis::get_func_3D_paramN(scalarFn_N fn1, scalarFn_N fn2, scalarFn_N fn3);
	// Curl 2D ; 

	vectorFn_N analysis::get_func_ND_paramN(scalarFn_N fn1, scalarFn_N fn2, scalarFn_N fn3);

	// Divergence 3D
	// Takes in a vector of functions , differentiates each component , at given a point , then returns a scalr value. 
	double analysis::divergence_3D(vectorFn_N vecFn , matrix<double> N );

	matrix<double> analysis::curl_3D(vectorFn_N vecFn, matrix<double> N);


	template<typename ... Ts>
	double divergence_ND(matrix<double> N, scalarFn_N first, Ts ... rest)
	{
		int current = 1;  /// This is used to tell which dimension to point to . 
		return partial_deriv_Center_multiDim(first, N, current) + divergence_ND_Wrapper(N, current, rest...);
	}


//--------------------------------------POLYNOMIAL APPROXIMATION --------------------------------------------------------------------------------------//

	/*
		NOTES:
			Evaluate a polynomial using few steps.
			Horner's method or synthetic division. Powers are not taken explicitly ..
		INPUT :
			x -> point at which poly is evaluated 
			coeffMatrix -> matrix of coefficients stored as a column matrix
			element 1 of a matrix is a0
			element 2 of a matrix is a1 
			

		OUTPUT:
			value -> poly evaluated at x 
	*/
	double polynomEval(double x, matDouble coeffMatrix)
	{
		int n = coeffMatrix.numRows();
		double value = 0; 
		value = coeffMatrix(n, 1);
		for (int k = n - 1; k > 0; k++)
		{
			value = x * value + coeffMatrix(k, 1);
		}
		return value;
	}




private:
	double _stepSize = 1e-5;
	double _tolerance = 1e-7;
	double _iterations = 1e7; 
	bool isInfinitesimal(double x){ return std::abs(x) < _tolerance; }
	static void error(const char* p){ std::string str = "analysis -> Error: "; std::cout << str << p << std::endl; }
	bool relativeChangeSmall (double x_i_1, double x_i){ return ((x_i_1 - x_i) / x_i < _tolerance) ? true : false; }
	bool stoppingCriterion(double x_i_1, double x_i){ return relativeChangeSmall(x_i_1, x_i) && isInfinitesimal(x_i_1); }
	la_pack la; 

	// For Gradient_ND

	double partial_deriv_Center_multiDim(std::function<double(matrix<double>)>  fn, matrix<double> N, int currIndex);
	
	// For Hessian_ND
	double mixed2_partial_deriv_Center_multiDim(std::function<double(matrix<double>)>  fn, matrix<double> N, int index1, int index2);


	// Fn is just an object which returns a value , for a value that you input.
	// Will create it later . 

	template<typename ... Ts>
	double divergence_ND_Wrapper(matrix<double> N, int& current, scalarFn_N first, Ts ... rest)
	{
		++current;
		return partial_deriv_Center_multiDim(first, N, current) + divergence_ND_Wrapper(N, current, rest...);
	}

	double divergence_ND_Wrapper(matrix<double> N, int& current, scalarFn_N first)
	{
		++current;
		return partial_deriv_Center_multiDim(first, N, current);
	}
	

	//Take Function input from user using lamba experssions.

};


// TEMPLATE FUNCTIONS




/*


DEPRECATED



matrix<double>  rootNewtonMultiple(std::function<double(double)> fn, double begin = -100000, double end = 100000); // Does not Work

double rootSecant(std::function<double(double)> fn, double begin = -100000, double end = 100000); // DOES NOT WORK








*/