#include "analysis.h"

using namespace std; 
analysis::analysis(void)
{		}


analysis::~analysis(void)
{		}



//----------------------------------------------SINGLE DIMENSION - DIFFERENTIATION-------------------------------------------//
double analysis::derivativeAverage(std::function<double(double)> fn, double x)
{
	double forward = derivativeForward(fn, x);
	double backward = derivativeBacward(fn, x);
	double center = derivativeCenter(fn, x);
	return (4 * center + 3 * forward + 3 * backward) / 10;
}


double analysis::derivativeCenter(std::function<double(double)> fn, double x)
{
	return (fn(x + _stepSize) - fn(x - _stepSize)) / (2 * _stepSize);
}


double analysis::derivativeBacward(std::function<double(double)> fn, double x)
{
	return (fn(x ) - fn(x - _stepSize)) / (_stepSize);
}


double analysis::derivativeForward(std::function<double(double)> fn, double x)
{
	return ( fn(x + _stepSize) - fn(x ) ) / (_stepSize);
}

double analysis::derivativeNash(std::function<double(double)> fn, double x)
{
	
	double epsMachine = std::numeric_limits<double>::epsilon();

	double stepSize = std::sqrt(epsMachine) *(std::abs(x) + std::sqrt(epsMachine));

	return (fn(x + stepSize) - fn(x - stepSize)) / (2 * stepSize);
}

double analysis::secondDerivative(std::function<double(double)> fn, double x)
{
	return ((fn(x + _stepSize) - 2 * fn(x) + fn(x - _stepSize)) / (_stepSize*_stepSize));
}


double analysis::derivative_1_H(std::function<double(double)> fn, double x)
{
	double tableau[9] = { 1 / 280 , -4/105 , 1/5 , -4/5 , 0 , 4/5 , -1/5 , 4/105 , -1/280};
	double numer1 = tableau[0] * fn(x - 4 * _stepSize) + tableau[1] * fn(x - 3 * _stepSize) + tableau[2] * fn(x - 2 * _stepSize) + tableau[3] * fn(x - _stepSize);
	double numer2 = tableau[8] * fn(x + 4 * _stepSize) + tableau[7] * fn(x + 3 * _stepSize) + tableau[6] * fn(x + 2 * _stepSize) + tableau[5] * fn(x - _stepSize);
	return (numer1 + numer2) / (_stepSize);
}

double analysis::derivative_2_H(std::function<double(double)> fn, double x)
{
	double tableau[9] = { -1 / 560, 8 / 315, -1 / 5, 8 / 5, -205 / 72, 8 / 5, -1 / 5, 8 / 315, -1 / 560 };
	double numer1 = tableau[0] * fn(x - 4 * _stepSize) + tableau[1] * fn(x - 3 * _stepSize) + tableau[2] * fn(x - 2 * _stepSize) + tableau[3] * fn(x - _stepSize) + tableau[4] * fn(x);
	double numer2 = tableau[8] * fn(x + 4 * _stepSize) + tableau[7] * fn(x + 3 * _stepSize) + tableau[6] * fn(x + 2 * _stepSize) + tableau[5] * fn(x - _stepSize);
	return (numer1 + numer2) / (_stepSize)*(_stepSize);
}

double analysis::derivative_3_H(std::function<double(double)> fn, double x)
{
	double tableau[9] = { -7 / 240, 3 / 10, -169 / 120, 61 / 30, 0, -61 / 30, 169 / 120, -3 / 10, -7 / 240 };
	double numer1 = tableau[0] * fn(x - 4 * _stepSize) + tableau[1] * fn(x - 3 * _stepSize) + tableau[2] * fn(x - 2 * _stepSize) + tableau[3] * fn(x - _stepSize) + tableau[4] * fn(x);
	double numer2 = tableau[8] * fn(x + 4 * _stepSize) + tableau[7] * fn(x + 3 * _stepSize) + tableau[6] * fn(x + 2 * _stepSize) + tableau[5] * fn(x - _stepSize);
	return (numer1 + numer2) / (_stepSize)*(_stepSize)*(_stepSize);
}

double analysis::derivative_4_H(std::function<double(double)> fn, double x)
{
	double tableau[9] = { 7 / 240, -2 / 5, 169 / 60 , -122 / 15, 91/8 , -122 / 15, 169 / 60, -2 / 5, 7 / 240 };
	double numer1 = tableau[0] * fn(x - 4 * _stepSize) + tableau[1] * fn(x - 3 * _stepSize) + tableau[2] * fn(x - 2 * _stepSize) + tableau[3] * fn(x - _stepSize) + tableau[4] * fn(x);
	double numer2 = tableau[8] * fn(x + 4 * _stepSize) + tableau[7] * fn(x + 3 * _stepSize) + tableau[6] * fn(x + 2 * _stepSize) + tableau[5] * fn(x - _stepSize);
	return (numer1 + numer2) / (_stepSize)*(_stepSize)*(_stepSize)*(_stepSize);
}

double analysis::derivative_5_H(std::function<double(double)> fn, double x)
{
	double tableau[9] = { 0 , -1/2 , 2 , -5/2 , 0 , 5/2 , -2 , 1/2 , 0 };
	double numer1 = tableau[0] * fn(x - 4 * _stepSize) + tableau[1] * fn(x - 3 * _stepSize) + tableau[2] * fn(x - 2 * _stepSize) + tableau[3] * fn(x - _stepSize) + tableau[4] * fn(x);
	double numer2 = tableau[8] * fn(x + 4 * _stepSize) + tableau[7] * fn(x + 3 * _stepSize) + tableau[6] * fn(x + 2 * _stepSize) + tableau[5] * fn(x - _stepSize);
	return (numer1 + numer2) / (_stepSize)*(_stepSize)*(_stepSize)*(_stepSize)*(_stepSize);
}

double analysis::derivative_1_S(std::function<double(double)> fn, double x)
{
	double numer = (-1 * fn(x + 2 * _stepSize) + 8 * fn(x + _stepSize) - 8 * fn(x - _stepSize) + fn(x - 2 * _stepSize));
	return numer / (12 * _stepSize);
}
double analysis::derivative_2_S(std::function<double(double)> fn, double x)
{
	double numer = (-1 * fn(x + 2 * _stepSize) + 16 * fn(x + _stepSize) - 30 *fn(x) + 16* fn(x - _stepSize) - fn(x - 2 * _stepSize));
	return numer / (12 * _stepSize * _stepSize);
}
double analysis::derivative_3_S(std::function<double(double)> fn, double x)
{
	double numer = ( fn(x + 2 * _stepSize) - 2 * fn(x + _stepSize) + 2 * fn(x - _stepSize) - fn(x - 2 * _stepSize) );
	return numer / (2 * _stepSize * _stepSize * _stepSize);
}
double analysis::derivative_4_S(std::function<double(double)> fn, double x)
{
	double numer = (fn(x + 2 * _stepSize) - 4 * fn(x + _stepSize) + 6 * fn(x) - 4 * fn(x - _stepSize) + fn(x - 2 * _stepSize));
	return numer / (_stepSize * _stepSize * _stepSize * _stepSize);
}


double analysis::laplacian_2D_S(std::function<double(double, double)> fn, double x, double y)
{
	double numer = fn(x + _stepSize, y) + fn(x - _stepSize, y) + fn(x, y + _stepSize) + fn(x, y - _stepSize) - 4 * fn(x, y);

	return numer / ( _stepSize * _stepSize ) ;
}

double analysis::derivative_1_Richardson_4O(std::function<double(double)> fn, double x, int n1, int n2) // UNTETSTED
{
	double h1 = 1 / n1; double h2 = 1 / n2; 
	cout << h1 << h2 << endl;

	double D1 = (fn(x + h1) - fn(x - h1)) / 2 * h1;
	double D2 = (fn(x + h2) - fn(x - h2)) / 2 * h2;

	return (4 * D1 - D2) / 3; 
}


double analysis::derivative_1_Richardson_6O(std::function<double(double)> fn, double x, int n1, int n2)// UNTETSTED
{
	double h1 = 1 / n1; double h2 = 1 / n2;
	cout << h1 << h2 << endl;

	double S1 = (-fn(x + 2 * h1) + 8 * fn(x + h1) - 8 * fn(x - h1) + fn(x - 2 * h1)) / (12 * h1); 
	double S2 = (-fn(x + 2 * h2) + 8 * fn(x + h2) - 8 * fn(x - h2) + fn(x - 2 * h2)) / (12 * h2);

	return ((16 * S1) - S2) / 15; 
}


double analysis::thirdDerivative_unsafe(std::function<double(double)> fn, double x)
{
	double numer = (fn(x + 3 * _stepSize) - 3 * fn(x + _stepSize) + 3 * fn(x - _stepSize) - fn(x - 3 * _stepSize));
	return (numer) / (_stepSize*_stepSize*_stepSize);
}

double analysis::fourthDerivative_unsafe(std::function<double(double)> fn, double x)
{
	double numer = ( fn(x + 4 * _stepSize) - 4 * fn(x + 2 * _stepSize) + 6 * fn(x) - 4 * fn(x - 2 * _stepSize) - fn(x - 4 * _stepSize));
	return (numer) / (_stepSize*_stepSize*_stepSize*_stepSize);
}

double analysis::fifthDerivative_unsafe(std::function<double(double)> fn, double x)
{
	double numer = (fn(x + 5 * _stepSize) - 5 * fn(x + 3 * _stepSize) + 10 * fn(x + _stepSize) - 10 * fn(x - _stepSize) - 5 * fn(x - 3 * _stepSize) - fn(x - 5 * _stepSize));
	return (numer) / (_stepSize*_stepSize*_stepSize*_stepSize*_stepSize);
}


double analysis::fixedPointBrouwer(std::function<double(double)> fn, double begin, double end) // UNTESTED 
{
	for (int i = begin; i < end; i += 0.001)
	{
		if (fn(i) - i < _tolerance)
		{
			return i; // i is fixed point 
		}
	}



}


//-----------------------------------------------ODE------------------------------------------------------------------------------------//




double analysis::dormandPrinceODE(std::function<double(double, double)> fn, double finalPoint, double initalValue, double initialPoint)
{
	double errorABS = _tolerance;
	double stepSize = _stepSize;
	//double xn = initialPoint;
	double yn = initalValue;
	double K1, K2, K3, K4, K5, K6, K7;
	double fifthOrder, fourthOrder, adaptive;
	for (double xn = initialPoint; xn < finalPoint; xn += stepSize) // Continue till we reach end 
	{
		K1 = fn(xn, yn);
		K2 = fn(xn + (1.0 / 5)* stepSize, yn + (1.0 / 5) * K1 * stepSize);
		K3 = fn(xn + (3.0 / 10)* stepSize, yn + (3.0 / 40) * K1*stepSize + (9.0 / 40) * K2 *stepSize);
		K4 = fn(xn + (4.0 / 5)* stepSize, yn + (44.0 / 45) * K1*stepSize + (-56.0 / 15) * K2 *stepSize + (32.0 / 19) * K3*stepSize);
		K5 = fn(xn + (8.0 / 9)* stepSize, yn + (19372.0 / 6561) * K1*stepSize + (-25360.0 / 2187) * K2 *stepSize + (64448.0 / 6561) * K3*stepSize + (-212.0 / 729) * K4*stepSize);
		K6 = fn(xn + (1)* stepSize, yn + (9017.0 / 3168) * K1*stepSize + (-355.0 / 33) * K2 *stepSize + (46732.0 / 5247) * K3 *stepSize + (49.0 / 176) * K4*stepSize + (-5103.0 / 18656) * K5*stepSize);
		K7 = fn(xn + (1)* stepSize, yn + (35.0 / 384) * K1*stepSize + (0) * K2 *stepSize + (500.0 / 1113) * K3* stepSize + (125.0 / 192) * K4*stepSize + (-2187.0 / 6784) * K5*stepSize + (11.0 / 84) * K6 * stepSize);
		
		fifthOrder = stepSize * ((35.0 / 384)*K1 + (0)*K2 + (500.0 / 1113)*K3 + (125.0 / 192)* K4 + (-2187.0 / 6784) * K5 + (11.0 / 84) * K6);
		fourthOrder = stepSize*((5179.0 / 57600) * K1 + (0) * K2 + (7571.0 / 16695) * K3 + (383.0 / 640) * K4 + (-92097.0 / 339200) * K5 + (187.0 / 2100) * K6 + (1.0 / 40) * K7);
		//std::cout << fourthOrder << endl; 

		adaptive = stepSize / (2 * (finalPoint - initialPoint)* std::abs(fifthOrder - fourthOrder));
		adaptive = std::pow(adaptive, (1 / 4.0));


		yn = yn + (fifthOrder); 
		//std::cout << yn << endl; 
	}
	return yn;
}


double analysis::dormandPrinceODE_adapt(std::function<double(double, double)> fn, double finalPoint, double initalValue, double initialPoint)
{
	double errorABS = _tolerance;
	double stepSize = _stepSize;
	//double xn = initialPoint;
	double yn = initalValue;
	double K1, K2, K3, K4, K5, K6, K7;
	double fifthOrder, fourthOrder, adaptive;
	for (double xn = initialPoint; xn < finalPoint; xn += stepSize) // Continue till we reach end 
	{
		K1 = fn(xn, yn);
		K2 = fn(xn + (1.0 / 5)* stepSize, yn + (1.0 / 5) * K1 * stepSize);
		K3 = fn(xn + (3.0 / 10)* stepSize, yn + (3.0 / 40) * K1*stepSize + (9.0 / 40) * K2 *stepSize);
		K4 = fn(xn + (4.0 / 5)* stepSize, yn + (44.0 / 45) * K1*stepSize + (-56.0 / 15) * K2 *stepSize + (32.0 / 19) * K3*stepSize);
		K5 = fn(xn + (8.0 / 9)* stepSize, yn + (19372.0 / 6561) * K1*stepSize + (-25360.0 / 2187) * K2 *stepSize + (64448.0 / 6561) * K3*stepSize + (-212.0 / 729) * K4*stepSize);
		K6 = fn(xn + (1)* stepSize, yn + (9017.0 / 3168) * K1*stepSize + (-355.0 / 33) * K2 *stepSize + (46732.0 / 5247) * K3 *stepSize + (49.0 / 176) * K4*stepSize + (-5103.0 / 18656) * K5*stepSize);
		K7 = fn(xn + (1)* stepSize, yn + (35.0 / 384) * K1*stepSize + (0) * K2 *stepSize + (500.0 / 1113) * K3* stepSize + (125.0 / 192) * K4*stepSize + (-2187.0 / 6784) * K5*stepSize + (11.0 / 84) * K6 * stepSize);

		fifthOrder = stepSize * ((35.0 / 384)*K1 + (0)*K2 + (500.0 / 1113)*K3 + (125.0 / 192)* K4 + (-2187.0 / 6784) * K5 + (11.0 / 84) * K6);
		fourthOrder = stepSize*((5179.0 / 57600) * K1 + (0) * K2 + (7571.0 / 16695) * K3 + (383.0 / 640) * K4 + (-92097.0 / 339200) * K5 + (187.0 / 2100) * K6 + (1.0 / 40) * K7);
		//std::cout << fourthOrder << endl; 

		adaptive = stepSize*errorABS / (2 * (finalPoint - initialPoint)* std::abs(fifthOrder - fourthOrder));
		adaptive = std::pow(adaptive, (1 / 4.0));

		//cout << adaptive << endl;
		//	cout << "S :"<<stepSize << endl;
		if (adaptive > 2 * errorABS)
		{
			stepSize *= 2;
		}
		else if (adaptive >= 1 * errorABS  && adaptive <= 2 * errorABS)
		{
			stepSize = stepSize;
		}
		else if (adaptive < 1 * errorABS)
		{
			stepSize /= 2;
		}

		// Update Values of x and y 
		//xn += stepSize; // DONE IN LOOP 
		yn = yn + (fifthOrder);
		//std::cout << yn << endl; 
	}
	return yn;
}

double analysis::eulerODE_1(std::function<double(double, double)> fn,double requiredPoint, double initalValue, double initialPoint)
{
	
	double x_i = initialPoint;
	double y_i = initalValue;
	double y_i_1 ;//= y_i + fn(x_i, y_i)*_stepSize;
	for (double i = initialPoint; i < requiredPoint; i += _stepSize)
	{
		y_i_1 = y_i + fn(i, y_i)*_stepSize;
		y_i = y_i_1;
		//std::cout << y_i << std::endl ;				//DEBUG
	}
	return y_i;
}

double analysis::heunODE_1(std::function<double(double, double)> fn, double requiredPoint, double initalValue, double initialPoint)
{
	double x_i = initialPoint;
	double y_i = initalValue;
	double y_i_1;//= y_i + fn(x_i, y_i)*_stepSize;
	double K1, K2; 
	for (double i = initialPoint; i < requiredPoint; i += _stepSize)
	{
		K1 = fn(i, y_i);
		K2 = fn(i + _stepSize, y_i + K1*_stepSize);

		y_i_1 = y_i + ( (K1 + K2) / 2)*_stepSize;
		y_i = y_i_1;
		//std::cout << y_i << std::endl ;				//DEBUG
	}
	return y_i;


}


double analysis::rungekuttaODE(std::function<double(double, double)> fn, double requiredPoint, double initalValue, double initialPoint)
{
	double x_i = initialPoint;
	double y_i = initalValue;
	double y_i_1;
	double K1, K2, K3, K4;
	for (double i = initialPoint; i < requiredPoint; i += _stepSize)
	{
		K1 = _stepSize * fn(i, y_i);
		K2 = _stepSize * fn(i + _stepSize / 2, y_i + K1 / 2);
		K3 = _stepSize * fn(i + _stepSize / 2, y_i + K2 / 2);
		K4 = _stepSize * fn(i + _stepSize, y_i + K3);

		y_i_1 = y_i + K1 / 6 + K2 / 3 + K3 / 3 + K4 / 6;
		y_i = y_i_1;
		//std::cout << y_i << std::endl ;				//DEBUG
	}
	return y_i;
}

matrix<double> analysis::eulerODE_1_range(std::function<double(double, double)> fn, double initalValue, double initialPoint, int value, int step)
{
	if (step == 0)
	{
		step = 1/_stepSize; // // if step is not specified by user , use global steps ;
	}

	matrix<double> R(2 * value * step, 2); // FOR EACH VALUE , WE TAKE _stepSize steps :: Values go from - values to + values 
	// Size of matrix will be 2 * values  * _stepSize , the inital Point will be at values  * _stepSize . final point at 2* values  * _stepSize and backward most point at 0  * _stepSize
	

	// Store inital X Value
	R(value * step, 1) = initialPoint;
	R(value * step, 2) = initalValue;
	

	// Filling array from initialPoint to value 
	for (long long i = 0; i < value*step; ++i) // Index 
	{
		// the i = 0 index is mapped to value*step 
		// FROM there we go forward
		// i is used for indexing into array 

		R(value * step + i + 1, 2) = R(value * step + i, 2) + fn(R(value * step + i, 1), R(value * step + i, 2))*_stepSize; // Update the Y values
		R(value * step + i + 1, 1) = R(value * step + i , 1) + _stepSize; // Updates the X values, X is incremented by _stepSize
		
	}

	// Filling array from 0 to initalPoint 
	for (long long i = value * step; i > 1 ; i--) // Index // Do not index at 0 // ASSERTION ERROR 
	{
		// the i = value*step index is mapped to value*step 
		// From there we go back 
		// i is used for indexing into array 
		// WE are going backwards from the inital point 
		R(i - 1, 2) = R(i, 2) - fn(R(i, 1), R(i, 2)) * _stepSize; // Update the Y values
		R(i - 1, 1) = R(i, 1) - _stepSize; // Updates the X values, X is incremented by _stepSize
	}
	return R; 
}



matrix<double> analysis::huenODE_1_range(std::function<double(double, double)> fn, double initalValue, double initialPoint, int value , int step )
{
	if (step == 0)
	{
		step = 1 / _stepSize; // // if step is not specified by user , use global steps ;
	}

	matrix<double> R(2 * value * step, 2); // FOR EACH VALUE , WE TAKE _stepSize steps :: Values go from - values to + values 
	// Size of matrix will be 2 * values  * _stepSize , the inital Point will be at values  * _stepSize . final point at 2* values  * _stepSize and backward most point at 0  * _stepSize


	// Store inital X Value
	R(value * step, 1) = initialPoint;
	R(value * step, 2) = initalValue;

	double K1, K2; 
	// Filling array from initialPoint to value 
	for (long long i = 0; i < value*step; ++i) // Index 
	{
		// the i = 0 index is mapped to value*step 
		// FROM there we go forward
		// i is used for indexing into array 


		K1 = fn(R(value * step + i, 1), R(value * step + i, 2));
		K2 = fn(R(value * step + i, 1) + _stepSize, R(value * step + i, 2) + K1*_stepSize);
		R(value * step + i + 1, 2) = R(value * step + i, 2) + ((K1 + K2) / 2)*_stepSize;


		R(value * step + i + 1, 1) = R(value * step + i, 1) + _stepSize; // Updates the X values, X is incremented by _stepSize

	}

	// Filling array from 0 to initalPoint 
	for (long long i = value * step; i > 1; i--) // Index // Do not index at 0 // ASSERTION ERROR 
	{
		// the i = value*step index is mapped to value*step 
		// From there we go back 
		// i is used for indexing into array 
		// WE are going backwards from the inital point 


		K1 = fn(R(i, 1), R(i, 2));
		K2 = fn(R(value * step + i, 1) + _stepSize, R(value * step + i, 2) + K1*_stepSize);

		R(value * step + i + 1, 2) = R(value * step + i, 2) + ((K1 + K2) / 2)*_stepSize;


		R(i - 1, 1) = R(i, 1) - _stepSize; // Updates the X values, X is incremented by _stepSize


		R(i - 1, 2) = R(i, 2) - fn(R(i, 1), R(i, 2)) * _stepSize; // Update the Y values
		R(i - 1, 1) = R(i, 1) - _stepSize; // Updates the X values, X is incremented by _stepSize
	}
	return R;
}






double analysis::lagrangeInterpolationPoly(std::function<double( double)> fn, double requiredPoint, int degree, double initialX, double finalX)
{
	double interval = ( finalX - initialX )/ degree; 
	matrix<double> X = la.getLinspaceCol(initialX, finalX, interval);
	double lagrange = 0;
	for (int i = 0; i < degree; ++i)
	{
		double basisPoly = 1; // CALCULATE THE BASIS LAGRANGE POLYNOMIAL 
		for (int j = 0; j < degree; ++j)
		{
			if (i != j)
			{
				basisPoly *= ((requiredPoint - X(j, 1)) / (X(i, 1) - X(j, 1)));
			}
		}
		lagrange += fn(X(i, 1)) *basisPoly;
	}
	return lagrange; 
}


//-------------------------------------------INTEGRATION ----------------------------------------------------//





double analysis::intergalSimple(std::function<double(double)> fn, double begin, double end)
{
	double integral = 0;
	for (long double i = begin; i < end; i += _stepSize)
	{
		integral += fn(i) * _stepSize;
		//std::cout << integral << std::endl; 
	}
	return integral;
}



double analysis::integralSimpsonComposite(std::function<double(double)> fn, double begin, double end, double precisionReduction)
{
	// Divide a to b into steps and use simpons to find value of each section. Then add it up 

	double integral = 0; 
	double size = _stepSize * precisionReduction;

	integral += ( fn(begin) + fn(end) );
	int regions = ( (end - begin) / size ); 

	double x = begin; 
	for (int i = 1; i < regions; ++i)
	{
		x += size; // Each time we pass a region , x move forward by the size of the region  
		if (i % 2 == 0) // EVEN 
		{
			integral += 2 * fn(x);
		}
		else
		{
			integral += 4 * fn(x);
		}
	}
	integral *= ( (end - begin) / (3 * regions) );
	return integral; 
}


double analysis::integralSimpson(std::function<double(double)> fn, double begin, double end, double precisionReduction)
{
	// Divide a to b into steps and use simpons to find value of each section. Then add it up 
	double integral = 0;
	integral = ((end - begin) / 6) * (fn(begin) + 4 * fn((begin + end) / 2) + fn(end));
	return integral;
}




double analysis::integralTrapazoidal(std::function<double(double)> fn, double begin, double end)
{
	return (end - begin)*(fn(begin) + fn(end)) / 2;
}



double analysis::integralTrapazoidalComposite(std::function<double(double)> fn, double begin, double end, double  precisionReduction)
{

	double integral = 0; 
	double size = _stepSize * precisionReduction;

	integral += (fn(begin) + fn(end));

	int regions = (end - begin) / size;

	double x = begin;
	for (int i = 1; i < regions; ++i)
	{
		x += size; // Each time we pass a region , x move forward by the size of the region  
		integral += fn(x);
	}

	integral *= (end - begin) / (2 * regions);
	return integral;

}


double analysis::integralSimpson3_8(std::function<double(double)> fn, double begin, double end)
{
	return (fn(begin) + 3 * fn((2 * begin + end) / 3) + 3 * fn((begin + 2 * end) / 3) + fn(end));
}


double  analysis::integralSimpson3_8Composite(std::function<double(double)> fn, double begin, double end, double  precisionReduction)
{

	double integral = 0;
	double size = _stepSize * precisionReduction;

	integral += ( fn(begin) + fn(end) );
	int regions = ( (end - begin) / (3 * size) );
	regions *= 3; // Make multiple of 3 ? // POINT OF ERROR . MAKE REGIONS A MULTIPLE OF 3 
	double x = begin;
	for (int i = 1; i < regions; ++i)
	{
		x += size; // Each time we pass a region , x move forward by the size of the region  
		if (i % 3 == 0) // Multiple of 3 we multiply by 2 else 3 
		{
			integral += 2 * fn(x);
		}
		else
		{
			integral += 3 * fn(x);
		}

	}
	integral *= ( (end - begin) / (3 * regions) );
	return integral;
}


//-------------------------------------------------------ROOT FINDING-----------------------------------------------------------------//

// Uses Intemediate value theorem , if the value of the function is positive and negative between two poitns respectively , 
// Then it will have 1 or odd numbers of root . 

// Bisections required for a given epsilon 
// ( b - a ) / 2 < esp ; 
// n > 1 / ln(2)  *(  ln(b - a) - ln (esp) ) 

// Never Fails ...... But VERY SLOW
double analysis::rootBisection(std::function<double(double)> fn, double bracketOpen, double bracketEnd )
{
	double root = 1; 
	if ((fn(bracketOpen) > 0 && fn(bracketEnd) > 0) || (fn(bracketOpen) < 0 && fn(bracketEnd) < 0)) // There is no change in sign => No root in bracke
	{
		error("rootBisection -> Bracket is not proper ");
	}
	else
	{
		 int counter = _iterations; 
		while (counter > 0)
		{
			counter--;

			// We divide region into half and selects the region where sign changes 
			//double newPoint = (bracketEnd + bracketOpen) / 2;
			double newPoint = bracketEnd + (bracketOpen - bracketEnd) / 2;  // Higher Accuracy !! 

			// Which Region to Choose
			if (fn(newPoint) == 0) // Root is the new point 
			{
				root = newPoint; return root; 
			}
			else if ((fn(bracketOpen) * fn(newPoint)) < 0) // Sign Changes
			{
				//std::cout << "@" << newPoint; 
				bracketEnd = newPoint; 	
			}
			else //(fn(bracketEnd)* fn(newPoint) < 0) // Sign Changes // If above conditions not met then root  lies in this interval
			{
				//std::cout << "@2";
				bracketOpen = newPoint; 
			}

			if (stoppingCriterion(fn(bracketOpen), fn(bracketEnd) ))
			{
				//cout << root; 
				root = bracketOpen;
				return root;
			}
			 

		}
		error("rootBisection -> Iteration Limit Exceeded ");
	}
	root = bracketOpen;
	return root; 
}


double analysis::rootNewtonSingle(std::function<double(double)> fn, double begin, double end)
{
	double x = begin;
	//cout << isInfinitesimal(fn(10));																					//DEBUG
	while (!isInfinitesimal(fn(x))) // Untill fn(x) is not close to zero , we try to continue the guess 
	{
		x = x - fn(x) / derivativeAverage(fn, x);
		//cout << "DEBUG 1" << endl;																					//DEBUG
		if (x > end)
		{
			error("newtonRootSingle -> Couldn't find root within specified Range");
			return 0;
		}
	}
	return x; // Successfully found root 
}


double analysis::rootNewtonMultiplicity(std::function<double(double)> fn, double begin, double end)
{

	double x = begin;
	//cout << isInfinitesimal(fn(10));																					//DEBUG
	while (!isInfinitesimal(fn(x))) // Untill fn(x) is not close to zero , we try to continue the guess 
	{
		x = x - ( fn(x) * derivativeAverage(fn, x) ) / ( derivativeAverage(fn, x)*derivativeAverage(fn, x) - fn(x)*secondDerivative(fn, x) );
		//cout << "DEBUG 1" << endl;																					//DEBUG
		if (x > end)
		{
			error("newtonRootSingle -> Couldn't find root within specified Range");
			return 0;
		}
	}
	return x; // Successfully found root 
}



matrix<double> analysis::rootQuadratic(double a, double b, double c)
{
	matrix<double> result(2, 1); // Column Vec , 2 row , 1 col 
	double D = b*b - 4 * a*c;
	if (D < 0)
	{
		error("rootQuadratic -> Complex Roots Currently Not Supported ");
	}
	else
	{
		result(1, 1) = -b + std::sqrt(D) / (2 * a);
		result(2, 1) = -b - std::sqrt(D) / (2 * a);
	}
	return result;
}

/*
There are no bounds on this root finding method 
*/
double analysis::rootSecant(std::function<double(double)> fn)  // DOES NOT WORK
{
	double x_i = 1; double x_i_1 = 1.5;  // RANDOM 
	double x_i_m1 = 0.5; // x i minus 1 

	//cout << isInfinitesimal(fn(10));		//DEBUG

	int counter = _iterations;
	while (counter > 0) // Untill fn(x) is not close to zero , we try to continue the guess
	{
		//x_i_1 = x_i - fn(x_i) * ( (x_i - x_i_0) / (fn(x_i) - fn(x_i_0)) );

		x_i_1 = (x_i_m1 * fn(x_i) - x_i * fn(x_i_m1)) / (fn(x_i) - fn(x_i_m1));

		x_i_m1 = x_i;
		x_i = x_i_1;

//		cout << x_i_1 << endl;
		if (stoppingCriterion(x_i_1, x_i))
			return x_i;
		counter--;
	}
	return x_i; // Successfully found root

}


/*
REGULAR FALSE POSITION METHOD 

Secant Method modified to include bounds 

We fix one point and move from the other point to the fixed point . The fixed point will be end here. 

For each root approx we find , we will see where the root lives using the intermediate value theorem .
*/
double analysis::rootRegula_Falsi(scalarFn_1 fn, double begin , double end )
{

	double x_i = end - 1; double x_i_1  ;  // RANDOM 
	double x_i_m1 = begin;

	int counter = _iterations;
	while (counter > 0 ) // Untill fn(x) is not close to zero , we try to continue the guess
	{
		x_i_1 = (x_i_m1 * fn(x_i) - x_i * fn(x_i_m1)) / (fn(x_i) - fn(x_i_m1));

	//WE FIND NEW APPROX , we check which interval the root is in 
		if ( fn(x_i_1) < _tolerance)
		{
			return x_i_1; // ROOT FOUND 
		}
		else if (fn(begin) * fn(x_i_1) < 0 ) // Since root lies between begin and x_i , end is no longer needed
		{ 
			//x_i_1 need not be changed and x_i is already changed 
			x_i_m1 = begin; // not end 
			end = x_i_1;
		}
		else // the root lies between x_i and end , hence 
		{
			x_i_m1 = end; // not end 
			x_i = x_i_1; 
		}
		
		if (stoppingCriterion(x_i_1, x_i))
			return x_i;
		counter--;
	}
	return x_i; // Successfully found root

}


double analysis::rootChebyshev(scalarFn_1 fn, double guess)
{
	double x, a_0, a_1, a_2 ;  // a_0 * x*x + a_1 *x + a_2  
	x = guess;
	a_0 = secondDerivative(fn, x) / 2;
	a_1 = derivativeAverage(fn, x) - 2 * a_0 * x; 
	a_2 = fn(x) - a_0 * x * x - a_1 * x; 

	double xk, xk_1, fk_2, fk_1, fk; xk = x; 
	double counter = 0; 
	while (counter < _iterations)
	{
		fk_2 = secondDerivative(fn, xk);
		fk_1 = derivativeAverage(fn, xk);
		if (fk_1 < _tolerance && fk_2 < _tolerance)
		{
			error(" -> rootChebyshev -> cannot calculate derivatives "); 
		}

		fk = fn(xk);

		xk_1 = xk - fk / fk_1 - (1 / 2) * ((fk_2)*  (fk*fk)) / (fk_1 * fk_1 * fk_1);
		
		if (fn(xk_1) < _tolerance)
			return xk_1; 
		if (stoppingCriterion(xk_1, xk))
			return xk_1;
		counter --; 
	}
	return xk_1; 
}


double analysis::rootMuller(scalarFn_1 fn, double guess_1, double guess_2, double guess_3)
{

	return 0; 
}

/*
	Find Root using Halley's Method 
*/
double analysis::rootHalley(std::function<double(double)> fn, double begin, double end )
{
	double x = begin;
	//cout << isInfinitesimal(fn(10));																					//DEBUG
	while (!isInfinitesimal(fn(x))) // Untill fn(x) is not close to zero , we try to continue the guess 
	{
		x = x - ( 2 * fn(x) * derivativeAverage(fn, x) ) / ( 2 * (std::pow(derivativeAverage(fn,x) , 2 )) - fn(x) * secondDerivative(fn,x) );
		//cout << "DEBUG 1" << endl;																					//DEBUG
		if (x > end)
		{
			error("rootHalley-> Couldn't find root within specified Range");
			return 0;
		}
	}
	return x; // Successfully found root 
}


//-----------------------------------------------SPEICAL INTEGRATIONS-------------------------------------------------//
double analysis::convolution(std::function<double(double)> fn, std::function<double(double)> fn2, double finalT)
{
	return integralBest([&](double tau) { return fn(tau)*fn2(tau-finalT) ; }, 0, finalT);
}







//-----------------------------------------------------BEST--------------------------------------------------------------------//
double analysis::differentiateBest(std::function<double(double)> fn, double x)
{
	return derivativeCenter(fn,  x);


}
// KEEP TRACK OF THE BEST INTEGRATION FORMULA 
double analysis::integralBest(std::function<double(double)> fn, double begin, double end)
{
	return integralSimpson3_8Composite(fn, begin, end);
}





//----------------------------------------------------MULTIVARIABLE --------------------------------------------------------------------------//

matrix<double> analysis::gradient_2D(scalarFn_2 fn, double x, double y)
{
	matrix<double> R(2, 1);

	std::function<double(double)> fnX = [fn, y](double xVar){ return fn(xVar, y); };
	std::function<double(double)> fnY = [fn, x](double yVar){ return fn(x, yVar); };

	R(1, 1) = differentiateBest(fnX, x);
	R(2, 1) = differentiateBest(fnY , y);

	return R;
}

matrix<double> analysis::gradient_3D(scalarFn_3 fn, double x, double y, double z)
{
	matrix<double> R(3, 1);

	std::function<double(double)> fnX = [fn, y, z](double xVar){ return fn(xVar, y,z); };
	std::function<double(double)> fnY = [fn, x, z](double yVar){ return fn(x ,yVar, z); };
	std::function<double(double)> fnZ = [fn, x, y](double zVar){ return fn(x, y, zVar); };

	R(1, 1) = differentiateBest(fnX, x);
	R(2, 1) = differentiateBest(fnY, y);
	R(3, 1) = differentiateBest(fnZ, z);

	return R;
}

/*
Computing the Hessian :
Hessian(fn)[i,j] https://inst.eecs.berkeley.edu/~ee127a/book/login/def_hessian.html
*/

matrix<double> analysis::hessian_ND(scalarFn_N fn, matDouble N) // UNTESTED
{
	matrix<double> R(N.numRows(), N.numRows());
	for (int i = 1; i <= R.numRows(); i++)
	{
		for (int j = 1; j <= R.numCols(); j++)
		{
			R(i, j) = mixed2_partial_deriv_Center_multiDim(fn, N, i, j);
		}
	}
	return R;
}


matDouble analysis::jacobian_3D(vectorFn_3 fn, double x, double y, double z) // UNTESTED
{
	matDouble R(3, 3);
	matrix<double> fixed(3, 1);
	fixed(1, 1) = x; fixed(2, 1) = y; fixed(3, 1) = z;
	// Assume vectorFn is a col Vector
	R.replaceCol(gradient_3D(fn(1, 1), fixed(1, 1), fixed(2, 2), fixed(3, 3)), 1); // We take the gradient of function and replace the row by it ! 
	R.replaceCol(gradient_3D(fn(2, 1), fixed(1, 1), fixed(2, 2), fixed(3, 3)), 2);
	R.replaceCol(gradient_3D(fn(3, 1), fixed(1, 1), fixed(2, 2), fixed(3, 3)), 3);
	return R;
}


matrix<double> analysis::jacobian_ND(vectorFn_N fn, matDouble N) // UNTESTED
{
	matDouble R(N.numRows(), N.numRows());
	for (int i = 1; i <= R.numRows(); i++)
	{
		R.replaceCol(gradient_ND(fn(i, 1), N), i);
	}
	R = la.tranpose(R); // Make it into correct form ; 
	return R;
}

/*

matrix<double> analysis::hessian_3D(scalarFn_3 fn, double x, double y, double z) // UNTESTED
{
matrix<double> R(3, 3);
matrix<double> fixed(3, 1);
fixed(1, 1) = x; fixed(2, 1) = y; fixed(3, 1) = z;
for (int i = 1; i <= R.numRows(); i++)
{
for (int j = 1; j <= R.numCols(); j++)
{
R(i, j) =  mixed2_partial_deriv_Center_multiDim(fn, fixed, i, j); // This function only takes scalarFn_N as fn input , not 3_N 
// I am not reimplementing this fuction for this reason 
R(i, j) = i * j;

}
}
return R;
}












*/

/*
IMPLEMENTATION
 
 I am passed a function with N inputs and a matrix with N values. 
 Gradient means I have to calculate how the function changes with respect to each of those N values. 

 ( f( x + h ) - f( x ) ) / h  for each of thoses N values , while the others are kept constant. 
 for this I will have to create N functions which takes in Just a single values , and the others are fixed. 

nDimensions has to be a column vector .

*/


/*
Learing Exercise Harder || 
template<typename R, typename ...A>
matrix<double> analysis::gradient_ND( std::function<R(A...)>  fn, matrix<double> N)

*/




// Easier Way to Do things. KISS 
// Go through Each value and take a gradient at that point. 
matrix<double> analysis::gradient_ND(scalarFn_N fn, matrix<double> N)
{
	matrix< double > A(N.numRows(), 1);
	for (int i = 1; i <= N.numRows(); ++i)
	{
		A(i, 1) = partial_deriv_Center_multiDim(fn, N, i);
	}
	return A; 
}


/*
Pass in a  function with N arguments and N dimensional Matrix. 
Also pass in an index so that the gradient will be found by varying that varaible
while the other variables are kept constant. 

return (fn(x + _stepSize) - fn(x - _stepSize)) / (2 * _stepSize);


Column Vectors are used. 
*/
double analysis::partial_deriv_Center_multiDim(scalarFn_N fn, matrix<double> N, int currIndex) // Private Function
{
	//cout <<"ROWS : " <<N.numRows() << endl;					///DEBUG
	//matrix<double> input_h_p(N.numRows(), 1); // fn(x + h) // p = plus
	//matrix<double> input_h_m(N.numRows(), 1); // fn(x - h ) // m = minus
	

	/*for (int i = 1 ; i <= N.numRows() ; ++i)
	{
		if (i == currIndex)  // This is the variable we want to wiggle 
		{
			input_h_p(i , 1) = N(i,1) + _stepSize;
			input_h_m(i, 1) = N(i, 1) - _stepSize;
		}
		else // Variables kept constant at these points
		{
			input_h_p(i, 1) = N(i, 1);
			input_h_m(i, 1) = N(i, 1);
		}
	}*/

	// Better Implementation 
	matrix<double >input_h_p = N; 
	matrix<double >input_h_m = N;

	input_h_p(currIndex, 1) = N(currIndex, 1) + _stepSize;
	input_h_m(currIndex, 1) = N(currIndex, 1) - _stepSize;

	return (fn(input_h_p) - fn(input_h_m)) / ( 2 * _stepSize ); 
}


double analysis::mixed2_partial_deriv_Center_multiDim(scalarFn_N fn, matrix<double> N, int index1, int index2) // UNTESTED 

/*
	Look at : https://wwz.unibas.ch/fileadmin/wwz/redaktion/qm/downloads/Lehre/CompEcon/12FS/CE_12_01_NumDiffInt.pdf
	Page 5 ; 

	Future Ideas : Optimize ? 
*/
{	
	double result = 0; 
	if (index1 == index2)
	{
		std::function<double(double)> fnVar = [fn, N,index1](double Var)
		{ // Create a new input to the function which has the same elements as N , but element at index1 being substitued by Var
			matrix<double> N_temp = N; 
			N_temp(index1,1) = Var; 
	//		N_temp.print();													//DEBUG
			return fn(N_temp);
		};
		//std::cout <<"fnVAR ::"<< fnVar(100) << endl;
		result = secondDerivative(fnVar, N(index1, 1));
		//std::cout << "fnVAR ::" << fnVar(100) << endl;
	}
	else
	{
		matrix<double> input_h_p_p = N; // fn(x + h) // p = plus
		matrix<double> input_h_m_m = N; // fn(x - h ) // m = minus // Will Replace the required Values
		matrix<double> input_h_p_m = N;
		matrix<double> input_h_m_p = N;
		input_h_p_p(index1, 1) = N(index1, 1) + _stepSize;
		input_h_m_m(index1, 1) = N(index1, 1) - _stepSize;
		input_h_m_p(index1, 1) = N(index1, 1) - _stepSize;
		input_h_p_m(index1, 1) = N(index1, 1) + _stepSize;

		input_h_p_p(index2, 1) = N(index2, 1) + _stepSize;
		input_h_m_m(index2, 1) = N(index2, 1) - _stepSize;
		input_h_m_p(index2, 1) = N(index2, 1) + _stepSize;
		input_h_p_m(index2, 1) = N(index2, 1) - _stepSize;

		//input_h_m_m.print();													//DEBUG
		//std::cout <<" fn(input_h_p_p) " << fn(input_h_p_p) << endl;
		//std::cout << " fn(input_h_m_p) " << fn(input_h_m_p) << endl;
		//std::cout << " fn(input_h_p_m) " << fn(input_h_p_m) << endl;
		//std::cout << " fn(input_h_m_m) " << fn(input_h_m_m) << endl;

		result = fn(input_h_p_p) + fn(input_h_m_m) - fn(input_h_m_p) - fn(input_h_p_m);
		//std::cout << "Difference "<< result << endl;
		result /= (4 * _stepSize * _stepSize);
	} 

	return result;
}


double analysis::divergence_3D(vectorFn_N vecFn, matrix<double> N)
{
	matrix<double> R(N.numRows(), 1); // No need for this complicated way , a simple double and adding it up will work . Done for generality.
	for (int i = 1; i <= R.numRows(); i++)  // Calculate derivative for i th component
	{
		R(i, 1) = partial_deriv_Center_multiDim(vecFn(i, 1), N, i);
	}
	double result = 0;
	for (int i = 1; i <= 3; i++)  // Add up the derivatives
	{
		result += R(i, 1);
	}
	return result;
}

matrix<double> analysis::curl_3D(vectorFn_N vecFn, matrix<double> N)
{
	matrix<double> R(N.numRows(), 1); // No need for this complicated way , a simple double and adding it up will work . Done for generality.

	R(1, 1) = (partial_deriv_Center_multiDim(vecFn(3, 1), N, 2) - partial_deriv_Center_multiDim(vecFn(2, 1), N, 3));
	R(2, 1) = (partial_deriv_Center_multiDim(vecFn(1, 1), N, 3) - partial_deriv_Center_multiDim(vecFn(3, 1), N, 1));
	R(3, 1) = (partial_deriv_Center_multiDim(vecFn(2, 1), N, 1) - partial_deriv_Center_multiDim(vecFn(1, 1), N, 2));

	return R; 
}





vectorFn_1 analysis::get_func_2D_paramT(scalarFn_1 fn1, scalarFn_1 fn2)
{
	matrix<scalarFn_1> R(2, 1); // Col Vec with 2 ;  
	R(1, 1) = fn1;
	R(2, 1) = fn2; 
	return R; 
}


vectorFn_1 analysis::get_func_3D_paramT(scalarFn_1 fn1, scalarFn_1 fn2, scalarFn_1 fn3)
{
	matrix<scalarFn_1> R(3, 1); // Col Vec with 3 ;  
	R(1, 1) = fn1;
	R(2, 1) = fn2;
	R(3, 1) = fn3;
	return R;
}


vectorFn_3 analysis::get_func_3D_param3(scalarFn_3 fn1, scalarFn_3 fn2, scalarFn_3 fn3)
{
	matrix<scalarFn_3> R(3, 1); // Col Vec with 3 ;  
	R(1, 1) = fn1;
	R(2, 1) = fn2;
	R(3, 1) = fn3;
	return R;
}


vectorFn_N analysis::get_func_3D_paramN(scalarFn_N fn1, scalarFn_N fn2, scalarFn_N fn3)
{
	matrix<scalarFn_N> R(3, 1); // Col Vec with 3 ;  
	R(1, 1) = fn1;
	R(2, 1) = fn2;
	R(3, 1) = fn3;
	return R;
}




























/*

DEPRECATED

matrix<double> analysis::rootNewtonMultiple(std::function<double(double)> fn, double begin, double end) // Not Working

Does newton's method , if a root is found , we perturb the guess forward by 10 (random),
and try to find another root.

Store the found root in an std::vector ;
then finally copy it back to a matrix and return the matrix. // Later Improvement , make the matrix Dynamic to prevent dependence on std::vector

{
	double x = begin; vector<double> storage;
	//cout << isInfinitesimal(fn(10));																					//DEBUG
	while (x < end) // Untill x is not close to end , we try to continue the guess 
	{
		x = x - fn(x) / derivativeAverage(fn, x);


		if (isInfinitesimal(fn(x))) // if A root has been found 
		{
			cout << "DEBUG 1" << fn(x) << endl;			//DEBUG
			storage.push_back(x);
			x += 100;
		}

	}

	int size = storage.size();
	matrix<double> result(size, 1); // col Vector
	if (size == 0)
	{
		error("newtinRootSingle -> Couldn't find root within specified Range");
		return result;
	}
	else
	{

		for (int i = 0; i < size; i++)
		{
			result(i + 1, 1) = storage[i];
		}
	}
	return result; // Successfully found root 
}









*/
