#pragma once

/*
	This class is used to ensure that what I do is tested , using actual Numerical values . 

	From the Source File call runTestSuite() ; 

	This will run all the test cases and return true or false . 

	I will use print statement to let the user know when things went wrong ! 

	Not Using a Test will be troublesome  , as I will not know which works and which doesn't .. 

	NNRALES AIRE typeZERO . 


	July 4th 


*/

//System
using namespace std;
#include <iostream>
#include <cmath>
#include "matrix.h" 
#include "la_pack.h"
#include <vector>
#include "analysis.h"
#include "func.h"
#include "dynamics.h"
#include "common.h"
#include "setMath.h"
#include "probability.h"

#include "filter.h"
#include "quaternion.h"
class testSuite
{
public:
	testSuite();
	~testSuite();
	bool runTestSuite();
	bool testGradient();
	bool testMatrix();
	bool testHessian();
	bool testJacobian();
	bool testLU();
	void setDebugMessage(bool a){ debugMessages = a;  }

	bool testFilter(); 
	bool testQuaternion();


private:
	bool pass = true; // Initially think that we fail , then on pass all the test , we will make it true !
	bool debugMessages = false; 
	static void error(const char* p){ std::string str = "testSuite-> Failed: "; std::cout << str << p << std::endl; }
	static void passMessage(const char* p){ std::string str = "testSuite-> Passed: "; std::cout << str << p << std::endl; }
	la_pack la_pack; analysis analysis; func func;
};

