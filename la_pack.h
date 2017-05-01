#ifndef LA_PACK_H
#define LA_PACK_H
/*
	Provide all the Functionality of a Linear Algebra Library 


	Create a lapack object and then call the functions 

	Interface
		general (applies operation to matrix and returns a new matrix)
		determinant
		inverse
		gauss Jordan 
		A|T method

	Implementation
		
		


	Log
	April 4 12:00 pm - Start 
	April 5 1:00 pm - Begin 
	APril 9 Slight Progress 

*/



//Personal Imports
#include "matrix.h" // We will use double matrices



//System Imports 
#include <cmath> 
#include <string>  // Error Handling 
#include <functional>
#include <iostream>
#include <tuple>




class la_pack
{
public:
	la_pack(void);
	~la_pack(void);
	
	
	double determinat(matrix<double> &A);
	matrix<double> getIdentity(long long  aRow );
	matrix<double> getCofactor(matrix<double> &A);
	matrix<double> tranpose(matrix<double> &A);
	matrix<double> inverseDeterminant(matrix<double> &A);
	matrix<double> inverseGuassJordan(matrix<double> &A);
	/*
		Pseudo Inverse 
		Give A will return (A'A)^(-1)*A'
		Used in Least Squares 
	*/
	matrix<double> inversePenrose(matrix<double> A);

	/*
	Takes the Pseudo Inverse using QR factorization 

	*/
	matrix<double>  inversePseudoQR(matrix<double> A);

	/*
		Takes in A and B , gives back X 
		X = inversePenrose(A)*B
	*/
	matrix<double> leastSquareSolver(matrix<double> A, matrix<double> B);  // Tested ! 

	/*
		Give two column vectors : Gives error between the two 
		Vec 1 - Vec 2
	*/
	matrix<double> getResidue(matrix<double> A, matrix<double> B);


	/*
		Sometimes A does not contain enough information for even least sqaures. 
		Then we do regularization.
		Tikonov regularization
	*/
	matrix<double>  regularizationTikonon(matrix<double> A, double alpha);

	matrix<double>  ridgeRegression(matrix<double> A, double alpha);



	/*
	Gaussian Elemination Using Partial Pivoting [Only Row Pivots]
	Returns AB with the A being converted into an upper Trainaguler Matrix . 
	*/
	matrix<double> gaussianElemPartialPivot(matrix<double> &A, matrix<double> &B , bool roundOff = true);

	/*
	Returns a single Matrix in the L U form
	User will have to Splice the L and U into seperate Matrices
	USE getL_from_LU and getU_from_LU to get the results. 
	*/
	matrix<double> LUDecomposePartialPivot(matrix<double> &A);
	matrix<double> getL_from_LU(matrix<double> &LU);
	matrix<double> getU_from_LU(matrix<double> &LU);




	matrix<double> gramSchmidtOrtho(matrix<double> A,bool normalize);
	std::tuple<int ,matrix<double> > getRank(matrix<double> A, bool normalize);
	/*
	Rank Revealing QR factorization
	Returns a tuple with the first element the rank and the second element the matrix ! 
	*/

	matrix<double> projectionMatrix(matrix<double> A);

	matrix<double> choleskyFactorize(matrix<double> A);






	matrix<double> backSubstitution(matrix<double> &A);
	matrix<double> outerProduct(matrix<double> &A, matrix<double> &B);
	double innerProduct(matrix<double> const &A, matrix<double> const &B); // General Form tran(A)*A
	double normEuclidean(matrix<double> &A);
	double normL1(matrix<double> &A);
	double normLInf(matrix<double> &A);
	double normChebeshev(matrix<double> &A);
	double normL_P(matrix<double> &A,double p );
	double normWeight_P(matrix<double> &A, matrix<double> Weight, double p);
	double  polynomial(matrix<double> Coefficient, double x);
	matrix<double>  vandermondeMatrix(matrix<double> X);
	matrix<double>  polynomialMatrix(matrix<double> Coefficient, matrix<double> vandermondeMatrix);




	/*
	Find Norm of a matrix.
	Go through each each element square it , add up , then take root
	Unwind matrix into single row , then take norm of the row .
	*/
	double normForbenius(matrix<double> &A);
	double trace(matrix<double> &A);

	/*
	Gives Projection of A onto B
	*/
	matrix<double> projectionOn(matrix<double> &A, matrix<double> &B);
	/*
	Gives Projection of A orthogonal to  B
	*/
	matrix<double> projectionOn_Ortho(matrix<double> &A, matrix<double> &B);
	matrix<double> solverGussianElem(matrix<double> &A, matrix<double> &B);
	matrix<double> solverLU(matrix<double> &A, matrix<double> &B);
	double corelationCoeff(matrix<double> &A, matrix<double> &B);
	double angle(matrix<double> &A, matrix<double> &B);
	bool isOrthogonalVectors(matrix<double> &A, matrix<double> &B);
	matrix<double> getTriDiagonal(long long Rows , double diagonalElem , double aboveDiagElem, double belowDiagElem);
	
	/*
	Q*transpose(Q) == I
	returns true or false
	if of diifferent size , returns false
	*/
	bool isSymetric(matrix<double> &Q);
	bool isPositiveDefinite(matrix<double> &Q);
	matrix<double>  getGramMatrix(matrix<double>& A);



	//--------------------------------ITERATIVE METHODS --------------------------------------------------------------------//


	matrix<double> jacobiSolver(matrix<double> &A, matrix<double> &B);
	matrix<double> gaussSidelSolver(matrix<double> &A, matrix<double> &B);
	matrix<double> getLinspaceRow(double start,double end, double interval = 0.5);
	matrix<double> getLinspaceCol(double start,double end, double interval = 0.5);
	matrix<double> joinCol(matrix<double> &A, matrix<double> &B); // A and B will same row Num
	matrix<double> joinRow(matrix<double> &A, matrix<double> &B); // A and B will same col Num
	matrix<double> spliceCopyRows(matrix<double>& rhs, int aRowStart , int aRowEnd );
	matrix<double> spliceCopyCols(matrix<double>& rhs, int aColStart, int aColEnd );
	matrix<double> swapRow(matrix<double>& rhs, int firstRow, int secondRow );
	matrix<double> swapCol(matrix<double>& rhs, int firstCol, int secondCol );
	matrix<double> getUnitVec(matrix<double> rhs);

	double sum(matrix<double>& rhs);

	//Scheme Map Function 
	// Applies a function to each value in the matrix and then return back a matrix 
	// Will only accept functions within input and output as double 
	// If function is a member function , it should be static . 
	matrix<double> mapMatrixDouble(matrix<double>& rhs, std::function<double(double)> fn);


	/*
		Returns a block column vector containing A and B as Column Blocks. 
		     _ _
		x = | A |
			| B |
			
		INPUT : 
			A and B must be matrix with same Column Size
		OUTPUT: 
			Will be a Matrix with 
			cols = A.cols = B.cols 
			rows = A.rows + B.rows 
	*/
	matrix<double> createBlockColVec_2(matrix<double> A, matrix<double> B);

	/*
	Returns a block row vector containing A and B as Row Blocks.
		 _     _
	x = | A  B |

	INPUT :
	A and B must be matrix with same Row Size
	OUTPUT:
	Will be a Matrix with
	cols = A.cols + B.cols
	rows = A.rows = B.rows
	*/
	matrix<double> createBlockRowVec_2(matrix<double> A, matrix<double> B);

	/*
	 Returns 
			 __  __
		x = | A	 C |
			| B	 D |

	INPUT : 
		A and B same Col Size ; C and D same Col Size ; 
		A and C same Row Size ; B and D same Row Size ; 
		Input and Index in Function Call 
		A - 1 
		B - 2 
		C - 3 
		D - 4
	OUTPUT : 
		Will be a matrix with 
			cols = A.cols + C.cols 
			rows = A.rows + B.rows 
	*/
	matrix<double> createBlockSquare_4(matrix<double> A, matrix<double> B, matrix<double> C, matrix<double> D);

	




	//----------------------------------------------TO IMPLEMENT --------------------------------------------------------------------------

	matrix<double> eigenvalues(matrix<double> A);
	matrix<double> eigenvectors(matrix<double> A);
	matrix<double> eigenvalueDecomposition(matrix<double> A);


	double  rayleighQuotient(matrix<double> A, matrix<double> X);



	/*
	
	Returns  matrix 
	Columns contain the Eigen Values 
	Last COlumn will contain the found eigen vectors . 
	*/
	matrix<double> eigenPower(matrix<double> A, bool rayleigh = false);
	matrix<double> eigenPowerInverse(matrix<double> A, bool rayleigh = false);

	matrix<double> eigenHotellingDeflation(matrix<double> A, bool rayleigh = false);






	matrix<double> eigenQR(matrix<double> A);

	/*
	DIFFERENT METHODS 

	Jacobi 
	Givens 
	Householder
	QR
	LR

	Krylov Subspace Method 
	Arnoldi Iteration 
	Lanczos Iteration 
	
	
	*/


	/*
		The matrix will contain vectors represented along a Row 
	*/
	matrix<double> QRFactorizeRow(matrix<double> A, bool normalize = false);

	/*
	Followed by Scientific Community 
	*/
	matrix<double> QRFactorizeCol(matrix<double> A, bool normalize = false);  // CORRECT

	// EXtract Q and R from QR ; 
	matrix<double> getQ_from_QR(matrix<double> QR);
	matrix<double> getR_from_QR(matrix<double> QR);


	/*
		SVD 
	*/
	

	std::tuple< matrix<double>, matrix<double>, matrix<double>> SVD(matrix<double> A); // Singular Value Decomposition ;/




	//------------------------------------------TRANFORMATIONS -----------------------------------------------------//
	/*
	Takes a Column vector
	Converts it into a unit vector
	Returns a Traidionagonal Matrix ,

	H = I - 2*u*u' // u -> unit vector 
	*/
	matrix<double> houseHolderTransformation(matrix<double> U);
	/*
	Scales the result rather than create a unit vector. 
	
	*/
	matrix<double> houseHolderTransformation_Scale(matrix<double> U);
	matrix<double>  houseHolderTransformationX(matrix<double> U, matrix<double> X);


	/*
	alpha > 1 => Strech 
	alpha < 1 => Shrink
	*/
	matrix<double> strech_shrink(matrix<double> U, double alpha);


//---------------------------------------------------ART OF SCIENTIIC PROGRAMMING------------------------------------------------------------------//
// NOT MINE : I ONLY HAVE A VAGUE UNDESTANDING OF THE CODE ... 
	/*
		Gauss Jordan With Pivoting
		NOTES : 
			Full Pivoting :: Inplace to save Storage 
			Rock Solid Method for taking inverse , but inefficeint compared to other methods
		INPUT :
			A and B from A x = B form
			A is Square . 
		OUTPUT : 
			Inplace Replace A and B with inv(A) and x
	*/
	void gaussJordan(matrix<double> &A, matrix<double> &B);

	/*
		NOTES:
			OverLoading Method
		INPUT: 
			A Square 
		OUTPUT:
			inv(A) INPLACE 
	*/
	void  gaussJordan(matrix<double> &A);

	/*
	LU Decomposition Method from 
	Crout's algorithm
	Returns a matrix , not in place >> InEfficient but Usable ..
	Better Method 
	*/
	matrix <double> decomposeLU(matrix<double> &A);


	/*
	Similar to above decomposeLU , but if input nool is true , will return the 
	permutations as the second element of the tuple 

	Bool Value does not matter , used only to help the compiler distinguish between the two 
	*/
	std::tuple<matrix<double>, matrix<int>> decomposeLU(matrix<double> &A,bool returnPivot);

	/*
		NOTES:
			Solves AX = B
			Using LU Method 
		INPUT: 
			A , B
		OUTPUT:
			X
		IMPLEMENTATION DETAILS: 
			Calls LU internally 
			
	*/
	matrix <double> solveLU(matrix<double> &A, matrix<double> &B);



	/*
		Does the same thing at solveLU above , but this is used when solveLU is called multiple times 
		and we do not want to decomposeLU each time. 
	
	*/
	matrix <double>  solveLU(matrix<double> &A, matrix<double> &B, matrix<double> &LU , matrix<int> pivot);


	/*
	NOTES:
	Solves AX = B
	Using LU Method
	But with X and B being matrices containing multiple vectors . 
	INPUT:
	A , B , n = num of right hand sides // Give it n = B.numCols();
	OUTPUT:
	X with solutions of each of the different right sides ..
	IMPLEMENTATION DETAILS:
	Calls LU internally

	*/
	matrix <double> solveLU(matrix<double> &A, matrix<double> &B , int numRightSides );


	matrix<double> inverseLU( matrix<double> & A);


	double  determinantLU(matrix<double> & A);




	matrix<double> mapMatrixdouble(matrix<double>& rhs, std::function<double(double)> fn);








private:
	double deterRecur(matrix<double> A);
	double _tolerance = 1e-4;
	double _iterations = 1e4;
	static void error(const char* p ){std::string str= "la_pack -> Error: "; std::cout << str << p << std::endl;}
	matrix<double> rowEchelonReduceUToI(matrix<double> &A, bool roundOff = true);
	bool unstable(double A) { return (std::abs(A) < ((10) ^ (-6))) ? true : false; };
	bool isInfinitesimal(double x){ return std::abs(x) < _tolerance; }
	bool relativeChangeSmall(double x_i_1, double x_i){ return ((x_i_1 - x_i) / x_i < _tolerance) ? true : false; }
	bool stoppingCriterion(double x_i_1, double x_i){ return relativeChangeSmall(x_i_1, x_i) && isInfinitesimal(x_i_1); }
	// No need for any private members .  


};


#endif

