#include "la_pack.h"


la_pack::la_pack(void){}


la_pack::~la_pack(void){}

// -----------------------------------------------------------------------------------------------------------------------------------------------
// Create numbers
matrix<double> la_pack::getLinspaceRow(double start,double end, double interval)
	/*
		Returns a rowVector with numbers begining at start , ending at end with interval interval ; 
		Inclusive Inclusive range 
	*/
{	
	double size = (end - start) / interval;
	matrix<double> R(1,size+1); // Sloppy figure out why size + 1
	for(double i = start , k = 1; i <= end ; i+= interval,k+=1) // Condition only depend on start and end 
	{
		R(1,k) = i ; 
	}
	return R ; 
}


matrix<double> la_pack::getLinspaceCol(double start,double end, double interval)
	/*
		Returns a colVector with numbers begining at start , ending at end with interval interval ; 
		Inclusive Inclusive range 
	*/
{	
	double size = (end - start) / interval;
	matrix<double> R(size + 1,1); //
	for(double i = start ,k = 1; i <= end ; i+= interval,k+=1)// Condition only depend on start and end 
	{
		R(k,1) = i ; 
	}
	return R ; 
}

matrix<double> la_pack::getIdentity(long long  aRow)
{
	matrix<double> A(aRow,aRow);
	for(long long  i = 1 ; i <= aRow ; i++)
	{
		for(long long  j = 1 ; j <= aRow ; j++)
		{
			if(i == j )
			{
				A(i,j) = 1 ; 
			}
		}
	}
	return A; 
}

matrix<double> la_pack::getGramMatrix(matrix<double>& A)
{
	matrix<double> AT = tranpose(A);
	return AT*A; 
}


matrix<double> la_pack::getTriDiagonal(long long Rows , double diagonalElem , double aboveDiagElem, double belowDiagElem)
{
	matrix<double> A(Rows,Rows);
	for(long long  i = 1 ; i <= Rows ; i++)
	{
		for(long long  j = 1 ; j <= Rows ; j++)
		{
			if(i == j )
			{
				A(i,j) = diagonalElem ;
			}
			else if( i - j == 1) //  diagonal
			{
				A(i,j) = belowDiagElem;
			}
			else if( j - i == 1)
			{
				A(i,j) = aboveDiagElem;
			}
			else
			{
				continue; 
			}

		}
	}
	return A; 

}
//----------------------------------------------------------------------------------------------------------------------------------------------
// Determinants
double la_pack::determinat(matrix<double> &A)
{
	if(A.isSquare())
	{
		return deterRecur(A);
	}	
	else
	{
		error("No Determinant for Non Square Matrices") ; // Replace with Error 
		return 0 ; 
	}
}


// Expands along row one 
double la_pack::deterRecur(matrix<double> A)
{ // assume A is square , only called if A is square 
	if(A.numRows() == static_cast<long long >(2))
	{
		return (A(1,1) *  A(2,2) - A(1,2) * A(2,1));
	}
	else
	{
		double deter = 0 ; int sign = 1; 
		for(long long  i = 1 ; i <= A.numRows(); i++ )
		{
				
			// Deciding Sign 
				if( i % 2 == 0 ) 
				{
					sign = -1; // for first col , sign is positive , for second row sign is negative ...
				}
				else
				{
					sign = 1; 
				}
				double current = A(1,i);
				matrix<double> R(A.numRows() - 1, A.numCols() - 1);
				R = A.removeRow(1).removeCol(i);
				deter += current*sign*deterRecur(R);
		}
		return deter;
	}
}


matrix<double> la_pack::getCofactor(matrix<double> &A)
{
	matrix<double> R( A.numRows() , A.numCols() );
	if(A.isSquare())
	{
		int sign = 1;
		
		for(long long  i = 1 ; i <= A.numRows() ;i++)
		{
			for(long long  j = 1; j <= A.numCols() ; j++)
			{
				(i+j % 2 == 0) ? sign = 1 : sign = -1;
					matrix<double> temp = A; 
					matrix<double> tempVar = temp.removeCol(j).removeRow(i); // to get around g++ error : cannot bind lvaule to const reference
					R(i,j) = determinat(tempVar) * sign; 
			}
		}
		return R; 
	}
	else
	{
		error( "Not Square " );
		return R ; 	
	}
}



matrix<double> la_pack::inverseDeterminant(matrix<double> &A)
{
	return getCofactor( A ) * ( 1 / determinat( A ));

}



//----------------------------------------------------------------------------------------------------

matrix<double> la_pack::inversePenrose(matrix<double> A)
{

	matrix<double> tempVar = (tranpose(A) * A) ;// to get around g++ error : cannot bind lvaule to const reference
	return inverseGuassJordan( tempVar ) * tranpose(A);
}

matrix<double> la_pack::inversePseudoQR(matrix<double> A)// NOT SURE ABOUT CORRECTNESS
{

	matrix<double> QR = QRFactorizeCol(A);
	matrix<double> tempVarA = (getR_from_QR(QR)) ;// to get around g++ error : cannot bind lvaule to const reference
	matrix<double> tempVarB = ( getQ_from_QR(QR)) ;// to get around g++ error : cannot bind lvaule to const reference
	return inverseGuassJordan(tempVarA) * tranpose(tempVarB);
}

matrix<double> la_pack::ridgeRegression(matrix<double> A, double alpha)
{
	return regularizationTikonon( A,  alpha );
}

matrix<double> la_pack::regularizationTikonon(matrix<double> A, double alpha)
{
	matrix<double> alphaI = getIdentity(A.numCols())*alpha;
	matrix<double> tempVar = ((tranpose(A) * A) - alphaI);// to get around g++ error : cannot bind lvaule to const reference
	return inverseGuassJordan(tempVar)*tranpose(A);
}

/*
	Using the pseudo inverse to find the least square 
	Least Squares is the BEST LINEAR UNBIASED ESTIMATOR 
*/
matrix<double> la_pack::leastSquareSolver(matrix<double> A , matrix<double> B)
{
	return inversePenrose( A ) * B;
}

matrix<double> la_pack::getResidue(matrix<double> A, matrix<double> B)
{
	if (A.isColVector() && B.isColVector())
	{
		matrix<double> R(1, A.numCols());
		R = A - B; 
		return R; 
	}
	else
	{
		error("Not Column Vectors ");
		return A*(0);
	}
}



//-------------------------------------------MATRIX SUM -------------------------------------------------------------------------------//


double la_pack::sum(matrix<double>& rhs)
{
	double sum = 0; 

	if (rhs.isColVector())
	{
		for (long long i = 1; i <= rhs.numRows(); i++) // Go through each row and add them up 
		{
			sum += rhs(i, 1);
		}
	}
	else if (rhs.isRowVector())
	{
		for (long long i = 1; i <= rhs.numCols(); i++) // Go through each row and add them up 
		{
			sum += rhs(1,i);
		}

	}
	else // rhs is a normal matrix , fat or thin 
	{
		for (long long i = 1; i <= rhs.numRows(); i++)
		{
			for (long long j = 1; j <= rhs.numCols(); j++)
			{
				sum += rhs(i, j);
			}
		}
	}
	return sum; 
}

//----------------------------------------------------------------------------------------------------------------------------------------------
// Joining Matices 

matrix<double> la_pack::joinCol(matrix<double> &A, matrix<double> &B)
/*
	Columns can only be joined if the number of rows are equal
*/
{
	if( A.numRows() == B.numRows() )
	{ 
		long long  newColSize = ( A.numCols() + B.numCols() );
		matrix<double> AB( A.numRows() , newColSize ); // Create Matrix
			for(long long  i = 1 ; i <= A.numRows() ; i++)
			{
				for(long long  j = 1 ; j <= newColSize ; j++)
				{
					// When going over columns of A , add col to AB , when going over columns of B add col to AB
					if( j <= A.numCols() )
					{
						AB(i,j) = A(i,j);
					}
					else
					{
						AB(i,j) = B(i,j-A.numCols());
					}
						
				}
			}
			return AB; 
	}else{
		error(" joinCol : Cannot join columns of matrices with unequal row number ");
		return A; // Nothing should be retured // HACK 
	}
}



matrix<double> la_pack::joinRow(matrix<double> &A, matrix<double> &B)
/*
	Rows can only be joined if the number of cols are equal
*/
{
	if(A.numCols() == B.numCols())
	{
		long long  newRowSize = (A.numRows() + B.numRows());
		matrix<double> AB(newRowSize,A.numCols()); // Create Matrix
		for(long long  i = 1 ; i <= newRowSize ;i++)
		{
			for(long long  j = 1 ; j <= A.numCols() ; j++)
			{
					// When going over columns of A , add col to AB , when going over columns of B add col to AB
					if(i <= A.numRows())
					{
						AB(i,j) = A(i,j);
					}
					else
					{
						AB(i,j) = B(i - A.numRows(),j);
					}
				}
			}

		return AB;
	}
	else
	{
		error(" joinRow :: Cannot join rows of matrices with unequal col number ");
		return A; // Nothing should be retured // HACK 
	}
}


//---------------------------------------------------------------------------------------------------------------------------------------
// Splicing Matrices 

matrix<double> la_pack::spliceCopyRows( matrix<double> &rhs, int aRowStart , int aRowEnd )
/*	
	Inclusive from both sides 
*/
{
	long long cols = rhs.numCols(); // keeping the columns same 
	long long rows = aRowEnd - aRowStart; 
	matrix<double> R(rows,cols); 
	
	for(long long i = 1 ; i <= rows; i++)
	{
		for(long long j = 1; j <= cols ; j++)
		{
			R(i,j) = rhs((aRowStart -1 + i ),j);
		}
	}
	return R;
}


matrix<double> la_pack::spliceCopyCols( matrix<double> &rhs, int aColStart , int aColEnd )
/*	
	Inclusive from both sides 
*/
{
	long long rows = rhs.numRows(); // keeping the rows same	
	long long cols = aColEnd - aColStart + 1; // 5 - 1 = 4 , Inclusive , so should be 5 
	matrix<double> R(rows,cols); 
	//std::cout << "DEBUG_2";
	for(long long i = 1 ; i <= rows; i++)
	{
		for(long long j = 1; j <= cols  ; j++)
		{
			
			R(i,j) = rhs(i,(aColStart -1 + j ));
		}
	}
	return R;
}




matrix<double> la_pack::swapRow(matrix<double>& rhs, int firstRow, int secondRow )
{
	// Copy of row 1
	//std::cout << "DEBUG 18" << rhs.numCols() ;
	matrix<double> temp (1,rhs.numCols());
	// Copy row 2 to row 1 
	// Copy temp to row 2
	for(long long j = 1 ; j <= rhs.numCols() ; j++)
	{
		//std::cout << "DEBUG 17" << j ;
		temp(1,j) = rhs(firstRow,j);
		rhs(firstRow,j) = rhs(secondRow,j);
		rhs(secondRow,j) = temp(1,j);
	}
	return rhs;
}

matrix<double> la_pack::swapCol(matrix<double>& rhs, int firstCol, int secondCol )
{
	// Copy ofcol 1
	matrix<double> temp (rhs.numRows(),1);
	// Copy col 2 to col 1 
	// Copy temp to col 2
	for(long long i = 1 ; i <= rhs.numRows() ; i++)
	{
		temp(i,1) = rhs(i,firstCol);
		rhs(i,firstCol) = rhs(i,secondCol);
		rhs(i,secondCol) = temp(i,1);
	}
	return rhs;
}


matrix<double> la_pack::getUnitVec(matrix<double> rhs)
{
	double norm; norm = normEuclidean(rhs);
		rhs = rhs / norm;
	return rhs; 
}

// ----------------------------------------------------------------------------------------------------

matrix<double> la_pack::gaussianElemPartialPivot(matrix<double> &A, matrix<double> &B, bool roundOff)
{
	//Create new matrix AB
	matrix<double> AB = joinCol(A,B);
	//AB.print();


	// Inside out for loop ,go over all the rows , before moving to the next row
	// 
	for(long long  j = 1 ; j <= A.numCols() ;j++ )
	{
		long long  pivotRowPosition = j ; 
		for(long long  i = j ; i <= A.numRows() ; i++)  // the i = j ensures that only lower triangular values are looped over 
		{
			// Gaussian Elemination 
			// Make 1 
			if(i==j)
			{
				double pivot = AB(i,j);
					if(pivot != 1.0) 
					{
						if (pivot != 0 && !unstable(pivot))
						{
							for(long long  k = j ; k <= AB.numCols();k++)
							{
								AB(i,k) = AB(i,k)*(1.0/pivot);
								//std::cout << "DEBUG _ 10";
							}
						}
						else
						{ 
							// To deal with it , interchange the rows so that you get a non zero row 
							// Pivoting 
							// for though column, find the largest element , swap this row and that(large no) row
							
							//std::cout << "DEBUG_ 12";
							double max = 0 ;int validPivot = 0 ; int zeroPivot = i; 
							for(long long m = pivotRowPosition ; m <= AB.numRows() ;m++) // Go down the column 
							{
								if(AB(i,m) > max && !unstable(AB(i,m))) // Checking if the new number is actually good , or just a very small number larger than max
								{
									//std::cout << "DEBUG_ 13" << m << i;
									max = AB(i,m);validPivot = m ; // m will be the row with the largest value in the matrix 
								} 
							}
							if(max != 0 && !unstable(max)) // Ensuing that what we divide by is not zero or a very small number
							{ // we can now swap the pivot row with the row with the max ; 
								//std::cout << "DEBUG_ 15";
								swapRow(AB,zeroPivot,validPivot);
								//std::cout << "DEBUG_ 16";
								// Now non zero Pivot , Reduced Pivot to 1 
								pivot = AB(i,j); // New Pivot Found by Pivoting 
								for(long long  k = j ; k <= AB.numCols();k++)
								{
									//std::cout << "DEBUG_ 14" << i << k << pivot;
									AB(i,k) = AB(i,k)*(1.0/pivot);
								}
							}
							else
							{
								error("Cannot Reduce Using Gaussian Elimination; Non Zero Pivot Not Found :");std::cout << j << i ;
								return A*(0);
							}
						}
					}
					// if pivot = 1 , nothing to be done . 
			}
			else
			{
				double firstInRow = AB(i,j); // firstInRow will be reduced to zero after this loop 
				//(firstInRow == 0) ? firstInRow = 1 : firstInRow = firstInRow;
				for(long long  k = j ; k <= AB.numCols();k++)
				{
					AB(i,k) = AB(i,k) - firstInRow*AB(pivotRowPosition,k);
					// TO get clean zero , if answe < 10^(-15) then we choose 0 ; 
					//if(roundOff && i > j && std::abs(AB(i,k)) < 10^(-15)) AB(i,k) = 0 ;
				}
					
			}
			//AB.print();												//DEBUG
		}
	}
	return AB; 
}


//-----------------------------------------------------------------------------------------------

matrix<double> la_pack::tranpose(matrix<double> &A)
{
	matrix<double> R(A.numCols(),A.numRows()); 
	for(long long  i = 1; i <= A.numRows() ; i++)
	{
		for(long long  j = 1; j <= A.numCols() ; j++)
		{
			R(j,i) = A(i,j);
		}
	}
	return R;
}



//----------------------------------------------------------------------------------------------------------------------------------------




//------------------------------------------------------------------------------------------------
matrix<double> la_pack::outerProduct(matrix<double> &A, matrix<double> &B)
/*
	input : Outper Product 
	ColVec * RowVec
*/
{
	matrix<double> R (A.numRows(),B.numCols());
	if(A.isColVector() && B.isRowVector() && (A.numRows() == B.numCols()))
	{
		for(long long  i = 1 ; i <= R.numRows();i++)
		{
			for(long long  j = 1 ; i <= R.numCols() ;j++)
			{
				R(i,j) = A(i,1)*B(1,j);
			}
		}
		return R;
	}
	else
	{
		error("outerProduct -> A and B are not in proper format ");
		return R*(0);
	}
}


//-----------------------------------------------------------------------------------------------


double la_pack::innerProduct(matrix<double> const &A, matrix<double> const &B)
/*
	inner Product :
	At * B // t = transpose
	Row Vec * Col Vec 
*/
{
	double result = 0 ; 
	if(A.isRowVector() && B.isColVector() && (A.numCols() == B.numRows()))
	{
		for(long long  i = 1 ; i <= A.numRows();i++)
		{	
			result += A(1,i)*B(i,1);
		}
		return result;
	}
	else if(A.isRowVector() && B.isRowVector() && (A.numCols() == B.numCols()))
	{
		for(long long  i = 1 ; i <= A.numCols();i++)
		{	
			result += A(1,i)*B(1,i);
		}
		return result;
	}
	else if(A.isColVector() && B.isColVector() && (A.numRows() == B.numRows()))
	{
		for(long long  i = 1 ; i <= A.numRows();i++)
		{	
			result += A(i,1)*B(i,1);
		}
		return result;
	}
	else
	{
		error("innerProduct -> A and B are not in proper format ");
		return result*(0);
	}
}

//------------------------------------------------------------------------------------------------------------------------------------

matrix<double> la_pack::backSubstitution(matrix<double> &AB)
/*
	Does the hill clmbing required to find X 
	Takes Input as the output of GE ie, an upper Triangular Matrix  
	AB has both A and B 
	A X = B 
	3*5 5*1 = 3*1    X row size = A col size 
*/
{
	matrix<double> X(AB.numCols()-1,1);
	double residue = 0 ; 
	long long  rowEndA = AB.numCols() - 1; 
	for(long long  i = AB.numRows(); i > 0 ;i-- ) // Number Row in AB = A ; 
	{
		residue = 0 ; // Resetting
		for(long long  j = rowEndA; j > 0 ; j--) // Goes through a column till end of A , inclusive // We just want A not AB 
		{
			if( j >= i)
			{
				if(i != j)  // Not At Diagonal // Calculate Residue 
				{
					residue += AB(i,j)*X(j,1); 	
				}
				else // a diagonal , calculate Xj using residue 
				{
					X(j,1) = AB(i,rowEndA + 1) - residue;  // Using rowEndA + 1 to ensure we stay at B 
					 
				}	
			}
			else
			{
				continue;
			}
			//std::cout << i << " " << j << std::endl  ;			                    //DEBUG
		}
		
		//AB.print();																	    //DEBUG
		//X.print();																	     //DEBUG
		//std::cout << residue << std::endl;									           //DEBUG
	}
	return X ; 
}



// ---------------------------------------------------------------------------------------------------------------------------------------

matrix<double> la_pack::inverseGuassJordan(matrix<double> &A)
/*
	Appends I to the right to A 
	Then Convert A to I 
	Take out I 
*/
{
	if(A.isSquare())
	{
		//std::cout << "DEBUG_3";
		matrix<double> I = getIdentity(A.numRows());
		//std::cout << "DEBUG_4";
		matrix<double> R = gaussianElemPartialPivot(A,I);
		//std::cout << "DEBUG_5";
		R = rowEchelonReduceUToI(R);
		//std::cout << "DEBUG_6";
		return spliceCopyCols(R , A.numCols() + 1 , R.numCols()); 
	}
	else
	{
		error("inverseGaussianElemination -> A is not of correct form ");
		return A*(0) ; 
	}
}




matrix<double> la_pack::rowEchelonReduceUToI(matrix<double> &A, bool roundOff) // Does  work 
/*
	doTillRow is inclusive 
	Row Echelon Form for the specific case of AI inverse finding method 
	A more row echelon form can be build from this // Later

	Only call this after , you have done GE on this //ie have an upper trainguler matrix with 
	1 at the pivot position.

	the operation will only be done till doTillRow // 

*/
{ // No checking , checking should be done by callers .. Private Functin 
	//matrix<double> R (A.numRows(),A.numCols());
	for(long long i = 2 ,j = 2 ; i <= A.numRows();i++,j++) // No operation to be done using first pivot
	{  // This loop only goes over the diagonal elements 
				// At pivot , now go from roof of matrix to the pivot , reducing each element above the pivot to zero ; 
				for(int l = 1 ; l < i ; l++ ) // goes from root to pivot
				{ 
					double reductionFactor = A(l,j) ;
					//(reductionFactor == 0) ? reductionFactor = 1 : reductionFactor = reductionFactor;
					for(long long k = j ; k <= A.numCols(); k++)
					{
						//std::cout << "DEBUG_1";
						A(l,k) = A(l,k) - reductionFactor * A(i,k); // A(i+1,j) is the pivot ( j as at i+1 ) 
						//if(roundOff && k <= A.numRows() && std::abs(A(l,k)) < 10^(-15)) A(l,k) = 0 ;
					}		
				}		
	}
		//if(roundOff && std::abs(A(1,2)) < 10^(-15) ) A(1,2) = 0 ;// Hacks ;Corner Case ; but saves cycles 
	return A ; 
}

//------------------------------------------------------------------------------------------------------

double la_pack::trace(matrix<double> &A)
/*
	retuns num of diagonal elements
*/
{
	double trace = 0 ;
		for(long long  i = 1 ; i <= A.numRows() ;i++)
		{
			for(long long  j = 1; j <= A.numCols() ; j++)
			{
					trace += A(i,j);
			}
		}
		return trace; 
}


//-------------------------------------------------------------------------------------------------------------------------

matrix<double> la_pack::LUDecomposePartialPivot(matrix<double> &AL)
{
	matrix<double> A = AL;
	matrix<double> L (A.numRows(),A.numCols());
	matrix<double> U (A.numRows(),A.numCols());

	for(long long  j = 1 ; j <= A.numCols() ;j++ )
	{
		long long  pivotRowPosition = j ; 
		for(long long  i = j ; i <= A.numRows() ; i++)  // the i = j ensures that only lower triangular values are looped over 
		{
			// Gaussian Elemination 
			// Make 1 
			if(i==j)
			{
				double pivot = A(i,j);
					if(pivot != 1.0)
					{
						if (pivot != 0 && !unstable(pivot))
						{
							L(i,j) = pivot ; 
							for(long long  k = j ; k <= A.numRows();k++)
							{
								
								A(i,k) = A(i,k)*(1.0/pivot);
								U(i,k) = A(i,k);
								// Filling in the U part 
							}
						}
						else
						{ 
							// To deal with it , interchange the rows so that you get a non zero row 
							// Pivoting 
							// for though column, find the largest element , swap this row and that(large no) row
							
							double max = 0 ;int validPivot = 0 ; int zeroPivot = i; 
							for(long long m = pivotRowPosition ; m <= A.numCols() ;m++) // Go down the column 
							{
								if(A(i,m) > max)
								{
									max = A(i,m);validPivot = m ; // m will be the row with the largest value in the matrix 
								} 
							}
							if(max != 0)
							{ // we can now swap the pivot row with the row with the max ; 
								swapRow(A,zeroPivot,validPivot);
								// Now non zero Pivot , Reduced Pivot to 1 
								pivot = A(i,j); // New Pivot Found by Pivoting 
								for(long long  k = j ; k <= A.numCols();k++)
								{									
									A(i,k) = A(i,k)*(1.0/pivot);
									U(i,k) = A(i,k);
								}
							}
							else
							{
								error("Cannot Reduce Using LU Decomposition ; Non Zero Pivot Not Found :");std::cout << j << i ;
								return A*(0); 
							}
						}
					}
			}
			else
			{
				double firstInRow = A(i,j); // firstInRow will be reduced to zero after this loop
				L(i,j) = firstInRow;
				//(firstInRow == 0) ? firstInRow = 1 : firstInRow = firstInRow;
				for(long long  k = j ; k <= A.numCols();k++)
				{
					A(i,k) = A(i,k) - firstInRow*A(pivotRowPosition,k);
					U(i, k) = A(i, k); // Copying A to U . 
					// TO get clean zero , if answe < 10^(-15) then we choose 0 ; 
					//if(roundOff && i > j && std::abs(AB(i,k)) < 10^(-15)) AB(i,k) = 0 ;
				}
					
			}
			//A.print();												//DEBUG
		}
	}

	//Preparing L and U 
	for (long long i = 1, j = 1; j <= L.numRows(); i++, j++)
	{

		if (i == j)
		{
			double multiple = L(i, j);
			U(i, j) = U(i, j)*multiple;
			L(i, j) = L(i, j) / multiple; // Make diagonal elements 1 ; 

		}


	}
	return joinCol(L,U);
}


matrix<double> la_pack::getL_from_LU(matrix<double> &LU)
{

	return spliceCopyCols(LU, 1, LU.numCols() / 2);

}
matrix<double> la_pack::getU_from_LU(matrix<double> &LU)
{

	return spliceCopyCols(LU, LU.numCols() / 2 + 1, LU.numCols());

}


//       LU DECOMPUTSITON :: Better ? From ART OF SCIENTIFIC PROGRAMMING 

// ---------------------------------------------------------------------------------------------------------------------------------


double la_pack::normEuclidean(matrix<double> &A)
/*
	Returns Euclidean or L2 norm of a matrix 
*/ 
{
	double innerProd = innerProduct(A,A);
	return std::sqrt(innerProd);
}

double la_pack::normChebeshev(matrix<double> &A)
{
	return normLInf(A);
}

// Takes only column matrix 
double la_pack::normL_P(matrix<double> &A, double p)
{
	if (!A.isColVector())
	{
		error("la_pack -> normL_P -> Input is not Column Vector ");
		return 0; 
	} 
	else
	{
		double result = 0; 
		for (int i = 1; i < A.numRows(); i++) // GO though each element applying the norm 
		{
			result += std::pow(A(i, 1), p);
		}
		result = std::pow(result, 1.0 / p);
		return result; 
	}
}


double la_pack::normWeight_P(matrix<double> &A, matrix<double> Weight, double p)
{
	if (!A.isColVector()  )
	{
		error("la_pack -> normL_P -> Input is not Column Vector ");
		if (!Weight.isSquare())
		{
			error("la_pack -> normL_P -> Weight is not a square Matrix");
		}
		return 0;
	}
	else
	{
		matrix<double> R(A.numRows(), 1);
		// Create a new vector having the weight multiplied by the 

		R = Weight * A;

		double result = 0;
		for (int i = 1; i < R.numRows(); i++) // GO though each element applying the norm 
		{
			result += std::pow(R(i, 1), p);
		}
		result = std::pow(result, 1.0 / p);
		return result;
	}
}

double la_pack::normL1(matrix<double> &A)
/*
	Returns L1 norm of a matrix 
*/ 
{
	bool col = false ; 
	long long size = 0 ;
	double result = 0 ; 
	if(A.isColVector())
	{
		col = true ;
		size = A.numRows();
	}
	else
	{
		col = false ; 
		size = A.numCols();
	}
	for(int i = 1; i <= size ;i++)
	{
		if(col)
		{
			result += std::abs(A(i,1)); 
		}
		else
		{
			result += std::abs(A(1,i)); 
		}
	}
	return result ; 
}

double la_pack::normLInf(matrix<double> &A) // Chebeshev Norm 
/*
	Returns LInf norm of a matrix 
*/ 
{
	bool col = false ; 
	long long size = 0 ;
	double result = 0 ; 
	if(A.isColVector())
	{
		col = true ;
		size = A.numRows();
	}
	else
	{
		col = false ; 
		size = A.numCols();
	}
	for(int i = 1; i <= size ;i++)
	{
		if(col)
		{
			double temp = std::abs(A(i,1));
			if( temp > result)
			{
				result = temp ; 
			}
		}
		else
		{
			double temp = std::abs(A(1,i));
			if( temp > result)
			{
				result = temp ; 
			}
		}
	}
	return result ; 
}


matrix<double> la_pack::projectionOn(matrix<double> &A, matrix<double> &B)
{
	double upper = innerProduct(A,B);
	double lower = innerProduct(B,B);
	return (B*(upper/lower));

}

matrix<double> la_pack::projectionOn_Ortho(matrix<double> &A, matrix<double> &B)
{
	return B.operator-(projectionOn(A, B));
}


//--------------------------------------------------------------------------------------------------------------------------


matrix<double> la_pack::solverGussianElem(matrix<double> &A, matrix<double> &B)
{
	matrix<double>ABReduced = gaussianElemPartialPivot(A,B);
	return backSubstitution(ABReduced);

}

matrix<double> la_pack::solverLU(matrix<double> &A, matrix<double> &B)
{   // Not correct 
	
	return A;



}

double la_pack::corelationCoeff(matrix<double> &A, matrix<double> &B)
{
	double upper = innerProduct(A,B);
	double lower = normEuclidean(A)*normEuclidean(B);
	return upper/lower;
}

double la_pack::angle(matrix<double> &A, matrix<double> &B)
{
	return std::acos(corelationCoeff(A,B));
}

bool la_pack::isOrthogonalVectors(matrix<double> &A, matrix<double> &B)
{
	if(innerProduct(A,B) == 0) 
	{
		return true ;
	} 
	else 
	{
		return false; 
	}
}


bool la_pack::isSymetric(matrix<double> & Q)
{
	return  Q.isEqual(tranpose(Q));
}


bool la_pack::isPositiveDefinite(matrix<double> &Q)
{
	return isSymetric(Q) && Q.isDiagonallyDominant();
}

matrix<double> la_pack::gramSchmidtOrtho(matrix<double> A,bool normalize)
{
	matrix<double> R = A;
	//R.print();																//DEBUG
	for(int i = 1, j = 1 ; i <= A.numRows() ;i++,j++)  // No operation to be done on row 1 , We make over rows orthogonal to row 1
	{
		matrix<double> residue (1,A.numCols());
		for(int q = j ; q >= 1 ; q--) // Moves from current row to top
		{
			if(q!=i)
			{
				matrix<double> ARow1 = A.removeRow(i);
				matrix<double> ARow2 = A.removeRow(i);
				residue = residue + projectionOn(ARow1,ARow2);   ///DEBUG
			}														
		}
		for(long long k = 1; k <= A.numCols() ; k++)
		{
			R(i,k) = R(i,k) - residue(1,k);
			if(normalize)
			{
				//R(i,k) = R(i,k)*(1/normEuclidean(A.returnRow(i)));
				
				matrix<double> ARow1 = A.removeRow(i);
					
				R(i,k) = R(i,k)*(1/normEuclidean(ARow1));
			}
		}
		//R.print();													//DEBUG
	}
	return R;
}



std::tuple < int, matrix<double> >  la_pack::getRank(matrix<double> A, bool normalize) // NOT IMPLEMENTED !! 
{
	matrix<double> R = A; int rank = 0; 
	//R.print();																//DEBUG
	for (int i = 1, j = 1; i <= A.numRows(); i++, j++)  // No operation to be done on row 1 , We make over rows orthogonal to row 1
	{ // Goes Through Each Column ... 
		matrix<double> residue(1, A.numCols());
		for (int q = j; q >= 1; q--) // Moves from current row to top
		{
			if (q != i)
			{

				matrix<double> ARow1 = A.removeRow(i);
				matrix<double> ARow2 = A.removeRow(q);
				residue = residue + projectionOn(ARow1, ARow2);   ///DEBUG
			}
		}
		matrix<double> temp = R.returnRow(i);
		temp = temp - residue.returnRow(1);
		double tempNorm = normEuclidean(temp);
		if (tempNorm != 0) // This means that columns i is not a LC of the previous columns..
		{
			rank++;
			for (long long k = 1; k <= A.numCols(); k++)
			{
				if (normalize)
				{
					R(i, k) = R(i, k)*(1 / tempNorm);
				}
			}
		}
		//R.print();													//DEBUG
	}
	std::tuple < int, matrix<double> >  result;
	result = std::make_tuple(rank, R);
	/*
	Store the rank and matrix in result and then return the result !

	std::get<0>(result) ; // First Value 
	std::get<1>(result) ; // Second Value 
	*/
	return result;
}



matrix<double> la_pack::projectionMatrix(matrix<double> A)
{
	return A*inversePenrose(A);
}


// A Vector is represented along a Row
matrix<double> la_pack::QRFactorizeRow(matrix<double> A, bool normalize)// NOT SURE ABOUT CORRECTNESS
{
	matrix<double> Q = A;
	matrix<double> R = A*(0);
	double upper, lower, multiplier; 
	//R.print();																//DEBUG
	for (int i = 1; i <= A.numRows(); i++)  // No operation to be done on row 1 , We make over rows orthogonal to row 1
	{
		matrix<double> residue(1, A.numCols());
		for (int q = i; q >= 1; q--) // Moves from current row to top
		{
			// Map row 1 to Column 1 
			// Map row 2 to Column 2
			// If row 3 had components 4 and 5 with row 1 and row 2 
			// We represent it as Q(1,3) = 4 ; Q(2,3) = 5 ; 
			if (q != i)
			{

				matrix<double> ARow1 = A.removeRow(i);
				matrix<double> ARow2 = A.removeRow(q);
	

				//upper = innerProduct(A.returnRow(i), A.returnRow(q));
				upper = innerProduct(ARow1, ARow2);
				//lower = innerProduct(A.returnRow(q), A.returnRow(q));
				lower = innerProduct(ARow2, ARow2);
				multiplier = upper / lower; 
				R(q,i) = multiplier;
				//residue = residue + A.returnRow(q)*multiplier;
				residue = residue + ARow2*multiplier;
			}
		}

		for (long long k = 1; k <= A.numCols(); k++)
		{
			Q(i, k) = Q(i, k) - residue(1, k);
			if (normalize)
			{

				matrix<double> ARow1 = A.removeRow(i);
				//Q(i, k) = Q(i, k)*(1 / normEuclidean(A.returnRow(i)));
				Q(i, k) = Q(i, k)*(1 / normEuclidean(ARow1));
			}
		}
		//R.print();									//DEBUG

		if (normalize)
		{
			matrix<double> unit = getUnitVec(A.returnRow(i));
			matrix<double> ARow1 = A.removeRow(i);
			//R(i, i) = innerProduct(A.returnRow(i), unit);
			R(i, i) = innerProduct(ARow1, unit);
		}
		else
		{
			R(i, i) = 1;

		}
	}

	return joinCol(Q,R); 
}





matrix<double> la_pack::QRFactorizeCol(matrix<double> A, bool normalize) // NOT SURE ABOUT CORRECTNESS
{
	matrix<double> Q = A;
	matrix<double> R = A*(0);
	double upper, lower, multiplier;
	//R.print();																//DEBUG
	for (int i = 1; i <= A.numCols(); i++)  // No operation to be done on COl 1 , We make over cols orthogonal to col 1
	{
		matrix<double> residue(A.numRows(), 1);
		for (int q = i; q >= 1; q--) // Moves from current Col to first
		{
			// Map row 1 to Column 1 
			// Map row 2 to Column 2
			// If row 3 had components 4 and 5 with row 1 and row 2 
			// We represent it as Q(1,3) = 4 ; Q(2,3) = 5 ; 
			if (q != i)
			{


				matrix<double> ARow1 = A.removeRow(i);
				matrix<double> ARow2 = A.removeRow(q);
	

				//upper = innerProduct(A.returnRow(i), A.returnRow(q));
				upper = innerProduct(ARow1, ARow2);
				//lower = innerProduct(A.returnRow(q), A.returnRow(q));
				lower = innerProduct(ARow2, ARow2);
		
				multiplier = upper / lower;
				R(q,i) = multiplier;
				//residue = residue + A.returnCol(q)*multiplier;
				residue = residue + ARow2*multiplier;
			}
		}

		for (long long k = 1; k <= A.numRows(); k++)
		{
			Q(i, k) = Q(i, k) - residue(k,1);
			if (normalize)
			{

				matrix<double> ARow1 = A.removeRow(i);
				Q(i, k) = Q(i, k)*(1 / normEuclidean(ARow1));
				//Q(i, k) = Q(i, k)*(1 / normEuclidean(A.returnCol(i)));
			}
		}
		//Q.print();									//DEBUG


		if (normalize)
		{
			matrix<double> unit = getUnitVec(A.returnCol(i));
			matrix<double> ARow1 = A.removeRow(i);
			//R(i, i) = innerProduct(A.returnRow(i), unit);
			R(i, i) = innerProduct(ARow1, unit);
			//R(i, i) = R(i, i) / normEuclidean(A.returnCol(i));

		}
		else
		{
			R(i, i) = 1; 

		}
	}

	return joinCol(Q, R);
}


matrix<double> la_pack::getQ_from_QR(matrix<double> QR)
{
	return spliceCopyCols(QR, 1, QR.numCols() / 2);
}
matrix<double> la_pack::getR_from_QR(matrix<double> QR)
{
	return spliceCopyCols(QR, QR.numCols() / 2 + 1, QR.numCols());
}


double la_pack::normForbenius(matrix<double> &A)
/*
Create a column and takes its norm 
*/
{
	matrix<double> unpacked (A.numRows()*A.numCols(),1) ;// Column vector
	for(long long i = 1 ; i < A.numRows();i++)
	{
		for(long long j = 1 ; j < A.numCols() ; j++)
		{
			unpacked(i + j,1) = A(i,j);
		}
	}
	return normEuclidean(unpacked);
}




matrix<double> la_pack::mapMatrixdouble(matrix<double>& rhs, std::function<double(double)> fn)
{
	if (rhs.isColVector()) // Column Vec
	{
		matrix<double> R(rhs.numRows(), 1);
		for (int i = 1; i <= rhs.numRows(); i++)
		{
			R(i, 1) = fn(rhs(i, 1));

		}
		return R;

	}
	else// Row Vec
	{
		matrix<double> R(1, rhs.numCols());
		for (int i = 1; i <= rhs.numCols(); i++)
		{
			R(1, i) = fn(rhs(1, i));

		}
		return R;
	}
}

matrix<double> la_pack::createBlockColVec_2(matrix<double> A, matrix<double> B)
{
	if (A.numCols() == B.numCols())
	{
		int Arows = A.numRows(), Acols = A.numCols();
		matrix<double> result(A.numRows() + B.numRows(), A.numCols());
		for (long long i = 1; i <= result.numRows(); i++)
		{
			for (long long k = 1; k <= result.numCols(); k++)
			{
				// First Copy from A , on reaching the last Row of A , switch to B and copy from there to result
				if (i <= A.numRows())
				{
					result(i, k) = A(i, k); 
				}
				else
				{
					
					result(i, k) = B(i - Arows , k); // i has gone over A that , to remove that the - Arows. 
				}
				
			}
		}
		return result; 
	}
	else
	{
		error("createColVecBlock_2Col -> A and B are have different Column Size :: Cannot Join Together ");
		return A * 0; 
	}
}

matrix<double> la_pack::createBlockRowVec_2(matrix<double> A, matrix<double> B)
{
	if (A.numRows() == B.numRows())
	{
		int Arows = A.numRows(), Acols = A.numCols();
		matrix<double> result(A.numRows(), A.numCols() + B.numCols());
		for (long long i = 1; i <= result.numRows(); i++)
		{
			for (long long k = 1; k <= result.numCols(); k++)
			{
				// Go through each row . Go through A first , or reaching end of A cols move to B cols 
				if (k <= Acols)
				{
					result(i, k) = A(i, k);
				}
				else
				{
					result(i, k) = B(i, k - Acols); // k has gone over A , to remove that the - Acols. 
				}

			}
		}
		return result;
	}
	else
	{
		error("createColVecBlock_2Col -> A and B are have different Row Size :: Cannot Join Together ");
		return A * 0;
	}
}

matrix<double> la_pack::createBlockSquare_4(matrix<double> A, matrix<double> B, matrix<double> C, matrix<double> D)
{
	// ERROR HANDLING 
	if (A.numCols() != B.numCols())
	{
		error("createColVecBlock_2Col -> A and B are have different Column Size :: Cannot Join Together ");
		return A * 0;
	}
	else if (C.numCols() != D.numCols())
	{
		error("createColVecBlock_2Col -> C and D are have different Column Size :: Cannot Join Together ");
		return A * 0;
	}
	else if (A.numRows() != C.numRows())
	{
		error("createColVecBlock_2Col -> A and C are have different Row Size :: Cannot Join Together ");
		return A * 0;
	}
	else if (B.numRows() != D.numRows())
	{
		error("createColVecBlock_2Col -> B and D are have different Row Size :: Cannot Join Together ");
		return A * 0;
	}
	else
	{

		matrix<double> result(A.numRows() + B.numRows(), A.numCols() + C.numCols());
		int Arows = A.numRows(), Acols = A.numCols();
		for (long long i = 1; i <= result.numRows(); i++)
		{
			for (long long k = 1; k <= result.numCols(); k++)
			{
				// First Copy the A and C , then go on to copy the B and D 
				if (i <= Arows)
				{
					// Copying of A and C happens at same time 
					// We go over the cols of A then go over the cols of C , while the cols of result is continuous
					// We copy the cols of A , then C on to result and then go to next row
					if (k <= Acols)  // INEFFICIENT , switching between A and C will lead to reduction in cache hits 
					{
						result(i, k) = A(i, k);
					}
					else
					{
						
						result(i, k) = C(i, k - Acols);
					}
				}
				else
				{
					
					if (k <= Acols) // A.cols = B.cols ; // Condition 
					{

						result(i, k) = B(i - Arows, k); // To reach the index of B
					}
					else
					{
						result(i, k) = D(i - Arows, k - Acols);
					}
				}
			}
		}
		return result;
	}
	
}
//-----------------------------------------------ITERATIVE METHODS --------------------------------------------------//


matrix<double> la_pack::jacobiSolver(matrix<double> &A, matrix<double> &B )
/*
	AX = b  => X rows equal to A columns 
	Initil guess is all zeros . 
*/
{
	 // We take the relaive error and try to figure. 
	matrix<double> X(A.numCols(),1);

	double oldValue; 
	double counter = _iterations;
	double maxChange = 0; // max used to check if convergence reached ! 
	// GUESS FOR INITIAL X(1,1);
	//X(1, 1) = guess; 
		while (counter > 0)
		{
			counter--; 
		
			for (long long i = 1; i <= X.numRows(); i++)
			{
				double jacobiSub = 0; // NEW ROW FLUSH VALUE 

				for (long long j = 1; j <= X.numRows(); j++)
				{
					if (i != j)
					{
						jacobiSub += (A(i, j) * X(j, 1));
					}
				}

				oldValue = X(i, 1); // TO CHECK CONVERGENCE

				X(i, 1) = ( B(i, 1) - jacobiSub ) / A(i, i);

	//			X.print();											//DEBUG
				//std::cout << jacobiSub << std::endl;			//DEBUG
				

				if (maxChange < std::abs((oldValue - X(i, 1))))
				{
					maxChange = std::abs((oldValue - X(i, 1))) ; // Stores the largest value of change ; 
				}
				//std::cout << "KKKK" << std::endl;
			}

			if (isInfinitesimal(maxChange)) // Maximum change in values is very small - > Convergence
			{
				break; 
			}
			//break;
		}
	
	return X; 
}


matrix<double> la_pack::gaussSidelSolver(matrix<double> &A, matrix<double> &B)
/*
	We try to leverage the values of X that we computed for better calculaion
*/
{
	matrix<double> X(A.numCols(), 1);

	double Sub_1  = 0;
	double Sub_2  = 0;

	double oldValue;
	double counter = _iterations;
	double maxChange = 0; // max used to check if convergence reached ! 
	while (counter > 0)
	{
		counter--;
		Sub_1 = 0, Sub_2 = 0; // Values returned to Zero State 
		for (long long i = 1; i <= X.numRows(); i++)
		{
			
			for (long long j = 1; j <= i - 1; j++)
			{
				Sub_1 += A(i, j) * X(j, 1);
			}

			oldValue = X(i, 1);
			X(i, 1) = (B(i, 1) - (Sub_1 + Sub_2)) / A(i, i);

	//		X.print();																			///DEBUG
			for (long long j = i; j <= X.numRows(); j++)
			{
				Sub_2 += A(i, j) * X(j, 1);
			}

			if (maxChange < std::abs((oldValue - X(i, 1))))
			{
				maxChange = std::abs((oldValue - X(i, 1))); // Stores the largest value of change ; 
			}
			//std::cout << counter; 
		}

		if (isInfinitesimal(maxChange)) // Maximum change in values is very small - > Convergence
		{
			//std::cout << "HELLO " << std::endl; 
			break;

		}
	}
	return X;
}


matrix<double> la_pack::choleskyFactorize(matrix<double> A)
{
	matrix<double> L(A.numRows(), A.numCols());
	double value = 0; double pivot = 0;

	for (long long j = 1; j <= A.numCols(); j++)
	{
		for (long long i = 1; i <= A.numRows(); i++)
		{
			if (j == 1)
			{
				if (i == j)
				{
					
					L(i, j) = std::sqrt(A(i, j));
					pivot = L(i, j);
				}
				else if (i >= j) // Moving Down the Diagonal , along the Column 
				{
					L(i, j) = A(i, j) *(1.0 / pivot);
				}
			}
			else
			{
				if (i == j)
				{
					double prev = 0;
					for (int k = i; k > 0; k--)
					{
						prev += L(i, k)*L(i,k);
					}
					value = A(i, j) - prev;
					pivot = i;
					L(i, j) = std::sqrt(value);
					value = 0; 
				}
				else if (i >= j) // Moving Down the Diagonal , along the Column 
				{
					double prev2 = 0;
					for (int k = i; k > 0; k--)
					{
						prev2 += L(i, k)*L(i, k);
					}
					value = A(i, j) - prev2;
					L(i, j) = value;
					value = 0;
				}
			}
//			L.print();													//DEBUG

		}
	}
	
	return L; 
}


//-----------------------------------------SINGLUAR VALUE DECOMPOSITION -------------------------------------------//

/*
	mymathlib matrices linearsystems svd
	// Householder Reduction to Bidiagonal Form 
	// Givens Reduction to Diagonal Form 
	// Sort By Decreasing Eigenvalues 
	 Step 1  A -> U1 B V1  || B -> Bidiagonal || U1 and V1 obtained by HouseHolder Tranfrom
	 Step 2  B -> U2 D V2  || Givens Tranformations 
	 Step 3 U = U2 * U1 and V = V2 * V1 || sort D in decreasing order of singular values. 

*/
//std::tuple< matrix<double>, matrix<double>, matrix<double>> SVD(matrix<double> A)



//------------------------------------------------EIGENVALUES & EIGENVECTORS ----------------------------------------//


matrix<double> la_pack::eigenPower(matrix<double> A , bool rayleigh)
{
	int iterations = 5;//_iterations /1000;

	matrix<double> R( A.numRows(), 1 ) ;
	matrix<double> eigenVec(A.numRows(), 1); // EigenVEctor Storage , COlumn Vector 
	//A.print();						//DEBUG
	double eigenVal = 0;
	eigenVec.randFill();
	for ( int n = 1; n <= A.numRows(); n++ )
	{
		for ( int i = 1; i < iterations; i++ )
		{
			eigenVec = A * eigenVec;
			if (rayleigh)
			{
				eigenVal = rayleighQuotient(A, eigenVec);
			}
			else
			{
				eigenVal = normLInf(eigenVec);
			}
			eigenVec = eigenVec / eigenVal;
			//x.print();							//DEBUG
			//std::cout << eigenVal << std::endl;			//DEBUG
		}

		R = joinCol(R, eigenVec);
		R(n, 1) = eigenVal;
	}

	return R; 
}


matrix<double> la_pack::eigenPowerInverse(matrix<double> A, bool rayleigh)
{
	int iterations = 5;//_iterations /1000;

	matrix<double> R(A.numRows(), 1);
	matrix<double> eigenVec(A.numRows(), 1); // EigenVEctor Storage , COlumn Vector 
	//A.print();						//DEBUG
	double eigenVal = 0;
	eigenVec.randFill();
	A = inverseGuassJordan(A);

	for (int n = 1; n <= A.numRows(); n++)
	{
		for (int i = 1; i < iterations; i++)
		{
			eigenVec = A * eigenVec;
			if (rayleigh)
			{
				eigenVal = rayleighQuotient(A, eigenVec);
			}
			else
			{
				eigenVal = normLInf(eigenVec);
			}
			eigenVec = eigenVec / eigenVal;
			//x.print();							//DEBUG
			//std::cout << eigenVal << std::endl;			//DEBUG
		}

		R = joinCol(R, eigenVec);
		R(n, 1) = eigenVal;
	}

	return R;

}



matrix<double> la_pack::eigenHotellingDeflation(matrix<double> A, bool rayleigh )
{
	int iterations =  _iterations / 1000;

	matrix<double> R(A.numRows() , 1);
	matrix<double> eigenVec(R.numRows() , 1); // EigenVEctor Storage , COlumn Vector 
	//A.print();						//DEBUG
	double eigenVal = 0;
	eigenVec.randFill();


	for (int n = 1; n <= A.numRows() + 10 ; n++)
	{
		for (int i = 1; i < iterations; i++)
		{
			eigenVec = A * eigenVec;
			if (rayleigh)
			{
				eigenVal = rayleighQuotient(A, eigenVec);
			}
			else
			{
				eigenVal = normLInf(eigenVec);
			}
			eigenVec = eigenVec / eigenVal;
			//eigenVec.print();							//DEBUG
			//std::cout << eigenVal << std::endl;			//DEBUG
		}

		R = joinCol(R, eigenVec);
		R(n, 1) = eigenVal;


		// Now we replace the found eigenvalues by zero , in A so that next smaller one can be found.
		//std::cout << "DEBUG" << std::endl;
		A = A - (eigenVec * tranpose(eigenVec))* eigenVal;
		A.print();
		// The new A has same eigenvales , except the largest one has been replaced by zero.
		// Thus we can use power method to find the next biggest . 
	}

	return R;

}


double la_pack::rayleighQuotient(matrix<double> A, matrix<double> X)
{
	return innerProduct(A*X, X)/ innerProduct(X,X);
}




matrix<double> la_pack::eigenQR(matrix<double> A)
{
	double iters = _iterations ;

	// Initialization .  Nothing more . 
	matrix<double> QR = QRFactorizeCol(A);
	matrix<double> Q = getQ_from_QR(QR);
	matrix<double> R = getR_from_QR(QR);
	A = R*Q;

	for (int i = 0; i < iters; i++)
	{
		QR = QRFactorizeCol(A);
		Q = getQ_from_QR(QR);
		R = getR_from_QR(QR);
		A = R*Q; 
	}
	return joinCol(A, Q);
}






//-----------------------------------------TRANSFORMATION --------------------------------------------//


matrix<double> la_pack::houseHolderTransformation(matrix<double> U)
{
	if (U.isColVector())
	{
		matrix<double> unit = getUnitVec(U);
		matrix<double> I = getIdentity(U.numRows());
		matrix<double> H = I - unit*tranpose(unit) * 2; 

		return H;
	}
	else
	{

		error("houseHolderTransformation -> Not a Column Vector ");
		return U*(0);
	}
}


matrix<double> la_pack::houseHolderTransformation_Scale(matrix<double> U)
{
	double H = (1 / 2)*innerProduct(U, U);
	matrix<double> I = getIdentity(U.numRows());
	matrix<double> P = I - ( U * tranpose(U) )/ H;
	return P;
}


matrix<double> la_pack::houseHolderTransformationX(matrix<double> U, matrix<double> X)
{
	if (U.isColVector())
	{
		matrix<double> unit = getUnitVec(U);
		matrix<double> I = getIdentity(U.numRows());
		matrix<double> H = X - unit*tranpose(unit) * 2 * X ;

		return H;
	}
	else
	{

		error("houseHolderTransformation -> Not a Column Vector ");
		return U*(0);
	}
}


matrix<double>  la_pack::strech_shrink(matrix<double> U, double alpha)
{
	
	matrix<double> I = getIdentity(U.numRows());
	I = I * alpha;

	return I * U; 
}





//------------------------------------------------------POLYNOMIAL ------------------------------------------------------------------------//


// Coefficient Matrix should be column Vector 
double la_pack::polynomial(matrix<double> Coefficient, double x) // Proof By Inspection 
{
	if (!Coefficient.isColVector())
	{
		error("la_pack -> polyMatrix -> Coefficient is not Column Vector ");
		return 0; 
	}
	else{
		double result = 0;
		for (int i = 1; i <= Coefficient.numRows(); i++)
		{
			result =+ Coefficient(1, i) * std::pow(x, i - 1);
		}
		return result;
	}
}

matrix<double> la_pack::vandermondeMatrix(matrix<double> X)
{
	int rows = X.numRows();
	matrix<double> M(rows + 1, rows + 1);
	if (!X.isColVector())
	{
		error("la_pack -> polyMatrix -> X is not Column Vector ");
		return M*0;
	}
	else
	{
		/*
			one row will represent a single polynomial with a single X value  
		*/
		
		for (int i = 1; i <= M.numRows(); i++)
		{
			for (int j = 1; j <= M.numCols(); j++)
			{
				if (j == 1)
				{
					M(i, j) = 1;
				}
				else
				{
					M(i, j) = std::pow(X(1, i), j - 1);
				}
				
			}
		}
		return M; 
	}
}

// Coefficient Matrix should be column Vector 
matrix<double> la_pack::polynomialMatrix(matrix<double> Coefficient, matrix<double> vandermondeMatrix)
{
	if (!Coefficient.isColVector())
	{
		error("la_pack -> polynomialMatrix -> Coefficient is not Column Vector ");
		return Coefficient*0;
	}
	else if (vandermondeMatrix.numCols() != Coefficient.numRows() )
	{
		error("la_pack -> polynomialMatrix ->  vandermondeMatrix.numCols() != Coefficient.numRows() ");
		return Coefficient * 0;
	}
	else
	{
		matrix<double> polynomialMatrix = vandermondeMatrix * Coefficient; 
		return polynomialMatrix;
	}
}



//---------------------------------------------------ART OF SCIENTIIC PROGRAMMING------------------------------------------------------------------//
// Starting Fresh . They will have better techniques . Eat your Ego . 



void la_pack::gaussJordan(matrix<double> &A, matrix<double> &B)
{
	int iCol, iRow;
	int iter = 0; int iter2 = 0;
	int nARows = A.numRows(); int nACols = nARows; // A is SQUARE !
	int nBCols = B.numCols();
	double largest, temp , pivotInverse; 
	// Rows used for Pivoting Bookeeping
	matrix<int> indexRow(nARows, 1);
	matrix<int> indexCol(nARows, 1);
	matrix<int> indexPivot(nARows, 1); 
	int pivotRow, pivotCol; // Position of pivot , which is the largest element in a column
	for (long long i = 1; i <= nARows; i++) // Main Loop Over Columns 
	{
		largest = 0; 
		for (long long j = 1; j <= nARows; ++j) // Main Loop Searching for Pivot
		{
			if (indexPivot(j, 1) != 1)
			{
				for (long long k = 1; k <= nACols; ++k)
				{
					if (indexPivot(k, 1) == 0)
					{
						if (abs(A(j, k)) >= largest)
						{
							largest = abs(A(j, k)); // Found Pivot 
							pivotRow = j; 
							pivotCol = k;
						}
					}
				}
			}
		}
		indexPivot(pivotCol, 1)++; 

		if (pivotRow != pivotCol)
		{	
			 
			for (iter = 1; iter < nARows; ++iter) A.swap(pivotRow, iter, pivotCol, iter);
			for (iter = 1; iter <= nBCols; ++iter) B.swap(pivotRow, iter, pivotCol, iter);
		}
		indexRow(i, 1) = pivotRow;
		indexCol(i, 1) = pivotCol;
		if (A(pivotCol, pivotCol) == 0) error(" guassJordan -> Singular Matrix");
		pivotInverse = 1.0 / (A(pivotCol, pivotCol));
		A(pivotCol, pivotCol) = 1.0; 
		for (iter = 1; iter <= nARows; ++iter) A(pivotCol, iter) *= pivotInverse;
		for (iter = 1; iter <= nBCols; ++iter) B(pivotCol, iter) *= pivotInverse;

		for (int iter = 1; iter <= nARows; ++iter)
		{
			if (iter != pivotCol)
			{
				temp = A(iter, pivotCol);
				A(iter, pivotCol) = 0; 
				for (iter2 = 1; iter2 <= nARows; iter2++) A(iter, iter2) -= A(pivotCol, iter2) * temp;
				for (iter2 = 1; iter2 <= nBCols; iter2++) B(iter, iter2) -= B(pivotCol, iter2) * temp;
			}

		}

		// Ordering the matrix in the proper form 
		for (iter = nARows; iter >= 1; --iter)
		{
			if (indexRow(iter, 1) != indexCol(iter, 1))
			{
				for (iter2 = 1; iter2 <= nARows; ++iter2)
					A.swap(iter2, indexRow(iter, 1), iter2, indexCol(iter, 1));
			}
		}
	}// MAIN FOR LOOP

}

void la_pack::gaussJordan(matrix<double> &A)
{
	matrix<double> B(A.numRows(), 0);
	gaussJordan(A, B);
}


matrix <double> la_pack::decomposeLU(matrix<double> &A)
{
	int nARows = A.numRows(); int nACols = A.numCols();
	matrix<double> joinedLU = A;// The matrix will have both the L and U combined .  
	matrix<int> pivotIndex(A.numRows(),1);// Stores the pivot permutations
	matrix<double> implicitScaling(A.numRows(),1);
	double largest = 0, temp = 0; // Used for scaling 
	int oddEven = 1; // Odd or Even number of matrix permutations 
	int imax = 0; 

	for (int i = 1; i <= nARows; ++i)  // LOOP OVER ROWS FOR IMPLICIT SCALING 
	{
		largest = 0; 
		for (int j = 1; j <= nARows; ++j)
		{
			if ((temp = abs(joinedLU(i, j))) > largest) largest = temp;
		}
		if (largest == 0) error(" LU -> Singular Matrix Input");
		implicitScaling(i, 1) = 1.0 / largest;
	}
	for (int k = 1; k <= nARows; ++k) // kij loop
	{
		largest = 0;
		imax = k;
		for (int i = k; i <= nARows; ++i)
		{
			temp = implicitScaling(i, 1) * abs(joinedLU(i, k));
			if (temp > largest)
			{
				largest = temp; 
				imax = i; // Find the column containing the largest element ? 
			}
		}
		if (k != imax)
		{
			for (int j = 1; j <= nARows; ++j)
			{
				joinedLU.swap(imax, j, k, j); // Interchange the row with small numbers , for the row with larger numbers
			}
			oddEven = -oddEven; // One interchange hence becomes odd 
			implicitScaling(imax, 1) = implicitScaling(k, 1);
		}
		pivotIndex(k, 1) = imax; 
		if (joinedLU(k, k) == 0) joinedLU(k, k) = _tolerance;
		for (int i = k + 1; i <= nARows; ++i)
		{
			temp = joinedLU(i, k);
			temp /= joinedLU(k, k);
			for (int j = k + 1; j <= nARows;  ++j)
				joinedLU(i, j) -= temp * joinedLU(k, j);
		}
	}
	return joinedLU;
}


std::tuple<matrix<double>, matrix<int>> la_pack::decomposeLU(matrix<double> &A, bool returnPivot)
{
	int nARows = A.numRows(); int nACols = A.numCols();
	matrix<double> joinedLU = A;// The matrix will have both the L and U combined .  
	matrix<int> pivotIndex(A.numRows(), 1);// Stores the pivot permutations
	matrix<double> implicitScaling(A.numRows(), 1);
	double largest = 0, temp = 0; // Used for scaling 
	int oddEven = 1; // Odd or Even number of matrix permutations 
	int imax = 0;

	for (int i = 1; i <= nARows; ++i)  // LOOP OVER ROWS FOR IMPLICIT SCALING 
	{
		largest = 0;
		for (int j = 1; j <= nARows; ++j)
		{
			if ((temp = abs(joinedLU(i, j))) > largest) largest = temp;
		}
		if (largest == 0) error(" LU -> Singular Matrix Input");
		implicitScaling(i, 1) = 1.0 / largest;
	}
	for (int k = 1; k <= nARows; ++k) // kij loop
	{
		largest = 0;
		imax = k;
		for (int i = k; i <= nARows; ++i)
		{
			temp = implicitScaling(i, 1) * abs(joinedLU(i, k));
			if (temp > largest)
			{
				largest = temp;
				imax = i; // Find the column containing the largest element ? 
			}
		}
		if (k != imax)
		{
			for (int j = 1; j <= nARows; ++j)
			{
				joinedLU.swap(imax, j, k, j); // Interchange the row with small numbers , for the row with larger numbers
			}
			oddEven = -oddEven; // One interchange hence becomes odd 
			implicitScaling(imax, 1) = implicitScaling(k, 1);
		}
		pivotIndex(k, 1) = imax;
		if (joinedLU(k, k) == 0) joinedLU(k, k) = _tolerance;
		for (int i = k + 1; i <= nARows; ++i)
		{
			temp = joinedLU(i, k);
			temp /= joinedLU(k, k);
			for (int j = k + 1; j <= nARows; ++j)
				joinedLU(i, j) -= temp * joinedLU(k, j);
		}
	}
	std::tuple<matrix<double>, matrix<int>> result(joinedLU,pivotIndex);
	return result; 
}


matrix <double> la_pack::solveLU(matrix<double> &A, matrix<double> &B )
{
	int nARows = A.numRows();
	int nonZero = 0; //  First Element of B , which is not nonZERO 
	auto LU_P = decomposeLU(A,true);
	matrix<double> LUm = std::get<0>(LU_P);
	matrix<int> pivot = std::get<1>(LU_P);
	matrix<double> X(nARows, 1);
	
	int ip; // Used to dealing with pivots .. 
	double sum; // Used for the forward and backward substitution 
	if (B.numRows() != nARows)
		error("solveLU -> incorrect size for B");
	for (int i = 1; i <= nARows; ++i)
	{
		X(i, 1) = B(i, 1);
	}
	for (int i = 1; i <= nARows; ++i)
	{
		ip = pivot(i, 1);
		sum = X(ip, 1);
		X(ip, 1) = X(i, 1);
		if (nonZero != 0)
		{
			for (int j = nonZero - 1; j < i; ++j) // There might be an Error here ....
			{
				sum -= LUm(i, j) * X(j, 1);
			}
		}
		else if (sum != 0)
		{
			nonZero = i + 1; 
		}
		X(i, 1) = sum; 
	}
	for (int i = nARows; nARows > 0; --i)
	{
		sum = X(i, 1);
		for (int j = i + 1; j <= nARows; ++j)
		{
			sum -= LUm(i, j) * X(j, 1);
		}
		X(i, 1) = sum / LUm(i, i);
	}
	return X; 
}




matrix <double> la_pack::solveLU(matrix<double> &A, matrix<double> &B, matrix<double> &LU, matrix<int> pivot)
{
	int nARows = A.numRows();
	int nonZero = 0; //  First Element of B , which is not nonZERO 
	
	//matrix<double> LUm = LU; // std::get<0>(LU_P);
	//matrix<int> pivot = pivotIndex; //  std::get<1>(LU_P);  // No need to create new matrices !  // Learn to conserve memory ! 
	matrix<double> X(nARows, 1);

	int ip; // Used to dealing with pivots .. 
	double sum; // Used for the forward and backward substitution 
	if (B.numRows() != nARows)
		error("solveLU -> incorrect size for B");
	for (int i = 1; i <= nARows; ++i)
	{
		X(i, 1) = B(i, 1);
	}
	for (int i = 1; i <= nARows; ++i)
	{
		ip = pivot(i, 1);
		sum = X(ip, 1);
		X(ip, 1) = X(i, 1);
		if (nonZero != 0)
		{
			for (int j = nonZero - 1; j < i; ++j) // There might be an Error here ....
			{
				sum -= LU(i, j) * X(j, 1);
			}
		}
		else if (sum != 0)
		{
			nonZero = i + 1;
		}
		X(i, 1) = sum;
	}
	for (int i = nARows; nARows > 0; --i)
	{
		sum = X(i, 1);
		for (int j = i + 1; j <= nARows; ++j)
		{
			sum -= LU(i, j) * X(j, 1);
		}
		X(i, 1) = sum / LU(i, i);
	}
	return X;
}



matrix <double> la_pack::solveLU(matrix<double> &A, matrix<double> &B, int numRightSides)  
{
	int nEqns = numRightSides;
	int nARows = A.numRows();
	int nBRows = B.numRows();
	int nBCols = B.numCols();
	matrix<double> X(nARows,nBCols);

	auto LU_P = decomposeLU(A, true);
	matrix<double> LUm = std::get<0>(LU_P);
	matrix<int> pivot = std::get<1>(LU_P);

	if (nEqns != B.numCols())
	{
		error(" Incorrect number of right Hand sides:: numRightSides and vectors in B does not match");
	}

	if (B.numRows() != nARows)
	{
		error("solveLU:: bad size");
	}
	matrix<double> tempCol(nBRows,1); // Used to store the columns of B one by one 
	for (int j = 1; j <= nBCols; ++j) // Go Thru Each Column in B and solve them one by one ! 
	{
		//X.replaceCol(B.returnCol(j), j);   // Store B col i  in X col i 
		

		//X.replaceCol(solveLU(A, B.returnCol(j), LUm, pivot),j); // give B col n to solve and write result in X col n
	//	matrix<double> matLU = solveLU(A, B.returnCol(j), LUm, pivot);

	// CHANGE THIS LEARN ABOUT WHAT IS HAPPENING

	}

	return X;
}

matrix<double> la_pack::inverseLU(matrix<double> & A)
{
	matrix<double> R = getIdentity(A.numRows());
	R = solveLU(A, R,R.numCols());
	return R; 
}


double la_pack::determinantLU(matrix<double> & A)
{
	double result = 1; 
	matrix<double> LU = decomposeLU(A);
	for (long long i = 1; i <= LU.numRows(); i++)
	{
		for (long long j = 1; j <= LU.numCols(); j++)
		{
			if (j == i)
			{
				//_matrix[(i - 1)*_cols + (j - 1)] = rowVec(i, 1);
				result *= LU(i, j);
			}
		}
	}
	return result;
}
