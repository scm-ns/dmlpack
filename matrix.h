#ifndef MATRIX_H
#define MATRIX_H


/*
Martirx
Allocated in Heap
Implement matrix as a single array for preventing chache misses
Access to rows and col normal indexed , starts from 1 , not 0
Maximum Number of Elements is 10^8 ie 10,000 rows and 10,000 cols ;; If higer gives std::length_error
//Not Implemented , change _size , _col _row to long long for larger matrices.

Total time Taken : 12
Started April 4th
End April 4th Night

*/

//System Include 
#include <iostream> //Output
#include <ostream>
#include <string>  // Error Handling 
#include <vector>  // Internal DS 
#include <iomanip> // Output 
#include <time.h>  // randFill
#include <stdlib.h> //randFill
#include <stdio.h>  //randFill
#include <cstdlib>


//------------------------------------------------------------------------------------------------

template <class T>
class matrix
{
public:
	//CONSTRUCTORS
	matrix(void) {}; // Give a function body . even if you do not give it a body to prevent linker errors
	~matrix(void); // On line 64 ; 
	matrix(long long rows, long long  cols) :_rows(rows), _cols(cols), _matrix(_rows*_cols){ _size = _rows*_cols; } // 
	matrix(const matrix<T>& rhs);

	//GENERAL
	void print();
	long long  numRows() const { return (_rows); }; // Implicit inline 
	long long  numCols() const { return  (_cols); };
	void randFill(T start = 0, T end = 1000); // Will only work for simple data types
	void symetricRandFill(T start = 0, T end = 1000); // Will only work for simple data types
	void setAllNum(T aNum);
	matrix<T> returnRow(int aRow) const;
	matrix<T> returnCol(int aCol) const;
	matrix<T> removeCol(int aCol);
	matrix<T> removeRow(int aRow);

	void replaceCol(matrix<T> colVec,int aCol) ;
	void replaceRow(matrix<T> rowVec, int aRow)  ;
	void swap(int aRow, int aCol, int bRow, int bCol) { int temp = this->get(aRow, aCol); this->insert(aRow, aCol, this->get(bRow, bCol)); this->insert(bRow, bCol, temp); };

	matrix<T> getDiagonalVector() const;
	matrix<T> getDiagonalMatrix() const;
	// Adds a single Column to the matrix 


	// Adds a single Column to the matrix 
	matrix<T> addCol(matrix<T> col);
	matrix<T> addRow(matrix<T> row);
	bool isSquare() const { return ((_rows == _cols) ? true : false); }
	bool isRowVector()const { return ((_rows == 1) ? true : false); }
	bool isColVector() const { return ((_cols == 1) ? true : false); }
	bool isDiagonal() const;
	bool isEqual(matrix<T> rhs) const;
	bool isUpperTriangular() const ;
	bool isLowerTriangular() const;
	bool isDiagonallyDominant() const;
	//OPERATORS
	//friend std::ostream& operator<<(std::ostream& out, matrix<T> &temp); // Not working 
	matrix<T>& operator=(const matrix<T>& rhs);
	matrix<T> operator*(const T rhs);
	matrix<T> operator/(const T rhs);
	matrix<T> operator*(const matrix<T> &rhs);
	matrix<T> operator+(const matrix<T> &rhs);
	matrix<T> operator-(const matrix<T> &rhs);
	
	bool operator==(const matrix<T> & rhs);
	bool operator!=(const matrix<T> & rhs){ return !(*this == rhs); }


	T& operator()(long long  rows, long long  cols) const ;//friend std::ostream& operator<<(std::ostream& out , matrix &temp);

	//------------------------------------------------------------------------------------------------



private:

	// CONSTRUCTORS
	//matrix(void);

	void init(); // sets up _matirx 

	//DATA // Make Constant in Future 
	long long  _rows;
	long long  _cols;
	long long  _size;
	std::vector<T> _matrix;

	// GENERAL
	void insert(long long i, long long j, T aVaule) { _matrix[(i - 1)*_cols + (j - 1)] = aVaule; }
	T& get(long long  i, long long  j) const { return const_cast<T &>( _matrix[(i - 1)*_cols + (j - 1)] ) ; }
	static void error(const char* p){ std::string str = "matrix -> Error: "; std::cout << str << p << std::endl; }
};




// TYPEDEF's 

typedef matrix<double> matDouble; 
typedef matrix<int> matInt;


// ----------------------------------------------------------------------------------------------------------------------------------------------


template <class T>
bool matrix<T>::operator==(const matrix<T> & rhs)
{
	return this->isEqual(rhs);
}


template <class T>
void matrix<T>::symetricRandFill(T start , T end )
{
	if (_rows == _cols)
	{
		std::srand(time(0));
		for (long long i = 1; i <= _rows; i++)
		{
			for (long long j = 1; j <= _cols; j++)
			{
				if (i >= j)
				{
					_matrix[(i - 1)*_cols + (j - 1)] = static_cast<T>(std::rand() / (end - start) + 1);
				}
			}
		}

		for (long long i = 1; i <= _rows; i++)
		{
			for (long long j = 1; j <= _cols; j++)
			{
				if (j > i)
				{
					_matrix[(i - 1)*_cols + (j - 1)] = _matrix[(j - 1)*_cols + (i - 1)];
				}
			}
		}
	}
	else
	{
		error("Non Square matrix Cannot be square "); 
	}
}


template <class T>
void matrix<T>::replaceCol(matrix<T> colVec, int aCol) 
/*
Replaces the specified column of matrix with a given column
Index as always starts at 1
*/
{
	if (colVec.isColVector())
	{
		for (long long i = 1; i <= _rows; i++)
		{
			for (long long j = 1; j <= _cols; j++)
			{
				if (aCol == j)
				{
					//_matrix[(i - 1)*_cols + (j - 1)] = ;
					insert(i, j, colVec(i, 1));
				}
			}
		}

	}
	else
	{
		error("replaceCol : input Vector is not Col Vec");
	}
}


template <class T>
void matrix<T>::replaceRow(matrix<T> rowVec, int aRow) 
/*
Replaces the specified row of matrix with a given row
Index as always starts at 1
*/
{
	if (rowVec.isRowVector())
	{
		for (long long i = 1; i <= _rows; i++)
		{
			for (long long j = 1; j <= _cols; j++)
			{
				if (aRow == i)
				{
					//_matrix[(i - 1)*_cols + (j - 1)] = rowVec(i, 1);
					insert(i, j, rowVec(1, j));
				}
			}
		}

	}
	else
	{
		error("replaceRow : input Vector is not Row Vec");
	}
}

template <class T>
matrix<T> matrix<T>::getDiagonalVector() const
{
	matrix<T> vector(_rows, 1);
		for (long long i = 1; i <= _rows; i++)
		{
			for (long long j = 1; j <= _cols; j++)
			{
				if (j == i)
				{
					//_matrix[(i - 1)*_cols + (j - 1)] = rowVec(i, 1);
					vector(i, 1) = get(i, j);
				}
			}
		}

	return vector;

}

template <class T>
matrix<T> matrix<T>::getDiagonalMatrix() const
{
	matrix<T> diagMatrix(_rows, _cols);
	for (long long i = 1; i <= _rows; i++)
	{
		for (long long j = 1; j <= _cols; j++)
		{
			if (j == i)
			{
				//_matrix[(i - 1)*_cols + (j - 1)] = rowVec(i, 1);
				diagMatrix(i, j) = get(i, j);
			}
		}
	}
	return diagMatrix;
}





//
//template <class T>
//void matrix<T>::symetricRandFill(T start, T end)
//{
//	if (_rows == _cols)
//	{
//		std::srand(time(0));
//		for (long long i = 1; i <= _rows; i++)
//		{
//			for (long long j = 1; j <= _cols; j++)
//			{
//				if (i >= j)
//				{
//					_matrix[(i - 1)*_cols + (j - 1)] = static_cast<T>(std::rand() / (end - start) + 1);
//				}
//			}
//		}
//
//		for (long long i = 1; i <= _rows; i++)
//		{
//			for (long long j = 1; j <= _cols; j++)
//			{
//				if (j > i)
//				{
//					_matrix[(i - 1)*_cols + (j - 1)] = _matrix[(j - 1)*_cols + (i - 1)];
//				}
//			}
//		}
//	}
//	else
//	{
//		error("Non Square matrix Cannot be square ");
//	}
//}


template <class T>
bool matrix<T>::isEqual(matrix<T> rhs) const
/*
returns True if equal , else false ;
*/
{
	bool equal = false;
	if (rhs.numRows() == numRows() && rhs.numCols() == numCols())
	{
		for (long long i = 1; i <= _rows; i++)
		{
			for (long long j = 1; j <= _cols; j++)
			{
				if (get(i, j) == rhs(i, j)) equal = true;
				else return false; // is even a single element does not match false . 
			}
		}
		return equal;
	}
	else
	{
		return equal;
	}
}




template <class T>
void matrix<T>::setAllNum(T aNum)
{
	for (long long i = 1; i <= _rows; i++)
	{
		for (long long j = 1; j <= _cols; j++)
		{
			get(i, j) = aNum;
		}
	}
}

template <class T>
bool matrix<T>::isDiagonallyDominant() const
{  // Diagonal element is greater than the other elements in row
	bool isDominant = false;
	for (long long i = 1; i <= _rows; i++)
	{
		int diagonalElement = 0;  // New values for each row 
		int rowSum = 0;
		for (long long j = 1; j <= _cols; j++)
		{
			if (i == j)
				diagonalElement = get(i, j);
			else
				rowSum = rowSum + get(i, j);
		}
		if (diagonalElement > rowSum)
			isDominant = true;
		else
			return false;  // is even one encountered we break out . 
	}
	return isDominant;
}


template <class T>
bool matrix<T>::isUpperTriangular() const
{
	if (isSquare())
	{
		bool isUpperTriangularFlag = false;
		for (long long i = 1; i <= _rows; i++)
		{
			for (long long j = 1; j <= _cols; j++)
			{
				if (i > j)
				{
					if (get(i, j) == 0)
					{
						isUpperTriangularFlag = true;
					}
					else
					{
						isUpperTriangularFlag = false;
						return isUpperTriangularFlag;
					}
				} // if 	
			}// for 
		}// for
		return isUpperTriangularFlag; // AFter going through each element	
	}
	else
		return false;
}

template <class T>
bool matrix<T>::isLowerTriangular() const
{
	if (isSquare()){
		bool isLowerTriangularFlag = false;
		for (long long i = 1; i <= _rows; i++)
		{
			for (long long j = 1; j <= _cols; j++)
			{
				if (i < j)
				{
					if (get(i, j) == 0)
						isLowerTriangularFlag = true;
					else
					{
						isLowerTriangularFlag = false;
						return isLowerTriangularFlag;
					}
				}
			}
		}
		return isLowerTriangularFlag; // AFter going through each element
	}
	else
		return false;
}







template <class T>
bool matrix<T>::isDiagonal() const
{
	if (isSquare())
	{
		bool isDiagonalFlag = false;
		for (long long i = 1; i <= _rows; i++)
		{
			for (long long j = 1; j <= _cols; j++)
			{
				if (i == j) continue;
				if (get(i, j) == 0)
				{
					isDiagonalFlag = true;
				}
				else
				{
					isDiagonalFlag = false;
					return isDiagonalFlag;
				}
			}
		}
		return isDiagonalFlag;
	}
	else
		return false;
}

//------------------------------------------------------------------------------------------------

template <class T>
matrix<T> matrix<T>::returnRow(int aRow) const
/*
Create a new matrix with same number of cols , but only a single row
*/
{
	matrix<T> result(1, _cols);
	for (long long i = 1; i <= _rows; i++)
	{
		for (long long j = 1; j <= _cols; j++)
		{
			if (aRow == i)
			{
				result(1, j) = get(i, j);
			}
		}
	}
	return result;
}

template <class T>
matrix<T> matrix<T>::removeCol(int aCol)
/*
No operation is done on the input matrix , the matrix returned is a new matrix
remove a col and returns a matrix of same size but without the specified col
*/
{
	matrix<T> R(_rows, _cols - 1);
	for (long long i = 1; i <= R._rows; i++)
	{
		for (long long j = 1; j <= R._cols; j++)
		{
			if (aCol <= j)
				R(i, j) = get(i, j + 1);
			else
				R(i, j) = get(i, j);
		}
	}
	return R;
}

template <class T>
matrix<T> matrix<T>::removeRow(int aRow)
/*
No operation is done on the input matrix , the matrix returned is a new matrix
remove a col and returns a matrix of same size but without the specified col
*/
{
	matrix<T> R(_rows - 1, _cols);
	for (long long i = 1; i <= R._rows; i++)
	{
		for (long long j = 1; j <= R._cols; j++)
		{
			if (aRow <= i)
				R(i, j) = get(i + 1, j);
			else
				R(i, j) = get(i, j);
		}
	}
	return R;
}

//----------------------------------------------------------------------------------------------------------------------------------------------

template <class T>
matrix<T> matrix<T>::returnCol(int aCol) const
/*
Create a new matrix with same number of cols , but only a single row
*/
{
	matrix<T> result(_rows, 1);
	for (long long i = 1; i <= _rows; i++)
	{
		for (long long j = 1; j <= _cols; j++)
		{
			if (aCol == j)
			{
				result(i, 1) = get(i, j);
			}
		}
	}
	return result;
}




//------------------------------------------------------------------------------------------------


template <class T>
matrix<T>::matrix(const matrix<T>& rhs)
{
	/*
	//Copy Constructor
	*/
	_matrix = rhs._matrix;
	_rows = rhs._rows;
	_cols = rhs._cols;
}


template <class T>
matrix<T>  matrix<T>::operator*(const matrix<T> & rhs)
/*
if takes the current matrix , multiplies it by the matrix on the right , and returns
a new matrix
*/
{
	matrix<T> result(_rows, rhs._cols);

	if (_cols == rhs._rows)
	{
		for (long long i = 1; i <= _rows; i++)
		{
			for (long long j = 1; j <= rhs._cols; j++)
			{
				for (long long k = 1; k <= _cols; k++)
				{
					result(i, j) += get(i, k) * rhs(k, j);
				}
			}
		}
	}
	else
	{
		error("M*M -> Rows And Col Does Not Match");
	}

	return result;
}



template <class T>
matrix<T> matrix<T>::operator*(const T rhs)
/*
Multiply by scalar
DoesNot Modify Input Matrix
*/
{
	matrix<T> result(_rows, _cols);// rhs is a T 
	for (long long i = 1; i <= _rows; i++)
	{
		for (long long j = 1; j <= _cols; j++)
		{
			result(i, j) = get(i, j)*rhs;
		}
	}
	return result;
}

template <class T>
matrix<T> matrix<T>::operator/(const T rhs)
/*
Divide by scalar
DoesNot Modify Input Matrix
*/
{
	matrix<T> result(_rows, _cols);// rhs is a T 
	for (long long i = 1; i <= _rows; i++)
	{
		for (long long j = 1; j <= _cols; j++)
		{
			result(i, j) = get(i, j) / rhs;
		}
	}
	return result;
}

//------------------------------------------------------------------------------------------------
// 



//------------------------------------------------------------------------------------------------

template <class T>
matrix<T>& matrix<T>::operator=(const matrix<T>& rhs)
{
	/*
	Copy Assignment Operator
	Only Support Single Type Copy
	Eg: int to int , long long  to long long
	*/
	// Check if same 
	if (&rhs == this)
		return *this;

	//Copy rows and cols
	_rows = rhs._rows;
	_cols = rhs._cols;
	_size = rhs._size; 
	_matrix = rhs._matrix; 
	//std::swap(_matrix ,rhs._matrix);
	//_matrix.swap(rhs._matrix);

	return *this;
}

//------------------------------------------------------------------------------------------------

template <class T>
matrix<T> matrix<T>::operator+(const  matrix<T> &rhs)
{

	matrix<T> R(_rows, _cols);
	if (_rows == rhs._rows && _cols == rhs._cols)
	{
		for (long long i = 1; i <= _rows; i++)
		{
			for (long long j = 1; j <= _cols; j++)
			{
				R(i, j) = get(i, j) + rhs(i, j);
			}
		}
		return R;
	}
	else
	{
		error("Not of same size ");
		return R;
	}
}



template <class T>
matrix<T> matrix<T>::operator-(const matrix<T> &rhs)
{

	matrix<T> R(_rows, _cols);
	if (_rows == rhs._rows && _cols == rhs._cols)
	{
//		matrix<T> R(_rows, _cols);
		for (long long i = 1; i <= _rows; i++)
		{
			for (long long j = 1; j <= _cols; j++)
			{
				R(i, j) = get(i, j) - rhs(i, j);
			}
		}
		return R;
	}
	else
	{
		error("Not of same size ");
		return R;
	}
}


//------------------------------------------------------------------------------------------------



template <class T>
void matrix<T>::print(){
	std::cout << std::endl << std::endl;
	for (long long i = 1; i <= _rows; i++)
	{
		for (long long j = 1; j <= _cols; j++)
		{
			//out << temp.get(i,j);
			std::cout << std::setw(15) << get(i, j);
			if (j == _cols) std::cout << std::endl;
		}
	}
	std::cout << std::endl;
}


//------------------------------------------------------------------------------------------------

template <class T>
T& matrix<T>::operator()(const long long  rows, const long long  cols) const
{
	/*std::cout << "Safdsad";
	_matrix will have _rows and _cols // Do not worry about 0 index , that is handled
	-----------------------------------*/
	if (rows > _rows || cols > _cols)
	{
		error("() -> Index out of range :: Returning "); std::cout << rows << cols << std::endl;
		return get(1, 1);
	}
	else
		return get(rows, cols);
}


//------------------------------------------------------------------------------------------------
template <class T>
void matrix<T>::randFill(T start, T end)
{
	std::srand(time(0));
	for (long long i = 1; i <= _rows; i++)
	{
		for (long long j = 1; j <= _cols; j++)
		{
			_matrix[(i - 1)*_cols + (j - 1)] = static_cast<T>(std::rand() / (end - start) + 1);
			// Look at Stack Over Flow Question
		}
	}
}

//------------------------------------------------------------------------------------------------

template <class T >
matrix<T>::~matrix(void)
{
	//delete [] _matrix;
	// Memory management handled by vector class	
	_matrix.clear();
}


//------------------------------------------------------------------------------------------------

template <class T>
void matrix<T>::init(){
	/*
	// Initlializer called by the matrix(row,col)
	---------------------------------------------*/
	_matrix = std::vector<T>(_rows*_cols);
	for (long long i = 1; i <= _rows; i++)
	{
		for (long long j = 1; j <= _cols; j++)
		{
			_matrix[(i - 1)*_cols + (j - 1)] = 0;
		}
	}
}

// instead of rand use std::random . implement later 

/*
template <class T ,typename Fn ,typename ...Args>
matrix<T>::loopApply(Fn fn , Args...args )
{
for(long long  i = 1; i <= _rows ;i++)
{
for(long long  j = 1 ; j <= _cols ; j++)
{
_matrix[(i - 1)*_cols + (j -1 )] = fn(args...) ;
}
}
}




template <class T>
std::ostream& operator<<(std::ostream& out, matrix<T> &temp)
{
	for (long long i = 1; i <= _rows; i++)
	{
		for (long long j = 1; j <= _cols; j++)
		{
			//out << temp.get(i,j);
			out << std::setw(7) << get(i, j);
			if (j == _cols) out << std::endl;
		}
	}
	return out;
}

*/


#endif
