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

#include <stdexcept>

template <class T>
class matrix;


//OPERATORS
template<typename T>	
std::ostream& operator<<(std::ostream& out, const matrix<T>& temp); // Not working 




//------------------------------------------------------------------------------------------------

template <class T>
class matrix
{
public:
	//CONSTRUCTORS
	matrix(void)  : _rows(0) , _cols(0) , _size(0)
	{

	};
       
	~matrix(void);


	matrix(size_t rows , size_t cols , T val = 0) : _rows(rows) , _cols(cols) , _matrix(_rows * _cols, val), _size(_rows * _cols)
	{

	}
matrix(const matrix<T>& rhs) : _rows(rhs._rows) , _cols(rhs._cols) , _size(rhs._size)
	{
		_matrix = rhs._matrix;
	};

	
	//GENERAL
	// resize the matrix	
	void resize(size_t rows , size_t cols , T val = 0)
	{
		_rows = rows; 
		_cols = cols;
		_size = rows * cols;		
	
		_matrix.resize(_size , val);

	}	


	void print();
	size_t  numRows() const { return (_rows); }; // Implicit inline 
	size_t numCols() const { return  (_cols); };
	void randFill(T start = 0, T end = 1000); // Will only work for simple data types
	void symetricRandFill(T start = 0, T end = 1000); // Will only work for simple data types
	void setAllNum(T aNum);
	matrix<T> returnRow(size_t aRow) const;
	matrix<T> returnCol(size_t aCol) const;
	matrix<T> removeCol(size_t aCol);
	matrix<T> removeRow(size_t aRow);

	void replaceCol(matrix<T> colVec,int aCol) ;
	void replaceRow(matrix<T> rowVec, int aRow)  ;
	void swap(int aRow, int aCol, int bRow, int bCol) 
	{ 
		T temp = this->get(aRow, aCol); 
	  	this->insert(aRow, aCol, this->get(bRow, bCol)); this->insert(bRow, bCol, temp); };

	matrix<T> getDiagonalVector() const;
	matrix<T> getDiagonalMatrix() const;


	matrix<T> addCol(const matrix<T>& col);
	matrix<T> addRow(const matrix<T>& row);
	bool isSquare() const { return ((_rows == _cols) ? true : false); }
	bool isRowVector()const { return ((_rows == 1) ? true : false); }
	bool isColVector() const { return ((_cols == 1) ? true : false); }
	bool isDiagonal() const;
	bool isEqual(matrix<T> rhs) const;
	bool isUpperTriangular() const ;
	bool isLowerTriangular() const;
	bool isDiagonallyDominant() const;


	matrix<T> getIdentity(long long  aRow);
	T sum();
	matrix<T> transpose();
	T innerProduct(const matrix<T>& B);

	matrix<T>& operator=(const matrix<T>& rhs);
	matrix<T> operator*(const T rhs);
	matrix<T> operator/(const T rhs);
	matrix<T> operator*(const matrix<T> &rhs);
	matrix<T> operator+(const matrix<T> &rhs);
	matrix<T> operator-(const matrix<T> &rhs);
	
	bool operator==(const matrix<T> & rhs);
	bool operator!=(const matrix<T> & rhs){ return !(*this == rhs); }

	T& operator()(const long long  rows, const long long  cols) const;


private:


	void init(); // sets up _matirx 

	// The size specifiers cannot be const as the matrix has the ability to resize	
	size_t  _rows;
	size_t  _cols;
	size_t _size;


	std::vector<T> _matrix;

	// GENERAL
	void insert(size_t i, size_t j, T aVaule) 
	{ 
		_matrix[(i - 1)*_cols + (j - 1)] = aVaule; 
	}

	T& get(size_t i, size_t j) const 
	{ 
		return const_cast<T &>( _matrix[(i - 1)*_cols + (j - 1)] ) ; 
	}

	static void error(const char* p)
	{ 
		std::string str = "matrix -> Error: "; std::cout << str << p << std::endl; 
	}
};




// TYPEDEF's 

typedef matrix<double> matDouble; 
typedef matrix<int> matInt;


/*
 * add support to add a row at the end of the current matrix.
 * 
 *
 */

template <class T>
matrix<T> matrix<T>::addRow(const matrix<T>& row)
{
	if(row.isRowVector )
	{
		if(row.numCols() != _cols && _cols != 0)
		{
			throw std::invalid_argument(std::string("Given row does not have eqaul number of columns as current matrix"));
		}

		// More optimial might to be reserve and then add move elmements ?
		
		_matrix.reserve(_size + row._size);	 // increase the allocated memory

		_matrix.insert(_matrix.end() , row.begin()  , row.end());	

		// update the bookkeeping
		_size += row._size;
		++_rows;
	}
	else
	{
		throw std::invalid_argument(std::string("Given row is not a row Vector in addRow"));
	}


}


// Fills in with 1. Make sure 1 can be casted into your template class
template <class T>
matrix<T> matrix<T>::getIdentity(long long  aRow)
{
	matrix<T> A(aRow,aRow);
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





template <class T>
bool matrix<T>::isEqual(matrix<T> rhs) const
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

// Diagonal element is greater than the other elements in row
template <class T>
bool matrix<T>::isDiagonallyDominant() const
{  
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
matrix<T> matrix<T>::returnRow(size_t aRow) const
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
matrix<T> matrix<T>::removeCol(size_t aCol)
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
matrix<T> matrix<T>::removeRow(size_t aRow)
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
matrix<T> matrix<T>::returnCol(size_t aCol) const
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
void matrix<T>::print()
{
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
		}
	}
}

//------------------------------------------------------------------------------------------------

template <class T >
matrix<T>::~matrix(void)
{
	_matrix.clear();
}


//------------------------------------------------------------------------------------------------

template <class T>
void matrix<T>::init(){
	// Initlializer called by the matrix(row,col)
	_matrix = std::vector<T>(_rows*_cols , 0 ); // Fills in the vector with zeros

}

template <class T>
T matrix<T>::sum()
{
	T sum = 0; 

	if (isColVector())
	{
		for (size_t i = 1; i <= numRows(); i++) // Go through each row and add them up 
		{
			sum += get(i, 1);
		}
	}
	else if (isRowVector())
	{
		for (size_t i = 1; i <= numCols(); i++) // Go through each row and add them up 
		{
			sum += get(1,i);
		}

	}
	else // matrix is a normal matrix , fat or thin 
	{
		for (size_t i = 1; i <= numRows(); i++)
		{
			for (size_t j = 1; j <= numCols(); j++)
			{
				sum += get(i, j);
			}
		}
	}

	return sum; 
}


template <class T>
std::ostream& operator<<(std::ostream& out, const matrix<T>& temp)
{
	out << std::endl;
	for (long long i = 1; i <= temp.numRows(); i++)
	{
		for (long long j = 1; j <= temp.numCols(); j++)
		{
		//	out << temp.get(i,j);
			out << std::setw(7) << temp(i, j);
			if (j == temp.numCols()) out << std::endl;
		}
	}
	return out;
}



/*
 * Tranpose the given matrix
 *
 */
template <class T>
matrix<T> matrix<T>::transpose()
{
	matrix<double> R(numCols(),numRows()); 
	for(long long  i = 1; i <= numRows() ; i++)
	{
		for(long long  j = 1; j <= numCols() ; j++)
		{
			R(j,i) = get(i,j);
		}
	}
	return R;
}



/*
 * Computes the dot product
	inner Product :
	At * B // t = transpose
	Row Vec * Col Vec 
	(1, n) * (n , 1)

	Allows the user to be sloppy
	Here A is the current matrix

*/
template <class T>
T matrix<T>::innerProduct(const matrix<T>& B)
{
	T result = 0 ; 
	
	if(isRowVector() && B.isColVector() && (numCols() == B.numRows()))
	{
		for(size_t  i = 1 ; i <= numRows();i++)
		{	
			result += get(1,i)*B(i,1);
		}
		return result;
	}
	else if(isRowVector() && B.isRowVector() && (numCols() == B.numCols()))
	{
		for(size_t i = 1 ; i <= numCols();i++)
		{	
			result += get(1,i)*B(1,i);
		}
		return result;
	}
	else if(isColVector() && B.isColVector() && (numRows() == B.numRows()))
	{
		for(size_t i = 1 ; i <= numRows();i++)
		{	
			result += get(i,1)*B(i,1);
		}
		return result;
	}
	else
	{
		error("innerProduct -> A and B are not in proper format ");
		return result*(0);
	}
}



#endif
