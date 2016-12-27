#ifndef MATRIX_H
#define MATRIX_H
/*
	Martirx
	Access to rows and col normal indexed , starts from 1 , not 0
	Maximum Number of Elements is 10^8 ie 10,000 rows and 10,000 cols ;; If higer gives std::length_error

	TO DO : Write a tensor class. Which uses this as a base class and extended it 

*/

//System Include 
#include <iostream> //Output
#include <ostream>
#include <string>  // Error Handling 
#include <vector>  // Internal DS 
#include <iomanip> // Output 
#include <time.h>  
#include <stdlib.h> 
#include <stdio.h>  
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <algorithm>
#include <functional>


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


	matrix<T>& operator=(const matrix<T>& rhs);


	// creates a row vector
	matrix(std::initializer_list<T> l);
	

	// TO DO : Functionality to convert a row vec to col vise-versa. Also to convert row/col vec into 2d matrix // this resize method seems to do it. But check.
	void resize(size_t rows , size_t cols , T val = 0);
	

	inline typename std::vector<T>::iterator begin();
	inline typename std::vector<T>::iterator end();
	typename std::vector<T>::iterator iterAtRowBegin(const size_t row_idx);



	size_t  numRows() const ;
	size_t numCols() const ;
	size_t size() const ;

	void randFillUniform(T start = 0 , T end = 1000);
	void randFill(T start = 0, T end = 1000); 
	void symetricRandFill(T start = 0, T end = 1000); 

	void resizeLinSpaceRow(T start , T end,T interval);
	void resizeLinSpaceCol(T start , T end,T interval);

	void setAllNum(T aNum);

	matrix<T> returnRow(size_t aRow) const;
	matrix<T> returnCol(size_t aCol) const;

	matrix<T> removeCol(size_t aCol);
	matrix<T> removeRow(size_t aRow);

	void replaceCol(matrix<T> colVec,int aCol) ;
	void replaceRow(matrix<T> rowVec, int aRow)  ;

	void swap(int aRow, int aCol, int bRow, int bCol) ;

	matrix<T> getDiagonalVector() const;
	matrix<T> getDiagonalMatrix() const;

	void addRow( matrix<T>& row);
	// To do : Add Col. 
	

	bool isSquare()		 const;
	bool isRowVector()       const;
	bool isColVector() 	 const;
	bool isDiagonal() 	 const;
	bool isEqual(matrix<T> rhs) const;
	bool isUpperTriangular()    const;
	bool isLowerTriangular()    const;
	bool isDiagonallyDominant() const;


	void createIdentity(long long  aRow);
	matrix<T> transpose() const;
	T innerProduct(const matrix<T>& B) const;
	T sum();

	T selfInnerProduct();

	// returns the index which has the maximum value in the vector // onlys works on row or column vectors, not on matrix
	size_t arg_max();
	
	template <class P>
	matrix<T> operator*(const P rhs) const;
	//template <class P>
	//void operator*(const P rhs) ;
	matrix<T> operator/(const T rhs) const;
	matrix<T> operator*(const matrix<T> &rhs) const;
	matrix<T> operator+(const matrix<T> &rhs) const;
	matrix<T> operator-(const matrix<T> &rhs) const;
	
	bool operator==(const matrix<T> & rhs) const ;
	bool operator!=(const matrix<T> & rhs) const { return !(*this == rhs); }

	T& operator()(const long long  rows, const long long  cols); // not a const operation as this can be used to change the value
	T operator()(const long long  rows, const long long  cols)  const;

	T normEuclidean();
	matrix<T> transform_create(std::size_t rows , std::size_t cols , std::function<T(std::size_t , std::size_t , matrix<T>)> lam);
	matrix<T> transform_inplace(std::function<T(std::size_t , std::size_t , T)> lam) ;
	

private:

	// The size specifiers cannot be const as the matrix has the ability to resize	
	size_t  _rows;
	size_t  _cols;
	size_t _size;

	std::vector<T> _matrix;

	// TO DO : Update the way to access an elemnet in a matrix. Right now it takes time to add and find the result. 
	// 	 : Since get is usually used to iterate over a row, create an iteartor for this, so that, we do not compute the index all over again. each time

	inline void insert(size_t i, size_t j, T aVaule) ;
	inline T get(size_t i, size_t j) const;
	inline T& get_ref(size_t i, size_t j);
	
};


template<typename T>	
std::ostream& operator<<(std::ostream& out, const matrix<T>& temp); 

// Create a row vector, by specificying the items the vector is to be filled it
// Discussion : 	
// 	Inefficinet as a copy of the value is created by the compiler ( a temporary ) 
// 	then it is copied into the std::vector
//	So use for small vectors outside of a for loop
template<typename T>	
matrix<T>::matrix(std::initializer_list<T> l)
{
	// itereators into list
	const T*   it = l.begin();
	const T* const end = l.end();
	
	// create a row vector
	_rows = 1 ;  
	_cols = l.size();
	_size = _rows * _cols;

	// reserve might be redundant. ?? Copy might call it ? 
	// allocates memory of required size, prevents reallocations and deallocation during addintion of new values
	_matrix.reserve(_size);
	
	// copy the values in the init.. list into the vector 
	std::copy(l.begin() , l.end() , std::back_inserter(_matrix));
}

template <class T >
matrix<T>::~matrix(void)
{
	_matrix.clear();
}


// Copy Assignment Operator
// 	Only Support Single Type Copy
// Eg: int to int , long long  to long long
template <class T>
matrix<T>& matrix<T>::operator=(const matrix<T>& rhs)
{
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

// HANDLE SIZE OF THE MATRIX

template<typename T>	
size_t  matrix<T>::numRows() const 
{ 
	return (_rows); 
}; 

template<typename T>	
size_t matrix<T>::numCols() const 
{
       	return  (_cols); 
};

template<typename T>	
size_t matrix<T>::size() const 
{ 
	return _matrix.size() ; 
};

template<typename T>	
inline typename std::vector<T>::iterator matrix<T>::begin() 
{
	return _matrix.begin();
}

template<typename T>	
inline typename std::vector<T>::iterator matrix<T>::end() 
{
	return _matrix.end();
}
	
 //Iterator at the begining of each of the rows
template<typename T>	
typename std::vector<T>::iterator matrix<T>::iterAtRowBegin(const size_t row_idx)
{
	typename std::vector<T>::iterator it = begin();
	std::advance(it , (row_idx * _cols));
	return it;
}

// insert value in a particular position in the vector
template<typename T>	
inline void matrix<T>::insert(size_t i, size_t j, T aVaule) 
{ 
	_matrix[(i - 1)*_cols + (j - 1)] = aVaule; 
}

// just returns the value.
// mutating
template<typename T>	
inline T matrix<T>::get(size_t i, size_t j) const
{ 
	return ( _matrix[(i - 1)*_cols + (j - 1)] ) ; 
}

// returns reference.
// mutating
template<typename T>	
inline T& matrix<T>::get_ref(size_t i, size_t j)
{
	return ( _matrix[(i - 1)*_cols + (j - 1)] ) ; 
}

// Helpers to check if the matrix if of a common type

template<typename T>	
bool matrix<T>::isSquare() const 
{ 
	if(_rows == _cols )
	{
		return true;
	}
	else
	{
		return false;
	}
}

template<typename T>	
bool matrix<T>::isRowVector()const 
{ 
	if(_rows == 1)
	{
		return true;
	}
	else
	{
		return false;
	}
}

template<typename T>	
bool matrix<T>::isColVector() const 
{ 
	if(_cols == 1)
	{
		return true;
	}
	else
	{
		return false;
	}
}

template<typename T>	
void matrix<T>::swap(int aRow, int aCol, int bRow, int bCol) 
{ 
	T temp = get(aRow, aCol); 
	insert(aRow, aCol, get(bRow, bCol));
	insert(bRow, bCol, temp); 
};



// Resize the vector into any shape you want. Keep the old items, pads new items to full fill new size requirement
template<typename T>	
void matrix<T>::resize(size_t rows , size_t cols , T val )
{
	_rows = rows; 
	_cols = cols;
	_size = rows * cols;		
	_matrix.resize(_size , val);
}	

// create a new matrix of specified rows and cols and applies the lambda function to each of them
template <class T>
matrix<T> matrix<T>::transform_create(std::size_t rows , std::size_t cols , std::function<T(std::size_t , std::size_t , matrix<T>)> lam) 
{
	matrix<T> res(rows , cols);
	for(std::size_t idx = 0 ; idx < rows ; ++idx)
	{
		for(std::size_t  jdx = 0 ; jdx < cols ; ++jdx)
		{
			res(idx , jdx) =  lam(idx , jdx , res);	
		}
	}	
	return res;
}


// apply the lambda function ot each element of the vector. 
template <class T>
matrix<T> matrix<T>::transform_inplace(std::function<T(std::size_t , std::size_t , T)> lam) 
{
	for(std::size_t idx = 0 ; idx < _rows ; ++idx)
	{
		for(std::size_t  jdx = 0 ; jdx < _cols ; ++jdx)
		{
			insert(idx , jdx , lam(idx , jdx , get(idx, jdx) ) );	 // highly inefficeint code. Make it simpler
		}
	}	
}

// Get the maximum element in a row or col vector
//  0 indexed
template <class T>
size_t matrix<T>::arg_max()
{
	if(isRowVector() || isColVector())
	{
		std::distance(std::begin(_matrix) , std::max_element(std::begin(_matrix) , std::end(_matrix))) + 1;
	}
	else
	{
		throw std::invalid_argument(std::string("arg_max can only be applied to a row or col vector"));
	}

}



 // add support to add a row at the end of the current matrix.
template <class T>
void matrix<T>::addRow( matrix<T>& row)
{
	if(row.isRowVector() )
	{
		if(row.numCols() != _cols && _cols != 0)
		{
			throw std::invalid_argument(std::string("Given row does not have eqaul number of columns as current matrix"));
		}

		// More optimial might to be reserve and then add move elmements ?
	
		if(_size == 0)
		{
			_matrix = row._matrix;
		}
		else
		{
			_matrix.insert(_matrix.end() , row.begin()  , row.end());	
		}

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
void matrix<T>::createIdentity(long long  aRow)
{
	for(long long  i = 1 ; i <= aRow ; i++)
	{
		for(long long  j = 1 ; j <= aRow ; j++)
		{
			if(i == j )
			{
				insert(i,j,1);
			}
		}
	}
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
		throw std::logic_error("Non Square matrix Cannot be square "); 
	}
}

// Replaces the specified column of matrix with a given column
//Index as always starts at 1
template <class T>
void matrix<T>::replaceCol(matrix<T> colVec, int aCol) 
{
	if (colVec.isColVector())
	{
		for (size_t i = 1 ; i <= _rows; ++i)
		{
			insert(i, aCol, colVec(i, 1));
		}

	}
	else
	{
		throw std::logic_error("replaceCol : input Vector is not Col Vec");
	}
}

// 	Replaces the specified row of matrix with a given row
//	Index as always starts at 1
template <class T>
void matrix<T>::replaceRow(matrix<T> rowVec, int aRow) 
{
	if (rowVec.isRowVector())
	{
		for (size_t j = 1; j <= _cols; ++j)
		{
			insert(aRow, j, rowVec(1, j));
		}

	}
	else
	{
		throw std::logic_error("replaceRow : input Vector is not Row Vec");
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
			get_ref(i, j) = aNum;
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

/*
Create a new matrix with same number of cols , but only a single row
*/
template <class T>
matrix<T> matrix<T>::returnRow(size_t aRow) const
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

//No operation is done on the input matrix , the matrix returned is a new matrix
//remove a col and returns a matrix of same size but without the specified col
template <class T>
matrix<T> matrix<T>::removeCol(size_t aCol)
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

//No operation is done on the input matrix , the matrix returned is a new matrix
//remove a col and returns a matrix of same size but without the specified col
template <class T>
matrix<T> matrix<T>::removeRow(size_t aRow)
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


//Create a new matrix with same number of cols , but only a single row
template <class T>
matrix<T> matrix<T>::returnCol(size_t aCol) const
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
bool matrix<T>::operator==(const matrix<T> & rhs) const
{
	return this->isEqual(rhs);
}


/*
if takes the current matrix , multiplies it by the matrix on the right , and returns
a new matrix
*/
template <class T>
matrix<T>  matrix<T>::operator*(const matrix<T> & rhs) const
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
		throw std::invalid_argument("M*M -> Rows And Col Does Not Match");
	}

	return result;
}


/*
Multiply by scalar
DoesNot Modify Input Matrix
*/
template <typename T>
template <typename P>
matrix<T> matrix<T>::operator*(const P rhs) const
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

/*
template <typename T>
template <typename P>
void matrix<T>::operator*(const P rhs) 
{
	for (long long i = 1; i <= _rows; i++)
	{
		for (long long j = 1; j <= _cols; j++)
		{
			 get(i, j) *= rhs;
		}
	}
}
*/


/*
Divide by scalar
DoesNot Modify Input Matrix
*/
template <class T>
matrix<T> matrix<T>::operator/(const T rhs) const
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



template <class T>
matrix<T> matrix<T>::operator+(const  matrix<T> &rhs) const
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
		throw std::invalid_argument(" Not of same size ");
	}
}



template <class T>
matrix<T> matrix<T>::operator-(const matrix<T> &rhs) const
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
		throw std::invalid_argument(" Not of same size ");
	}
}


//------------------------------------------------------------------------------------------------

template <class T>
T& matrix<T>::operator()(const long long  rows, const long long  cols) 
{
	/*std::cout << "Safdsad";
	_matrix will have _rows and _cols // Do not worry about 0 index , that is handled
	-----------------------------------*/
	if (rows > _rows || cols > _cols)
	{
		throw std::invalid_argument(" index out of range ");
	}
	else
		return  get_ref(rows, cols) ; 
}



template <class T>
T matrix<T>::operator()(const long long  rows, const long long  cols)  const
{
	/*std::cout << "Safdsad";
	_matrix will have _rows and _cols // Do not worry about 0 index , that is handled
	-----------------------------------*/
	if (rows > _rows || cols > _cols)
	{
		throw std::invalid_argument(" index out of range ");
	}
	else
		return  get(rows, cols) ; 
}


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

template <class T>
void matrix<T>::randFillUniform(T start, T end)
{
	std::random_device rd; 
	std::mt19937 eng(rd());
	std::uniform_real_distribution<> distr(start, end);	

	std::srand(time(0));
	for (long long i = 1; i <= _rows; i++)
	{
		for (long long j = 1; j <= _cols; j++)
		{
			_matrix[(i - 1)*_cols + (j - 1)] = static_cast<T>( distr(eng) );
		}
	}
}

/*
	Creates rowVector with numbers begining at start , ending at end with interval interval ; 
	Inclusive Inclusive range 
*/
template <class T>
void matrix<T>::resizeLinSpaceRow(T start , T end,T interval)
	
{	
	double size = (end - start) / interval;
	resize(1 , size + 1);
	auto iter = begin();
	for(double i = start ; i <= end ; i+= interval ) // Condition only depend on start and end 
	{
		 *iter = i ; 
		 ++iter;
	}
}

/*
	Creates a colVector with numbers begining at start , ending at end with interval interval ; 
	Inclusive Inclusive range 
*/
template <class T>
void matrix<T>::resizeLinSpaceCol(T start , T end,T interval)
	
{	
	std::size_t size = (end - start) / interval;
	resize(size + 1,1); 
	auto iter = begin();
	for(double i = start ; i <= end ; i+= interval ) // Condition only depend on start and end 
	{
		 *iter = i ; 
		 ++iter;
	}
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
			out << std::setw(1) << temp(i, j);
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
matrix<T> matrix<T>::transpose() const
{
	matrix<T> R(numCols(),numRows()); 
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
T matrix<T>::innerProduct(const matrix<T>& B) const
{
	T result = 0 ; 
	
	if(isRowVector() && B.isColVector() && (numCols() == B.numRows()))
	{
		for(size_t  i = 1 ; i <= numRows();i++)
		{	
			result += get(1,i) * B(i,1);
		}
		return result;
	}
	else if(isRowVector() && B.isRowVector() && (numCols() == B.numCols()))
	{
		for(size_t i = 1 ; i <= numCols();i++)
		{	
			result += get(1,i) * B(1,i);
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
		throw std::logic_error("innerProduct -> A and B are not in proper format ");
		return result*(0);
	}
}

template <class T>
T matrix<T>::selfInnerProduct()
{
	return innerProduct(*this);	
}


/*
	Returns Euclidean or L2 norm of a matrix 
*/
template <class T>
T matrix<T>::normEuclidean()
{
	T innerProd = selfInnerProduct();
	return std::sqrt(innerProd);
}


#endif