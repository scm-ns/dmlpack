#ifndef MATRIX_H
#define MATRIX_H
/*
	Martirx
	Access to rows and col normal indexed , starts from 1 , not 0
	Maximum Number of Elements is 10^8 ie 10,000 rows and 10,000 cols ;; If higer gives std::length_error

	Memory ordering : The memory is stored by filling up one 1 row before going to the next
			: ROW major 

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
#include <iterator>

// Intrsincis for sse
#include <xmmintrin.h>
#include <memory>


#ifdef DEBUG_D
#define dout std::cout << __FILE__<< " (" << __LINE__ << ") " << "DEBUG : "
#else
#define dout 0 && std::cout
#endif

/*
// custom allocator for alligned std::vector for sse
//
// Is this required ? new and new[] is supposed to give alligned memory blocks
//
//
template <typename T, std::size_t ALIGN = 16 , std::size_t BLOCK = 8>
class aligned_vec_alloc : public std::allocator<T>
{
	public:
		aligned_vec_alloc() 
		{

		}

		aligned_vec_alloc& operator=(const aligned_vec_alloc& rhs)
		{
			std::allocator<T>::operator=(rhs);
			return *this;
		}
		
		T* allocate(std::size_t n , const void* hint)
		{



		}	
}

*/


namespace matrix_op
{


	template <class T>
	class matrix
	{
	public:
		//CONSTRUCTORS
		matrix(void);
		~matrix(void);
		matrix(size_t rows , size_t cols , T val = 0);

		// Copy ctor and copy assignment ctor	
		matrix(const matrix<T>& rhs) : _rows(rhs._rows) , _cols(rhs._cols) , _size(rhs._size)
		{
			_matrix = rhs._matrix;
		};
		
		matrix<T>& operator=(const matrix<T>& rhs);
		// TO DO : add move ctor and move assignemnt ctor


		// creates a row vector
		matrix(std::initializer_list<T> l);
		
		// TO DO : Functionality to convert a row vec to col vise-versa. Also to convert row/col vec into 2d matrix // this resize method seems to do it. But check.
		void resize(size_t rows , size_t cols , T val = 0);
		

		inline typename std::vector<T>::iterator begin();
		inline typename std::vector<T>::const_iterator cbegin() const;
		inline typename std::vector<T>::iterator end();
		inline typename std::vector<T>::const_iterator cend() const;
		inline typename std::vector<T>::const_iterator constIterAtRowBegin(const size_t row_idx) const;
		inline typename std::vector<T>::iterator iterAtRowBegin(const size_t row_idx);
		inline const T* ptrAtRowBegin(const size_t row_idx) const;



		size_t  numRows() const ;
		size_t numCols() const ;
		size_t size() const ;

		void randFillUniform(T start = 0 , T end = 1000);
		void randFill(T start = 0, T end = 1000); 
		void symetricRandFill(T start = 0, T end = 1000); 


		void resizeLinSpaceRow(T start , T end,T interval);
		void resizeLinSpaceCol(T start , T end,T interval);

		void setAllNum(T aNum);
		void setAllZero(); 
		void set_all_num(T aNum);


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

		// returns the index which has the maximum value in the vector // onlys works on row or column vectors, not on matrix
		size_t arg_max();
		
		template <class P>
		matrix<T> operator*(const P rhs) const;
		template <class P>
		void operator*=(const P rhs) ;
		matrix<T> operator/(const T rhs) const;
		matrix<T> mul(const matrix<T> & rhs) const; // NOT FOR RELEASE
		matrix<T> mul_iter(const matrix<T> & rhs) const; // NOT FOR RELEASE
		matrix<T> operator*(const matrix<T> &rhs) const;
		matrix<T> add(const  matrix<T> &rhs) const; // NOT FOR RELEASE
		matrix<T> operator+(const matrix<T> &rhs) const;
		matrix<T> operator+(const T &rhs) const;
		matrix<T> operator-(const matrix<T> &rhs) const;
		
		bool operator==(const matrix<T> & rhs) const ;
		bool operator!=(const matrix<T> & rhs) const { return !(*this == rhs); }

		T& operator()(const long long  rows, const long long  cols); // not a const operation as this can be used to change the value
		T operator()(const long long  rows, const long long  cols)  const;

		matrix<T> transform_create(std::size_t rows , std::size_t cols , std::function<T(std::size_t , std::size_t , matrix<T>)> lam);
		matrix<T> transform_inplace(std::function<T(std::size_t , std::size_t , T)> lam) ;

		bool check_sse_allignment();	
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


	/*
	 * Pattern to handle providing of SSE instructions
	 * If the data type is something that supports sse. Then use sse. 
	 * For this better allignment of std::vector is required and this can be done by specializing the ctros for the supported types and then writing our on allocator
	 *
	 *
	 */

	template<typename T>
	matrix<T>::matrix(size_t rows , size_t cols , T val) : _rows(rows) , _cols(cols) , _matrix(_rows * _cols, val), _size(_rows * _cols)
	{

	};

	template<typename T>
	matrix<T>::matrix(void) : _rows(0) , _cols(0) , _size(0)
	{

	};


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
		return (_rows); }; 
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
	inline typename std::vector<T>::const_iterator matrix<T>::cbegin() const
	{
		return _matrix.cbegin();
	}


	template<typename T>	
	inline typename std::vector<T>::iterator matrix<T>::end() 
	{
		return _matrix.end();
	}
		
	template<typename T>	
	inline typename std::vector<T>::const_iterator matrix<T>::cend() const
	{
		return _matrix.cend();
	}

	 //Iterator at the begining of each of the rows
	template<typename T>	
	inline typename std::vector<T>::iterator matrix<T>::iterAtRowBegin(const size_t row_idx) 
	{
		typename std::vector<T>::iterator it = cbegin();
		std::advance(it , (row_idx * _cols));
		return it;
	}


	template<typename T>	
	inline const T* matrix<T>::ptrAtRowBegin(const size_t row_idx) const 
	{
		const T* beg_ptr = _matrix.data();	
		return beg_ptr 	+ (_cols * row_idx);
	}


	template<typename T>	
	inline typename std::vector<T>::const_iterator matrix<T>::constIterAtRowBegin(const size_t row_idx) const
	{
		typename std::vector<T>::const_iterator it = cbegin();
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
		auto vec_beg = _matrix.begin();
		auto vec_end = _matrix.end();
			
		for(auto vec_itr = vec_beg ; vec_itr < vec_end ; ++vec_itr)
		{
			*vec_itr = aNum;	

		}
	}

	/*
	template <>
	inline void matrix<int>::setAllNum(int aNum)
	{
		for (long long i = 1; i <= _rows; i++)
		{
			for (long long j = 1; j <= _cols; j++)
			{
				get_ref(i, j) = aNum;
			}
		}
	}
	*/

	template <class T>
	void matrix<T>::setAllZero()
	{
		setAllNum(0);
	}

	template <class T>
	bool matrix<T>::check_sse_allignment()
	{
		return false;
	}


	template <>
	inline bool matrix<int>::check_sse_allignment()
	{
		int* ptr = _matrix.data();
		return *ptr % __alignof__(__m128);
	}

	template <>
	inline void matrix<int>::setAllZero() 
	{
		__m128i zero_128_int = _mm_setzero_si128(); // 128i is int

		__m128i* const vec_beg = reinterpret_cast<__m128i *>(_matrix.data());

		// Two loops are required. 
		// 	1. To handle till the last of 4 elements which are alligned
		// 	2. Handle the last <=3 elements
		//
		// Ex : 
		// 	11 element vec
		//	[0 - 3]  [4 - 7]  8 9 10
		//	   1        1      2
		//
		// 	12 element vec
		//	[0 - 3]  [4 - 7]  [8 9 10 11]
		//	   1        1          1
		//
		// write zero to entire vector
		
		//  idx alligement refers to the property that there is enough 4 alligned index 
		//  TO : DO find better wording
			
		int idx_alligned_end = (_size - (_size % 4));	
		__m128i* const vec_idx_alligned_end = reinterpret_cast<__m128i *>(_matrix.data() + idx_alligned_end);


		__m128i * vec_itr = vec_beg ;

		// sse compute 
		// Zero out the alligned data
		for(; vec_itr < vec_idx_alligned_end ; ++vec_itr )
		{
			_mm_store_si128(vec_itr , zero_128_int);
		}
		
		// non sse compute
		// Zero out the non alligned last 4 data
		int* ptr = reinterpret_cast<int *>(vec_itr);
		for(std::size_t idx = 0 ; idx < (_size % 4) ; ++idx)
		{
			*ptr = 0 ; ++ptr;
		}		

	}

	template <>
	inline void matrix<float>::setAllZero()
	{
		__m128 zero_128_float = _mm_setzero_ps(); 

		float* const vec_beg = (_matrix.data());
			
		// find end of non alligend part of vec
		int idx_alligned_end = (_size - (_size % 4));	
		float* const vec_idx_alligned_end = ( _matrix.data() + idx_alligned_end );

		float* vec_itr = vec_beg ;

		// sse compute
		// Zero out the alligned data
		for(; vec_itr < vec_idx_alligned_end ; ++vec_itr )
		{
			_mm_store_ss(vec_itr , zero_128_float);
		}
		
		// non sse compute
		// Zero out the non alligned last 4 data
		for(std::size_t idx = 0 ; idx < (_size % 4) ; ++idx)
		{
			*vec_itr = 0 ; ++vec_itr;
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
		for (std::size_t i = 1; i <= _rows; i++)
		{
			for (std::size_t j = 1; j <= _cols; j++)
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
		for (std::size_t i = 1; i <= R._rows; i++)
		{
			for (std::size_t j = 1; j <= R._cols; j++)
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
		for (std::size_t i = 1; i <= R._rows; i++)
		{
			for (std::size_t j = 1; j <= R._cols; j++)
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
		for (std::size_t i = 1; i <= _rows; i++)
		{
			for (std::size_t j = 1; j <= _cols; j++)
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
	template <>
	inline matrix<int>  matrix<int>::operator*(const matrix<int> & rhs) const
	{
		matrix<int> result(_rows, rhs._cols);


			// cast to __m128i* to allow looping over the data with the sizeof(__m128i) and also for loading the data. 
			//__m128i* rhs_ptr =  reinterpret_cast<__m128i*>(rhs_mem_ptr);
			//__m128i* ptr = reinterpret_cast<__m128i*>(mem_ptr);
			// pointers into R, where data will be stored	
			__m128i* itr = reinterpret_cast<__m128i*>(R._matrix.data()); // used to loop over data
			auto vec_end_itr = R._matrix.end(); // pointer after the last element
			--vec_end_itr; // point to the last element

			__m128i* itr_end = reinterpret_cast<__m128i*>(*vec_end_itr);

			for (; itr <= itr_end  ; ++itr , ++rhs_ptr , ++ptr)
			{
				// load the data from buffer to variable. Due to loop 4 32 bit / 1 128 bit at a time
				__m128i ld = _mm_load_si128(ptr); 
				__m128i rd = _mm_load_si128(rhs_ptr);

				// _mm_add_epi32 adds the bits stored in the two 128 bit registers, as 4 32 bit intergers
				// _mm_store_si128(*p , a) stores a into 16 bit alligned mem location p
				_mm_store_si128(itr ,  _mm_add_epi32(ld , rd)); 
			}

		if(_cols != rhs._rows) throw std::invalid_argument("M*M -> Rows And Col Does Not Match");

		// Rather than indexing using idices, which takes up time due to having to calculate the index again for each iter of the loop.
		// use pointers, so that on each iter of the loop, a single +1 increment only needs to be done
		
		std::vector<int>::iterator res__along_row_ptr = result.begin();

		for (std::size_t i = 1; i <= _rows; i++)
		{
			std::vector<int>::const_iterator curr_row_iter = constIterAtRowBegin(i);

			for (std::size_t j = 1; j <= rhs._cols; j++)
			{
				for (std::size_t k = 1; k <= _cols; k++)
				{
					// increment of k leads to increment of pointers into the two memory blocks in different ways
					
					std::vector<int>::const_iterator rhs_row_begin_iter = rhs.constIterAtRowBegin(k);
					// get the jth item in the row of rhs	
					rhs_row_begin_iter += j;		

					//result(i, j) += get(i, k) * rhs(k, j);
					*res__along_row_ptr += *curr_row_iter  + * rhs_row_begin_iter;
					++curr_row_iter; // move along the current row k times
				}
				++res__along_row_ptr; // fills up the first column before moving to the next
			}
		}

		return result;
	}
	*/


	/*
		it takes the current matrix , multiplies it by the matrix on the right , and returns
		a new matrix
	*/
	template <class T>
	matrix<T>  matrix<T>::mul(const matrix<T> & rhs) const
	{
		matrix<T> result(_rows, rhs._cols);
		if (_cols == rhs._rows)
		{
			for (std::size_t i = 1; i <= _rows; i++)
			{
				for (std::size_t j = 1; j <= rhs._cols; j++)
				{
					for (std::size_t k = 1; k <= _cols; k++)
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

	// Currently mul is faster than op* by about 10 - 20% for large matrices ( 100 * 100), but op* is faster for smaller matrices (10 * 10)
	// The slow down might be due to using the iterators instead of raw pointers.
	// std advance being used in the iterator calculation is the prime suspect (iter at row begin)
	template <class T>
	matrix<T>  matrix<T>::mul_iter(const matrix<T> & rhs) const // NOT FOR RELEASE
	{
		if(_cols != rhs._rows) throw std::invalid_argument("M*M -> Rows And Col Does Not Match");

		matrix<T> result(_rows, rhs._cols);

		// Rather than indexing using indices, which takes up time due to having to calculate the index again for each iter of the loop.
		// use pointers, so that on each iter of the loop, a single +1 increment only needs to be done
		
		auto res__along_row_ptr = result.begin();
		typename std::vector<T>::const_iterator curr_row_iter;
		typename std::vector<T>::const_iterator rhs_row_begin_iter ;
		std::size_t i = 0 , j = 0 , k = 0;
		for (i = 1; i <= _rows; i++)
		{
			curr_row_iter = constIterAtRowBegin(i);

			for (j = 1; j <= rhs._cols; j++)
			{
				for (k = 1; k <= _cols; k++)
				{
					// increment of k leads to increment of pointers into the two memory blocks in different ways
					// get the jth item in the row of rhs	

					//result(i, j) += get(i, k) * rhs(k, j);
					*res__along_row_ptr += (*curr_row_iter)  * (* (rhs.constIterAtRowBegin(k) + j) ); 
					++curr_row_iter; // move along the current row k times
				}
				++res__along_row_ptr; // fills up the first column before moving to the next
			}
		}

		return result;
	}


	template <class T>
	matrix<T>  matrix<T>::operator*(const matrix<T> & rhs) const // NOT FOR RELEASE
	{
		if(_cols != rhs._rows) throw std::invalid_argument("M*M -> Rows And Col Does Not Match");

		matrix<T> result(_rows, rhs._cols);

		// Rather than indexing using indices, which takes up time due to having to calculate the index again for each iter of the loop.
		// use pointers, so that on each iter of the loop, a single +1 increment only needs to be done
		
		T* res__along_row_ptr = const_cast<T*>(result._matrix.data());
		const T* curr_row_iter;
		std::size_t i = 0 , j = 0 , k = 0;
		for (i = 1; i <= _rows; i++)
		{
			curr_row_iter = ptrAtRowBegin(i);

			for (j = 1; j <= rhs._cols; j++)
			{
				for (k = 1; k <= _cols; k++)
				{
					// increment of k leads to increment of pointers into the two memory blocks in different ways
					// get the jth item in the row of rhs	

					//result(i, j) += get(i, k) * rhs(k, j);
					*res__along_row_ptr += (*curr_row_iter)  * (* (rhs.ptrAtRowBegin(k) + j) ); 
					++curr_row_iter; // move along the current row k times
				}
				++res__along_row_ptr; // fills up the first column before moving to the next
			}
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

	// Inplace scalar multiply
	// TO DO : Restrict P to scalar values 
	template <typename T>
	template <typename P>
	void matrix<T>::operator*=(const P rhs) 
	{
		for (long long i = 1; i <= _rows; i++)
		{
			for (long long j = 1; j <= _cols; j++)
			{
				 get_ref(i, j) *= rhs;
			}
		}
	}


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
	matrix<T> matrix<T>::add(const  matrix<T> &rhs) const // NOT FOR RELEASE
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


	/*
	 * Adds a scalar value to the matrix. 
	 * Adds the same value to each element of the matrix
	 * Not strictly mathematical, but helps with using the matrix
	 */
	template <class T>
	matrix<T> matrix<T>::operator+(const T &rhs) const
	{
		matrix<T> res(numRows() , numCols());
		for(std::size_t row = 1 ; row <= numRows() ; ++row)
		{
			for(std::size_t col = 1 ; col <= numCols() ; ++col)
			{
				res(row , col) =  get(row, col) + rhs;
			}
		}
		return res;

	}


	template <class T>
	matrix<T> matrix<T>::operator+(const  matrix<T> &rhs) const
	{
		matrix<T> R(_rows, _cols);
		if (_rows == rhs._rows && _cols == rhs._cols)
		{
			// Rather than indexing using idices, which takes up time due to having to calculate the index again for each iter of the loop.
			// use pointers, so that on each iter of the loop, a single +1 increment only needs to be done
			auto rhs_ptr =  rhs.cbegin();
			auto ptr = cbegin();
			auto ret_ptr = R.begin();

			for (; ret_ptr != R.end() ; ++ptr , ++rhs_ptr, ++ret_ptr)
			{
				*ret_ptr = *ptr + *rhs_ptr;		
			}

			return R;
		}
		else
		{
			throw std::invalid_argument(" Not of same size ");
		}
	}


	template <>
	inline matrix<int> matrix<int>::operator+(const  matrix<int> &rhs) const
	{
		matrix<int> R(_rows, _cols);
		if (_rows == rhs._rows && _cols == rhs._cols)
		{
			// Rather than indexing using idices, which takes up time due to having to calculate the index again for each iter of the loop.
			// use pointers, so that on each iter of the loop, a single +1 increment only needs to be done
			
			// get the start address of rhs	memory block
			int* rhs_mem_ptr = const_cast<int*>(rhs._matrix.data()); // cast away const for simplicity. Guarantee to make sure rhs is not modified
			// get start addr of `this` matrix memory block
			int* mem_ptr = const_cast<int*>(_matrix.data());

			// cast to __m128i* to allow looping over the data with the sizeof(__m128i) and also for loading the data. 
			__m128i* rhs_ptr =  reinterpret_cast<__m128i*>(rhs_mem_ptr);
			__m128i* ptr = reinterpret_cast<__m128i*>(mem_ptr);
		
			// pointers into R, where data will be stored	
			__m128i* itr = reinterpret_cast<__m128i*>(R._matrix.data()); // used to loop over data
			auto vec_end_itr = R._matrix.end(); // pointer after the last element
			--vec_end_itr; // point to the last element

			__m128i* itr_end = reinterpret_cast<__m128i*>(*vec_end_itr);

			for (; itr <= itr_end  ; ++itr , ++rhs_ptr , ++ptr)
			{
				// load the data from buffer to variable. Due to loop 4 32 bit / 1 128 bit at a time
				__m128i ld = _mm_load_si128(ptr); 
				__m128i rd = _mm_load_si128(rhs_ptr);

				// _mm_add_epi32 adds the bits stored in the two 128 bit registers, as 4 32 bit intergers
				// _mm_store_si128(*p , a) stores a into 16 bit alligned mem location p
				_mm_store_si128(itr ,  _mm_add_epi32(ld , rd)); 
			}

			return R;
		}
		else
		{
			throw std::invalid_argument("MATRICES NOT OF SAME SIZE");
		}
	}




	// inline required to follow the one definition rule. 
	// ODR means that a definition for a class / function should only be done once in a  compilation unit or entire program
	// One of practical implications of this is that if you provide defenition for a function within the header file, then the funtion/class will have multiple definitons in
	// all the different files which includes the header. 
	//	The special cases when this does not apply are 
	//		1. the function/class is a template, then you can provide definitions within the header
	//		2. the function is inlined. { can be done by including the funcion within the class or marking with inline property.
	// So for explict tempate specializtion, inline it.
	template <>
	inline matrix<float> matrix<float>::operator+(const  matrix<float> &rhs) const 
	{

		matrix<float> R(_rows, _cols);
		if (_rows == rhs._rows && _cols == rhs._cols)
		{
			for (std::size_t i = 1; i <= _rows; i++)
			{
				for (std::size_t j = 1; j <= _cols; j++)
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
	std::ostream& operator<<(std::ostream& out, const matrix<T>& temp)
	{
		out << std::endl;
		
	//	std::copy(temp.cbegin() , temp.cend() , std::ostreambuf_iterator<T>(out , " "));


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




	
	// scalar exp
	template <typename T>
	inline T exp(T val)
	{
		return std::exp(val);
	}


	// matrix exp
	template <class T>
	matrix<T> exp(const matrix<T>& A)
	{
		matrix<T> res(A.numRows() , A.numCols());
		for(std::size_t row = 1 ; row <= A.numRows() ; ++row)
		{
			for(std::size_t col = 1 ; col <= A.numCols() ; ++col)
			{
				res(row , col) = std::exp(A(row , col));			
			}
		}

		return res;
	}


	// Constant pi
	constexpr double pi()
	{
		return std::acos(-1);
	}

	
	// scalar sin
	template <typename T>
	inline T sin(T val)
	{
		return std::sin(val);
	}


	// matrix sin
	template <class T>
	matrix<T> sin(const matrix<T>& A)
	{
		matrix<T> res(A.numRows() , A.numCols());
		for(std::size_t row = 1 ; row <= A.numRows() ; ++row)
		{
			for(std::size_t col = 1 ; col <= A.numCols() ; ++col)
			{
				res(row , col) = std::sin(A(row , col));			
			}
		}

		return res;
	}


	
	// scalar cos
	template <typename T>
	inline T cos(T val)
	{
		return std::sin(val);
	}


	// matrix cos
	template <class T>
	matrix<T> cos(const matrix<T>& A)
	{
		matrix<T> res(A.numRows() , A.numCols());
		for(std::size_t row = 1 ; row <= A.numRows() ; ++row)
		{
			for(std::size_t col = 1 ; col <= A.numCols() ; ++col)
			{
				res(row , col) = std::cos(A(row , col));			
			}
		}
		return res;
	}



	// scalar tanh
	template <typename T>
	inline T tanh(T val)
	{
		return std::tanh(val);
	}


	// matrix tanh
	template <class T>
	matrix<T> tanh(const matrix<T>& A)
	{
		matrix<T> res(A.numRows() , A.numCols());
		for(std::size_t row = 1 ; row <= A.numRows() ; ++row)
		{
			for(std::size_t col = 1 ; col <= A.numCols() ; ++col)
			{
				res(row , col) = std::tanh(A(row , col));			
			}
		}
		return res;
	}


	// scalar sigmoid
	template <typename T>
	inline T sigmoid(T val)
	{
		return 1.0/(1 + std::exp(-val));
	}


	// matrix sigmoid
	template <class T>
	matrix<T> sigmoid(const matrix<T>& A)
	{
		matrix<T> res(A.numRows() , A.numCols());
		for(std::size_t row = 1 ; row <= A.numRows() ; ++row)
		{
			for(std::size_t col = 1 ; col <= A.numCols() ; ++col)
			{
				res(row , col) = matrix_op::sigmoid(A(row , col));			
			}
		}
		return res;
	}



	// scalar relu
	template <typename T>
	inline T relu(T val)
	{
		return std::max(static_cast<T>(0) , val);
	}

	// matrix relu
	template <class T>
	matrix<T> relu(const matrix<T>& A)
	{
		matrix<T> res(A.numRows() , A.numCols());
		for(std::size_t row = 1 ; row <= A.numRows() ; ++row)
		{
			for(std::size_t col = 1 ; col <= A.numCols() ; ++col)
			{
				res(row , col) = matrix_op::relu(A(row , col));			
			}
		}
		return res;
	}



	// scalar leaky_relu
	template <typename T>
	inline T leaky_relu(T val,float slope=0.01)
	{
		if( val > static_cast<T>(0) )
		{
			return val;
		}
		else
		{
			return  val * static_cast<T>(slope);
		}
	}

	// matrix leaky_relu
	template <class T>
	matrix<T> leaky_relu(const matrix<T>& A, float slope = 0.01)
	{
		matrix<T> res(A.numRows() , A.numCols());
		for(std::size_t row = 1 ; row <= A.numRows() ; ++row)
		{
			for(std::size_t col = 1 ; col <= A.numCols() ; ++col)
			{
				res(row , col) = matrix_op::leaky_relu(A(row , col), slope);			
			}
		}
		return res;
	}


	
	// TODO : Think about sse implementation and about writing a helper fuction for iteration over the sse blocks
	// TODO : create vector sum, where by we can add in the case of a 2D matrix, along the rows giving back a col vec or else along the columns giving back a row vec
	/*
	 * Takes the sum of the matrix
	 * If the matrix is row vector, adds up along the row
	 * If matrix is a column vector, adds up along the column
	 * If matrix is 2D , then adds up along both axis
	 */	
	template <class T>
	T sum(const matrix<T>& A)
	{
		T sum = 0; 

		if (A.isColVector())
		{
			for (size_t i = 1; i <= A.numRows(); i++) // Go through each row and add them up 
			{
				sum += A(i, 1);
			}
		}
		else if (A.isRowVector())
		{
			for (size_t i = 1; i <= A.numCols(); i++) // Go through each row and add them up 
			{
				sum += A(1,i);
			}

		}
		else // matrix is a normal matrix , fat or thin 
		{
			for (size_t i = 1; i <= A.numRows(); i++)
			{
				for (size_t j = 1; j <= A.numCols(); j++)
				{
					sum += A(i, j);
				}
			}
		}
		return sum; 
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
	T inner_product(const matrix<T>& A , const matrix<T>& B) 
	{
		T result = 0 ; 
		
		if(A.isRowVector() && B.isColVector() && (A.numCols() == B.numRows()))
		{
			for(size_t  i = 1 ; i <= A.numRows();i++)
			{	
				result += A(1,i) * B(i,1);
			}
			return result;
		}
		else if(A.isRowVector() && B.isRowVector() && (A.numCols() == B.numCols()))
		{
			for(size_t i = 1 ; i <= A.numCols();i++)
			{	
				result += A(1,i) * B(1,i);
			}
			return result;
		}
		else if(A.isColVector() && B.isColVector() && (A.numRows() == B.numRows()))
		{
			for(size_t i = 1 ; i <= A.numRows();i++)
			{	
				result += A(i,1)*B(i,1);
			}
			return result;
		}
		else
		{
			throw std::logic_error("innerProduct -> A and B are not in proper format ");
		}
	}


	/*
		Returns Euclidean or L2 norm of a matrix 
	*/
	template <class T>
	T norm_euclidean(const matrix<T>& A)
	{
		T inner_prod = inner_product(A,A);
		return std::sqrt(inner_prod);
	}


	//Returns a rowVector with numbers begining at start , ending at end with interval interval ; 
	//Inclusive Inclusive range 
	template<class T>
	matrix<T> linspace_row(T start,T end, T interval)
	{	
		std::size_t size = (end - start) / interval;
		matrix<T> R(1,size+1); 
		for(std::size_t i = start , k = 1; i <= end ; i+= interval,k+=1) // Condition only depend on start and end 
		{
			R(1,k) = i ; 
		}
		return R ; 
	}


	template <class T> // Make the generic implementation similar to that used for ints. This way, long, long long and smaller int types like uint8 are supported
	matrix<T> rand_fill(std::size_t rows , std::size_t cols , T low, T high)
	{
		matrix<T> R(rows , cols);
		std::srand(time(0));
		for (std::size_t i = 1; i <= R.numRows(); i++)
		{
			for (std::size_t j = 1; j <= R.numCols() ;  j++)
			{
				R(i , j) = static_cast<T>( ( static_cast<T>(std::rand())  % static_cast<T>( high - low )  + 1  ) + low ) ; // bias towards low if rand not divisble by (high - low) + 1
			}
		}
		return R;
	}


	template <>
	inline matrix<float> rand_fill(std::size_t rows , std::size_t cols , float low, float high)
	{
		matrix<float> R(rows , cols);
		std::srand(time(0));
		for (std::size_t i = 1; i <= R.numRows(); i++)
		{
			for (std::size_t j = 1; j <= R.numCols() ;  j++)
			{
				R(i , j) = ( static_cast<float>(std::rand())  / static_cast<float>(RAND_MAX/ (high - low ))  ) + low ; // bias towards low if rand not divisble by (high - low) + 1
			}
		}
		return R;
	}


	template <>
	inline matrix<double> rand_fill(std::size_t rows , std::size_t cols , double low, double high)
	{
		matrix<double> R(rows , cols);
		std::srand(time(0));
		for (std::size_t i = 1; i <= R.numRows(); i++)
		{
			for (std::size_t j = 1; j <= R.numCols() ;  j++)
			{
				R(i , j) = ( static_cast<double>(std::rand())  / static_cast<double>(RAND_MAX/ (high - low ))  ) + low ; // bias towards low if rand not divisble by (high - low) + 1
			}
		}
		return R;
	}




} // end of namespace matrix_op






#endif
