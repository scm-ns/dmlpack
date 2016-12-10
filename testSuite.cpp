#include "testSuite.h"


testSuite::testSuite()
{
}


testSuite::~testSuite()
{
}


bool testSuite::runTestSuite()
{
	if (!testMatrix())
	{
		pass = false;
		error("matrix");
	}
	if (!testHessian())
	{
		pass = false;
		error("hessian");
	}
	if (!testJacobian())
	{
		pass = false;
		error("Jacobian");
	}
	if (!testLU())
	{
		pass = false;
		error("LU");
	}
	if (!testFilter())
	{
		pass = false;
		error("Filter");
	}
	if (!testQuaternion())
	{
		pass = false;
		error("Quaternion");
	}
	
	return pass;
}


bool testSuite::testHessian()
{
	
	matDouble N(2, 1);
	N(1, 1) = 1; 
	N(2, 1) = 1; 




	matDouble A = analysis.hessian_ND(func.FN__N_x3_y3_xy, N);
	if (debugMessages) A.print();
	N(1, 1) = 10;
	N(2, 1) = 1;
	A = analysis.hessian_ND(func.FN__N_x3_y3, N);
	if (debugMessages) A.print();
	A = analysis.hessian_ND(func.FN__N_x3_y3_x2y, N);
	if (debugMessages) A.print();
	A = analysis.hessian_ND(func.FN__N_cos_xy, N);
	if (debugMessages) A.print();
	A = analysis.hessian_ND(func.FN__N_xy2_x4_y4_x2y, N);
	if (debugMessages) A.print();

	return pass; 
}

bool testSuite::testJacobian()
{

	vectorFn_N fn(2,1);
	fn(1, 1) = [](matDouble N){ return N(1, 1) * N(2, 1); };

	fn(2, 1) = [](matDouble N){ return N(1, 1) * N(2, 1) * N(1,1); };

	matDouble N(2, 1);
	N(2, 1) = 10; 
	N(1, 1) = 10; 

	matDouble A = analysis.jacobian_ND(fn, N);
	if (debugMessages) A.print();

	A = analysis.jacobian_ND(func.get_FN_2_2__xy__x2y(), N); if (debugMessages) A.print();
	
	matDouble N3(3, 1);
	N3(1, 1) = 0.1; 
	N3(2, 1) = 0.1;
	N3(3, 1) = 0.12;
	A = analysis.jacobian_ND(func.get_FN_3_2__xy__x2y__x3(), N3); if (debugMessages)  A.print();
	A = analysis.jacobian_ND(func.get_FN_3_3__xyz__x2yz__x3z(), N3);  if (debugMessages) A.print();
	A = analysis.jacobian_ND(func.get_FN_3_3__xyz_9_xyz__x2yz_7_xyz__x3z_7_xyz(), N3); if (debugMessages)  A.print();



	return pass;
}


bool testSuite::testLU()
{
	matrix<double> A(4, 4);
	A(1, 1) = 10;
	A(1, 2) = -1;
	A(1, 3) = 2;
	A(1, 4) = 0;
	A(2, 1) = -1;
	A(2, 2) = 11;
	A(2, 3) = -1;
	A(2, 4) = 3;
	A(3, 1) = 2;
	A(3, 2) = -1;
	A(3, 3) = 10;
	A(3, 4) = -1;
	A(4, 1) = 0;
	A(4, 2) = 3;
	A(4, 3) = -1;
	A(4, 4) = 8;
	matrix<double> B(4, 1);
	B(1, 1) = 6;
	B(2, 1) = 25;
	B(3, 1) = -11;
	B(4, 1) = 15;



	//la.LUDecomposePartialPivot(E).print();


	matrix<double> LU = la_pack.LUDecomposePartialPivot(A);
	matrix<double> LU2 = la_pack.decomposeLU(A);
	LU.print();
	LU2.print();

	matrix<double> L = la_pack.getL_from_LU(LU);
	matrix<double> L2 = la_pack.getL_from_LU(LU2);
	L.print(); L2.print();
	matrix<double> U = la_pack.getU_from_LU(LU);
	matrix<double> U2 = la_pack.getU_from_LU(LU2);
	U.print(); U2.print();

	matrix<double> A2(3, 3);

	A2(1, 1) = 8;
	A2(1, 2) = 2;
	A2(1, 3) = 9;

	A2(2, 1) = 4;
	A2(2, 2) = 9;
	A2(2, 3) = 4;

	A2(3, 1) = 6;
	A2(3, 2) = 7;
	A2(3, 3) = 9;

	matrix<double> ALU = la_pack.LUDecomposePartialPivot(A2);
	la_pack.getL_from_LU(ALU).print();
	la_pack.getU_from_LU(ALU).print();

	//A2.print();
	matrix<double> A3 = L*U;
	//A3.print();
	return pass; 
}

bool testSuite::testMatrix()
{

	matrix<int> ABC = matrix<int>(2, 4);
	matrix<int> ABC_1 = matrix<int>(4, 2);
	matrix<int> ABC_2 = matrix<int>(4000, 4000);


	//ABC(1,1) = 100; 

	matrix<double> BQ(4, 4);
	BQ.randFill();

	matrix<double> CQ = BQ.removeCol(3);
	matrix<double> DQ = BQ.removeRow(1);

	if (debugMessages)
	{
		BQ.print();
		CQ.print();
		DQ.print();
		ABC.print();
	}



	
	ABC.randFill();
	//std::cout << ABC(1,1) << std::endl;
	if (debugMessages) ABC.print();
	//std::cout << ABC(1,1); std::cout << std::endl ;
	//	std::cout << ABC.numCols();
	//std::cout << ABC(1000,1000);

	matrix<int> A(3, 4);
	A.randFill();
	if (debugMessages)
	A.print();

	matrix<int> B(4, 3);
	B.randFill();
	if (debugMessages)
	B.print();

	//matrix<int> B = ABC;
	//B.print();

	matrix<int> C = A*B;
	if (debugMessages)
	C.print();
	C = C * 3;
	if (debugMessages)
	C.print();

	matrix<int> L = C.removeCol(1);
	if (debugMessages)	L.print();

	matrix<int> M = C.removeRow(2);
	if (debugMessages)	M.print();

	if (debugMessages)	C.print();

	if (debugMessages)   // WRONG , TEST DEPENDS ON THIS 
	{
		std::cout << C.isSquare() << std::endl;
		std::cout << C.isColVector() << std::endl;
		std::cout << C.isRowVector() << std::endl;
		std::cout << C.isDiagonal() << std::endl;
		std::cout << C.isUpperTriangular() << std::endl;
		std::cout << C.isLowerTriangular() << std::endl;
		std::cout << C.isDiagonallyDominant() << std::endl;

	}



	matrix<int> D = matrix<int>(3, 3);
	if (debugMessages) std::cout << D.isDiagonal() << std::endl;
	// Stress Test ; 
	//matrix<int> K (100000,100000);
	//matrix<double> F(10000, 10000);
	//matrix<double> E(3000, 4000); E.randFill();
	//E.print();

	// Running all the functions : If Errors they will show up 
	return pass;
}


bool testSuite::testFilter()
{
	filter Kalman; 

	return true; 
}

bool testSuite::testQuaternion()
{
	quaternion x(1, 2, 3, 4);
	//std::cout << x(1);
	return true; 

}