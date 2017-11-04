#define _wassert wassert_awf
#include <cassert>

#include <iostream>
#include <iomanip>

#include <random>
#include <ctime>

#include <Eigen/Eigen>

#include "Logger.h"

#include "Optimization/BAFunctor.h"

#include "BlockDiagonalSparseQR_Ext.h"
#include "BandedBlockedSparseQR_Ext.h"

#define OUTPUT_MAT 1

typedef SparseMatrix<Scalar, ColMajor, SuiteSparse_long> JacobianType;
typedef SparseMatrix<Scalar, RowMajor, SuiteSparse_long> JacobianTypeRowMajor;
typedef Matrix<Scalar, Dynamic, Dynamic> MatrixType;
typedef BandedBlockedSparseQR<JacobianType, NaturalOrdering<int>, 8, false> BandedBlockedQRSolver;

// QR for J1 is banded blocked QR
//typedef BandedBlockedSparseQR<JacobianType, NaturalOrdering<SparseDataType>, 2, false> LeftSuperBlockSolver;
//typedef SPQR<JacobianType> LeftSuperBlockSolver;
// QR for J1'J2 is general dense (faster than general sparse by about 1.5x for n=500K)
//typedef ColPivHouseholderQR<Matrix<Scalar, Dynamic, Dynamic> > RightSuperBlockSolver;
//typedef BandedBlockedSparseQR<JacobianType, NaturalOrdering<SparseDataType>, 2, false> RightSuperBlockSolver;
//typedef SPQR<JacobianType> RightSuperBlockSolver;
//typedef ColPivHouseholderQR<Matrix<Scalar, Dynamic, Dynamic> > RightSuperBlockSolver;
// QR solver is concatenation of the above.
//typedef BlockAngularSparseQR<JacobianType, LeftSuperBlockSolver, RightSuperBlockSolver> SchurlikeQRSolver;
typedef ColPivHouseholderQR<Matrix<Scalar, Dynamic, Dynamic> > DenseBlockSolver;
typedef BlockDiagonalSparseQR_Ext<JacobianType, DenseBlockSolver> LeftSuperBlockSolver;
//typedef BlockDiagonalSparseQR_Ext<JacobianType, DenseBlockSolver> RightSuperBlockSolver;
typedef BandedBlockedSparseQR<JacobianType, NaturalOrdering<SparseDataType>, 2, false> RightSuperBlockSolver;
typedef BlockAngularSparseQR_Ext<JacobianType, LeftSuperBlockSolver, RightSuperBlockSolver> SchurlikeQRSolver;
typedef SchurlikeQRSolver QRSolver;

typedef SPQR<JacobianType> SPQRSolver;

/*
* Generate block diagonal sparse matrix.
*/
void generate_block_diagonal_matrix(const Eigen::Index numLeftBlockParams, const Eigen::Index leftBlockParamDim, Eigen::Index &numResiduals, JacobianType &spJ) {
	std::default_random_engine gen;
	std::uniform_real_distribution<double> dist(0.5, 5.0);
	std::uniform_int_distribution<int> distStride(10, 20);
	//std::uniform_int_distribution<int> distDense(4, 10);

	numResiduals = 0;
	Eigen::TripletArray<Scalar, typename JacobianType::Index> jvals(numLeftBlockParams * numResiduals * 0.1);
	int stride = distStride(gen);
	int lastRow = 0;
	for (int i = 0; i < numLeftBlockParams; i += leftBlockParamDim) {
		for (int j = i; j < i + leftBlockParamDim; j++) {
			for (int k = 0; k < stride; k++) {
				jvals.add(numResiduals + k, j, dist(gen));
				lastRow = numResiduals + k;
			}
		}
		numResiduals += stride;
		stride = distStride(gen);
	}
	int denseRows = numResiduals;
	for (int j = 0; j < numLeftBlockParams; j++) {
		for (int k = 0; k < denseRows; k++) {
			jvals.add(numResiduals + k, j, dist(gen));
		}
	}
	numResiduals += denseRows;

	spJ.resize(numResiduals, numLeftBlockParams);
	spJ.setFromTriplets(jvals.begin(), jvals.end());
	spJ.makeCompressed();

	// Permute Jacobian right block rows (we want to see how our QR handles a general matrix)	
	if (0) {
		PermutationMatrix<Dynamic, Dynamic, SuiteSparse_long> perm(spJ.rows());
		perm.setIdentity();
		std::random_shuffle(perm.indices().data(), perm.indices().data() + perm.indices().size());
		spJ = perm * spJ;
	}
}

int main() {
	Logger::createLogger("runtime_log.log");
	Logger::instance()->log(Logger::Info, "QR Bundle Adjustment Test STARTED!");

	std::default_random_engine gen;
	std::uniform_real_distribution<double> dist(0.5, 5.0);

	Eigen::Index nPoints = 384;//512;//1024;//64;// // 50000;//512;
	Eigen::Index nPointDim = 6;//3;
	Eigen::Index nBlockParams = nPoints * nPointDim;

	Eigen::Index nResiduals = 0;

	clock_t begin;


	/*//////////////////////////////
	MatrixType mat = MatrixType::Random(256000, 650 * 4);
	MatrixType vec = MatrixType::Random(256000, 1);
	begin = clock();
	for (int i = 0; i < 2600; i++) {
		vec = vec * mat.col(i);
	}
	std::cout << "Multry:   " << double(clock() - begin) / CLOCKS_PER_SEC << "s\n";
	

	MatrixType mat = MatrixType::Random(2600, 150000);
	begin = clock();
	ColPivHouseholderQR<MatrixType> houseqr;
	houseqr.compute(mat);
	std::cout << "Thintry 1 block:   " << double(clock() - begin) / CLOCKS_PER_SEC << "s\n";
	
	begin = clock();
	HouseholderQR<MatrixType> houseqr2;
	houseqr2.compute(mat);
	std::cout << "Thintry 1 block:   " << double(clock() - begin) / CLOCKS_PER_SEC << "s\n";
	/*
	std::cout << "thintrystart" << std::endl;
	begin = clock();
	for (int i = 0; i < 650; i++) {
		HouseholderQR<MatrixType> houseqr;
		houseqr.compute(mat.block<256000, 4>(0, i * 4));
	}
	std::cout << "Thintry:   " << double(clock() - begin) / CLOCKS_PER_SEC << "s\n";
	///////////////////////////////////*/



	JacobianType spJ;
	generate_block_diagonal_matrix(nBlockParams, nPointDim, nResiduals, spJ);

	typedef BandedBlockedSparseQR_Ext<JacobianType, NaturalOrdering<SuiteSparse_long>, 2, false> BandedBlockedQRSolver;
	BandedBlockedQRSolver bbqr;
	begin = clock();
	bbqr.compute(spJ);
	std::cout << "bbqr.compute(spJ) ... " << double(clock() - begin) / CLOCKS_PER_SEC << "s" << std::endl;
	JacobianType XXXrp = bbqr.rowsPermutation() * spJ;
	JacobianType I(XXXrp.rows(), XXXrp.rows());
	I.setIdentity();
	JacobianType bbqrQ(XXXrp.rows(), XXXrp.rows());
	bbqrQ = bbqr.matrixQ() * I;
	std::cout << "||Q    * R - J||_2 = " << (bbqrQ * bbqr.matrixR() - XXXrp).norm() << std::endl;
	//std::cout << "||Q.T  * J - R||_2 = " << (slvrQ.transpose() * spJRowPerm - slvr.matrixR()).norm() << std::endl;		
	Logger::instance()->logMatrixCSV(bbqrQ.toDense(), "BBQR_Q.csv");
	Logger::instance()->logMatrixCSV(bbqr.matrixR().toDense(), "BBQR_R.csv");
	Logger::instance()->logMatrixCSV(spJ.toDense(), "BBQR.csv");
	Logger::instance()->logMatrixCSV(bbqr.rowsPermutation().inverse().eval().indices(), "BBQR_rpi_inv.csv");
	Logger::instance()->logMatrixCSV(bbqr.rowsPermutation().indices(), "BBQR_rpi.csv");

	return 0;
}

// Override system assert so one can set a breakpoint in it rather than clicking "Retry" and "Break"
void __cdecl _wassert(_In_z_ wchar_t const* _Message, _In_z_ wchar_t const* _File, _In_ unsigned _Line)
{
	std::wcerr << _File << "(" << _Line << "): ASSERT FAILED [" << _Message << "]\n";

	abort();
}
