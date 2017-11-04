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

#include "TallSkinnyQR.h"

#include "DenseBlockedThinSparseQR.h"

#include "DenseBlockedThinQR.h"

//#define OUTPUT_MAT 1

typedef SuiteSparse_long SpDataType;
//typedef int SpDataType;

typedef SparseMatrix<Scalar, ColMajor, SpDataType> JacobianType;
typedef SparseMatrix<Scalar, RowMajor, SpDataType> JacobianTypeRowMajor;
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
//typedef BandedBlockedSparseQR<JacobianType, NaturalOrdering<SparseDataType>, 2, false> RightSuperBlockSolver;
//typedef DenseBlockedThinSparseQR<JacobianType, NaturalOrdering<SparseDataType>, 10, true> RightSuperBlockSolver;
typedef DenseBlockedThinQR<MatrixType, NaturalOrdering<SparseDataType>, 10, true> RightSuperBlockSolver;
typedef BlockAngularSparseQR_Ext<JacobianType, LeftSuperBlockSolver, RightSuperBlockSolver> SchurlikeQRSolver;
typedef SchurlikeQRSolver QRSolver;

typedef SPQR<JacobianType> SPQRSolver;

typedef HouseholderQR<Matrix<Scalar, Dynamic, Dynamic> > HouseQRSolver;
typedef TallSkinnyQR<JacobianType, HouseQRSolver> TSQR;

typedef CholmodSimplicialLLT<JacobianType, Upper> CholeskySolver;

/*
* Generate block diagonal sparse matrix.
*/
void generate_block_diagonal_matrix(const Eigen::Index numLeftBlockParams, const Eigen::Index leftBlockParamDim, const Eigen::Index numRightBlockParams, const Eigen::Index rightBlockParamDim, Eigen::Index &numResiduals, JacobianType &spJ) {
	std::default_random_engine gen;
	std::uniform_real_distribution<double> dist(0.5, 5.0);
	std::uniform_int_distribution<int> distStride(4, 8);
	std::uniform_int_distribution<int> distCam(0, numRightBlockParams / rightBlockParamDim - 1);

	numResiduals = 0;
	Eigen::TripletArray<Scalar, typename JacobianType::Index> jvals(numLeftBlockParams * numResiduals * 0.1 + numRightBlockParams * numResiduals * 0.1);
	int stride = distStride(gen);
	int lastRow = 0;
	for (int i = 0; i < numLeftBlockParams; i += leftBlockParamDim) {
		for (int j = i; j < i + leftBlockParamDim; j++) {
			// For each row add point measurements
			for (int k = 0; k < stride; k++) {
				jvals.add(numResiduals + k, j, dist(gen));
				lastRow = numResiduals + k;
			}
		}

		// For each row select one camera and add its parameters
		std::vector<int> selCam;
		for (int k = 0; k < stride; k++) {
			selCam.push_back(distCam(gen));
		}
		std::sort(selCam.begin(), selCam.end());
		// Fill selected camera params
		for (int k = 0; k < stride; k++) {
			int start = selCam[k] * rightBlockParamDim;
			for (int j = start; j < start + rightBlockParamDim; j++) {
				jvals.add(numResiduals + k, numLeftBlockParams + j, dist(gen));
			}
		}
		numResiduals += stride;
		stride = distStride(gen);
	}
	/*
	int rightStride = ceil(numResiduals / (numRightBlockParams / rightBlockParamDim));
	int rightPos = 0;
	for (int i = 0; i < numRightBlockParams; i += rightBlockParamDim) {
		for (int j = i; j < i + rightBlockParamDim; j++) {
			for (int k = 0; k < rightStride; k++) {
				jvals.add(rightPos + k, numLeftBlockParams + j, dist(gen));
				lastRow = rightPos + k;
			}
		}
		rightPos += rightStride;
	}*/

	spJ.resize(numResiduals, numLeftBlockParams + numRightBlockParams);
	spJ.setFromTriplets(jvals.begin(), jvals.end());
	spJ.makeCompressed();

	// Permute Jacobian right block rows (we want to see how our QR handles a general matrix)	
	if (0) {
		PermutationMatrix<Dynamic, Dynamic, SpDataType> perm(spJ.rows());
		perm.setIdentity();
		std::random_shuffle(perm.indices().data(), perm.indices().data() + perm.indices().size());
		spJ.rightCols(numRightBlockParams) = perm * spJ.rightCols(numRightBlockParams);
	}
}

void generate_expected_R_with_diag(const Eigen::Index numLeftBlockParams, const Eigen::Index leftBlockParamDim, const Eigen::Index numRightBlockParams, const Eigen::Index rightBlockParamDim, Eigen::Index &numResiduals, JacobianTypeRowMajor &spJ) {
	std::default_random_engine gen;
	std::uniform_real_distribution<double> dist(0.5, 5.0);
	std::uniform_int_distribution<int> distIdxs(0, numRightBlockParams - 1);

	numResiduals = 0;
	Eigen::TripletArray<Scalar, typename JacobianTypeRowMajor::Index> jvals(numLeftBlockParams * (numLeftBlockParams + numRightBlockParams) * 0.01);
	int stride = leftBlockParamDim;

	for (int i = 0; i < numLeftBlockParams + numRightBlockParams; i += leftBlockParamDim) {
		for (int j = i; j < i + leftBlockParamDim; j++) {
			// For each row add point measurements
			for (int k = 0; k < stride; k++) {
				if (k <= j - i) {
					jvals.add(numResiduals + k, j, dist(gen));
				}
			}
		}

		// Right block is going to be 50% dense upper triangular
		std::vector<int> idxsj(numRightBlockParams / 2);
		for (int k = 0; k < stride; k++) {
			if (numResiduals + k < numLeftBlockParams) {
				std::generate(idxsj.begin(), idxsj.end(), [&]() { return distIdxs(gen); });
				for (auto idxj = idxsj.begin(); idxj != idxsj.end(); ++idxj) {
					jvals.add(numResiduals + k, numLeftBlockParams + *idxj, dist(gen));
				}
				//for (int j = 0; j < numRightBlockParams; j++) {
					
				//}
			} else {
				for (int j = 0; j < numRightBlockParams; j++) {
					if(numResiduals + k <= numLeftBlockParams + j) {
						jvals.add(numResiduals + k, numLeftBlockParams + j, dist(gen));
					}
				}
			}

		}
		numResiduals += stride;
		stride = 3;
	}

	for (int i = 0; i < numLeftBlockParams + numRightBlockParams; i++) {
		jvals.add(numResiduals, i, dist(gen));
		numResiduals += 1;
	}

	/*
	int rightStride = ceil(numResiduals / (numRightBlockParams / rightBlockParamDim));
	int rightPos = 0;
	for (int i = 0; i < numRightBlockParams; i += rightBlockParamDim) {
	for (int j = i; j < i + rightBlockParamDim; j++) {
	for (int k = 0; k < rightStride; k++) {
	jvals.add(rightPos + k, numLeftBlockParams + j, dist(gen));
	lastRow = rightPos + k;
	}
	}
	rightPos += rightStride;
	}*/

	spJ.resize(numResiduals, numLeftBlockParams + numRightBlockParams);
	spJ.setFromTriplets(jvals.begin(), jvals.end());
	spJ.makeCompressed();
}

int main() {
	Logger::createLogger("runtime_log.log");
	Logger::instance()->log(Logger::Info, "QR Bundle Adjustment Test STARTED!");

	clock_t begin;

	Eigen::MatrixXd t1;
	JacobianType t2;

	std::default_random_engine gen;
	std::uniform_real_distribution<double> dist(0.5, 5.0);

	Eigen::Index nPoints = 72000;//72000;//5120;// 512;//40000;//512;//40000;//1536;//45000;//768 * 7;//50000;//512;
	Eigen::Index nPointDim = 3;
	Eigen::Index nCams = 261;//12;//260;//12;//150;//3 * 2;//224;//35;
	Eigen::Index nCamDim = 10;
	Eigen::Index nLeftBlockParams = nPoints * nPointDim;
	Eigen::Index nRightBlockParams = nCams * nCamDim;

	Eigen::Index nResiduals = 0;
	/*
	MatrixType matt = MatrixType::Random(32000, 320);
	clock_t begin;
	HouseholderQR<MatrixType> houseqr2;
	begin = clock();
	houseqr2.compute(matt);
	std::cout << "Thintry 1 block:   " << double(clock() - begin) / CLOCKS_PER_SEC << "s\n";
	*/
	/*
	MatrixType Y = MatrixType(150000, 32);
	MatrixType T = MatrixType(32, 32);
	MatrixType matt2 = MatrixType(150000, 2048);
	MatrixType hvec = MatrixType::Random(150000, 1);
	MatrixType matt = MatrixType::Random(150000, 32);
	begin = clock();
	for (int i = 0; i < 384; i++) {
		matt2.col(i) += (Y * (T * (Y.transpose() * matt2.col(i))));
	}
	std::cout << "Multvec:   " << double(clock() - begin) / CLOCKS_PER_SEC << "s\n";
	//mat = mat.triangularView<Upper>();
	HouseholderQR<MatrixType> houseqr2;
	begin = clock();
	houseqr2.compute(matt);
	std::cout << "Thintry 1 block:   " << double(clock() - begin) / CLOCKS_PER_SEC << "s\n";
	/*
	begin = clock();
	matt2 = houseqr2.matrixQ().transpose() * matt2;
	std::cout << "Thintry 1 block Qmult:   " << double(clock() - begin) / CLOCKS_PER_SEC << "s\n";
	//*/

#ifdef EIGEN_USE_BLAS
	std::cout << "EIGEN_USE_BLAS defined" << std::endl;
#endif
	/*
	//Eigen::MatrixXd mat2 = Eigen::MatrixXd::Identity(48, 16);
	Eigen::MatrixXd mat2 = Eigen::MatrixXd::Identity(256000, 2560);
	//Eigen::MatrixXd matH = Eigen::MatrixXd::Identity(256000, 2560);
	//matH.block(mat2.rows() / 2, 0, mat2.rows() / 2, mat2.cols()) = -1 * Eigen::MatrixXd::Ones(mat2.rows() / 2, mat2.cols());

	//mat2 = mat2;
	
	//Eigen::MatrixXd mat2 = Eigen::MatrixXd::Identity(25600, 256);
	Index numDenseRows = 32000;
	Index startDenseRow = mat2.rows() - numDenseRows;
	mat2.block(startDenseRow, 0, numDenseRows, mat2.cols()) = (Eigen::MatrixXd::Random(numDenseRows, mat2.cols()).array() > 0.25).cast<double>() * Eigen::MatrixXd::Random(numDenseRows, mat2.cols()).array();
	//mat2.block(mat2.cols(), 0, mat2.rows() - mat2.cols(), mat2.cols()) = (Eigen::MatrixXd::Random(mat2.rows() - mat2.cols(), mat2.cols()).array() > 0.25).cast<double>() * Eigen::MatrixXd::Random(mat2.rows() - mat2.cols(), mat2.cols()).array();
	//mat2.block(mat2.rows() / 2, 0, mat2.rows() / 2, mat2.cols()) = (Eigen::MatrixXd::Random(mat2.rows() / 2, mat2.cols()).array() > 0.25).cast<double>() * Eigen::MatrixXd::Random(mat2.rows() / 2, mat2.cols()).array();
	//mat2.block(mat2.rows() / 2, 0, mat2.rows() / 2, mat2.cols()) = Eigen::MatrixXd::Random(mat2.rows() / 2, mat2.cols());
	//Logger::instance()->logMatrixCSV(mat2, "mat.csv");
	
	typedef DenseSubblockBlockedSparseQR<JacobianType, NaturalOrdering<SparseDataType>, 8, false> DenseSubblockBQR;
	
	DenseSubblockBQR dsbqr;
	//dsbqr.setDenseBlockStartRow(mat2.rows() / 2);
	dsbqr.setDenseBlockStartRow(startDenseRow);
	begin = clock();
	dsbqr.compute(mat2.sparseView(), true);
	std::cout << "DenseSubblockBQR:   " << double(clock() - begin) / CLOCKS_PER_SEC << "s\n";
	//Logger::instance()->logMatrixCSV(dsbqr.matrixR().toDense(), "new_R.csv");
	//*/


	/*//////////////////////////////
	MatrixType mat = MatrixType::Random(256000, 650 * 4);
	MatrixType vec = MatrixType::Random(256000, 1);
	begin = clock();
	for (int i = 0; i < 2600; i++) {
		vec = vec * mat.col(i);
	}
	std::cout << "Multry:   " << double(clock() - begin) / CLOCKS_PER_SEC << "s\n";


	begin = clock();
	ColPivHouseholderQR<MatrixType> houseqr;
	houseqr.compute(mat.block<256000, 4>(0, 0));
	std::cout << "Thintry 1 block:   " << double(clock() - begin) / CLOCKS_PER_SEC << "s\n";

	begin = clock();
	HouseholderQR<MatrixType> houseqr2;
	houseqr2.compute(mat.block<256000, 4>(0, 0));
	std::cout << "Thintry 1 block:   " << double(clock() - begin) / CLOCKS_PER_SEC << "s\n";
	
	std::cout << "thintrystart" << std::endl;
	begin = clock();
	for(int i = 0; i < 650; i++) {
		HouseholderQR<MatrixType> houseqr;
		houseqr.compute(mat.block<256000, 4>(0, i * 4));
	}
	std::cout << "Thintry:   " << double(clock() - begin) / CLOCKS_PER_SEC << "s\n";
	///////////////////////////////////*/

	/*
	* Set-up the problem to be solved
	*/
	// Generate the sparse matrix
	begin = clock();
	JacobianTypeRowMajor spJ;
	generate_expected_R_with_diag(nLeftBlockParams, nPointDim, nRightBlockParams, nCamDim, nResiduals, spJ);
	std::cout << "Generate J:   " << double(clock() - begin) / CLOCKS_PER_SEC << "s\n";
	/*
	TSQR tsqrSolver;
	std::cout << "Input size: " << spJ.rightCols(nRightBlockParams).rows() << ", " << spJ.rightCols(nRightBlockParams).cols() << std::endl;
	begin = clock();
	tsqrSolver.compute(spJ.rightCols(nRightBlockParams));
	std::cout << "TSQR compute:   " << double(clock() - begin) / CLOCKS_PER_SEC << "s\n";

	return 0;
	*/
	/*
	Eigen::MatrixXd mat2 = Eigen::MatrixXd::Identity(7168, 2560);
	mat2.block(mat2.rows() / 2, 0, mat2.rows() / 2, mat2.cols()) = (Eigen::MatrixXd::Random(mat2.rows() / 2, mat2.cols()).array() > 0.25).cast<double>() * Eigen::MatrixXd::Random(mat2.rows() / 2, mat2.cols()).array();
	Logger::instance()->logMatrixCSV(mat2, "mat.csv");
	Eigen::MatrixXd mat = mat2;
	//Eigen::MatrixXd mat = spJ.rightCols(nRightBlockParams).toDense();
	Eigen::MatrixXd M, Q, R;
	Eigen::VectorXd D;
	Eigen::MatrixXcd sincos;
	std::cout << mat.rows() << ", " << mat.cols() << std::endl;
	begin = clock();
	givens(mat, sincos, R);
	std::cout << "Givens QR:   " << double(clock() - begin) / CLOCKS_PER_SEC << "s\n";
	begin = clock();
	givens_Q(sincos, Q);
	std::cout << "Givens Q expr:   " << double(clock() - begin) / CLOCKS_PER_SEC << "s\n";
	//fast_givens(mat, D, M, R);
	//Logger::instance()->logMatrixCSV(Q, "givens_Q.csv");
	Logger::instance()->logMatrixCSV(R, "givens_R.csv");


	return 0;
	//*/

#if !defined(_DEBUG) && defined(OUTPUT_MAT)
	Logger::instance()->logMatrixCSV(spJ.toDense(), "slvrJ.csv");
#endif
	/*
	begin = clock();
	typedef PermutationMatrix<Dynamic, Dynamic, SpDataType> PermutationType;
	PermutationType rowPerm;
	Eigen::Matrix<JacobianType::StorageIndex, Dynamic, 1> rowpermIndices(spJ.rows());
	
	for (JacobianType::StorageIndex i = 0; i < spJ.cols(); i++) {
		rowpermIndices(i) = i * 2;
		rowpermIndices(i + spJ.cols()) = i * 2 + 1;
	}
	
	for (JacobianType::StorageIndex i = 0; i < nLeftBlockParams / nPointDim; i++) {
		rowpermIndices(i * nPointDim) = i * nPointDim * 2;
		rowpermIndices(i * nPointDim + 1) = i * nPointDim * 2 + 1;
		rowpermIndices(i * nPointDim + 2) = i * nPointDim * 2 + 2;
		rowpermIndices(i * nPointDim + spJ.cols()) = i * nPointDim * 2 + 3;
		rowpermIndices(i * nPointDim + spJ.cols() + 1) = i * nPointDim * 2 + 4;
		rowpermIndices(i * nPointDim + spJ.cols() + 2) = i * nPointDim * 2 + 5;
	}
	for (JacobianType::StorageIndex i = 0; i < nRightBlockParams / nCamDim; i++) {
		for (int j = 0; j < nCamDim; j++) {
			rowpermIndices(nLeftBlockParams + i * nCamDim + j) = nLeftBlockParams * 2 + i * nCamDim * 2 + j;
		}
		for (int j = 0; j < nCamDim; j++) {
			rowpermIndices(nLeftBlockParams + i * nCamDim + spJ.cols() + j) = nLeftBlockParams * 2 + i * nCamDim * 2 + nCamDim + j;
		}
	}
	rowPerm = PermutationType(rowpermIndices);
	std::cout << "CreatePerm:   " << double(clock() - begin) / CLOCKS_PER_SEC << "s\n";
	begin = clock();
	spJ = rowPerm * spJ;
	std::cout << "Permute:   " << double(clock() - begin) / CLOCKS_PER_SEC << "s\n";
	*/
	
	begin = clock();
	// Permutation is slow - try to create the Jacobian from memory instead
	JacobianTypeRowMajor pspJ(spJ.rows(), spJ.cols());
	pspJ.reserve(spJ.nonZeros());
	Index blocksz = 1;
	Index stride = blocksz * 2;
	for (int i = 0; i < spJ.cols() / blocksz; i+=1) {
		for(int j = 0; j < blocksz; j++) {
			pspJ.startVec(i * stride + j);
			// Insert i-th row of the upper triangular part (upper half)
			for (JacobianTypeRowMajor::InnerIterator rowIt(spJ, i * blocksz + j); rowIt; ++rowIt) {
				pspJ.insertBack(i * stride + j, rowIt.index()) = rowIt.value();
			}
		}
		// Insert i-th row of the diagonal part (lower half)
		for(int j = blocksz; j < stride; j++) {
			pspJ.startVec(i * stride + j);
			for (JacobianTypeRowMajor::InnerIterator rowIt(spJ, i * blocksz + spJ.cols() + j - blocksz); rowIt; ++rowIt) {
				pspJ.insertBack(i * stride + j, rowIt.index()) = rowIt.value();
			}
		}
	}
	pspJ.finalize();
	spJ = pspJ;
	std::cout << "Interleave:   " << double(clock() - begin) / CLOCKS_PER_SEC << "s\n";

	//return 0;

#if !defined(_DEBUG) && defined(OUTPUT_MAT)
	Logger::instance()->logMatrixCSV(spJ.toDense(), "slvrJperm.csv");
#endif

	// Auxiliary identity matrix (for later use)
	JacobianType I(spJ.rows(), spJ.rows());
	I.setIdentity();

	std::cout << "####################################################" << std::endl;
	std::cout << "Problem size (r x c): " << spJ.rows() << " x " << spJ.cols() << std::endl;
	std::cout << "####################################################" << std::endl;
	
	/*
	CholeskySolver chol;
	begin = clock();
	JacobianType JtJ = spJ.transpose() * spJ;
	std::cout << "J.T * J:   " << double(clock() - begin) / CLOCKS_PER_SEC << "s\n";
	begin = clock();
	//chol.compute(spJ.rightCols(2600).bottomRows(spJ.rows() / 2).transpose() * spJ.rightCols(2600).bottomRows(spJ.rows() / 2));
	chol.compute(JtJ);
	std::cout << "Cholesky Factorization:   " << double(clock() - begin) / CLOCKS_PER_SEC << "s\n";
	return 0;
	//*/

	//SPQRSolver spqr;
	//begin = clock();
	//spqr.compute(spJ);
	//std::cout << "SPQR Factorization:   " << double(clock() - begin) / CLOCKS_PER_SEC << "s\n";

	//return 0;

	/*
	* Solve the problem using the banded blocked QR solver.
	*/
	///*
	std::cout << "Solver: Sparse Banded Blocked QR" << std::endl;
	std::cout << "---------------------- Timing ----------------------" << std::endl;
	
	QRSolver slvr;
	//slvr.getLeftSolver().setSparseBlockParams(6, 3);
	slvr.setSparseBlockParams(nResiduals - nRightBlockParams * 2, nLeftBlockParams);
	slvr.getLeftSolver().setPattern(nResiduals - nRightBlockParams * 2, nLeftBlockParams, nPointDim * 2, nPointDim);
	//slvr.getLeftSolver().setPattern(nResiduals - nRightBlockParams * 2, nLeftBlockParams, nPointDim * 4, nPointDim * 2);

	// 1) Factorization
	begin = clock();
	slvr.compute(spJ);
	std::cout << "Factorization:   " << double(clock() - begin) / CLOCKS_PER_SEC << "s\n";


	return 0;
//	Logger::instance()->logMatrixCSV(slvr.matrixR().toDense(), "slvrR.csv");

	// 2) Benchmark expressing full Q 
	// Q * I
	std::cout << "Express full Q: " << std::endl;

	begin = clock();
	JacobianTypeRowMajor slvrQ(spJ.rows(), spJ.rows());
	slvrQ = slvr.matrixQ() * I;
	std::cout << "matrixQ()   * I: " << double(clock() - begin) / CLOCKS_PER_SEC << "s\n";
//	Logger::instance()->logMatrixCSV(slvrQ.toDense(), "slvrQ.csv");

	
	// Q.T * I
	begin = clock();
	JacobianTypeRowMajor slvrQt(spJ.rows(), spJ.rows());
	slvrQt = slvr.matrixQ().transpose() * I;
	std::cout << "matrixQ().T * I: " << double(clock() - begin) / CLOCKS_PER_SEC << "s\n";
	//Logger::instance()->logMatrixCSV(slvrQt.toDense(), "slvrQtranspose.csv");
	
	// 4) Apply computed row reordering
	//JacobianType spJRowPerm = (slvr.rowsPermutation() * spJ);
	
	JacobianType spJRowPerm = spJ;// slvr.rowperm1() * spJ;
	//spJRowPerm = slvr.rowperm2() * spJRowPerm;
	spJRowPerm = spJRowPerm * slvr.colsPermutation();
	// 5) Show statistics and residuals
//	Logger::instance()->logMatrixCSV(slvr.colsPermutation().indices(), "colPerm.csv");
	//Logger::instance()->logMatrixCSV(slvr.matrixR().toDense(), "slvrR.csv");
//	Logger::instance()->logMatrixCSV(spJ.toDense(), "slvrJ.csv");
	std::cout << "---------------------- Stats -----------------------" << std::endl;
	std::cout << "Q non-zeros: " << slvrQ.nonZeros() << " (" << (slvrQ.nonZeros() / double(slvrQ.rows() * slvrQ.cols())) * 100 << "%)" << std::endl;
	std::cout << "---------------------- Errors ----------------------" << std::endl;
	std::cout << "||Q    * R - J||_2 = " << (slvrQ * slvr.matrixR() - spJRowPerm).norm() << std::endl;
	std::cout << "||Q.T  * J - R||_2 = " << (slvrQt * spJRowPerm - slvr.matrixR()).norm() << std::endl;
	//std::cout << "||Qt.T * R - J||_2 = " << (slvrQt.transpose() * slvr.matrixR() - spJRowPerm).norm() << std::endl;
	//std::cout << "||Qt   * J - R||_2 = " << (slvrQt * spJRowPerm - slvr.matrixR()).norm() << std::endl;
	//std::cout << "||Q.T  * Q - I||_2 = " << (slvrQ.transpose() * slvrQ - I).norm() << std::endl;
	std::cout << "####################################################" << std::endl;
	//*/

	return 0;
}

// Override system assert so one can set a breakpoint in it rather than clicking "Retry" and "Break"
void __cdecl _wassert(_In_z_ wchar_t const* _Message, _In_z_ wchar_t const* _File, _In_ unsigned _Line)
{
	std::wcerr << _File << "(" << _Line << "): ASSERT FAILED [" << _Message << "]\n";

	abort();
}
