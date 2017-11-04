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

#define OUTPUT_MAT 1

typedef int SpDataType;

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
typedef DenseBlockedThinSparseQR<JacobianType, NaturalOrdering<SparseDataType>, 10, true> RightSuperBlockSolver;
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

void fast_giv_2x2(const Eigen::Vector2d &x, Eigen::Vector2d &d, double alpha, double beta, bool type) {
	if (x(1) != 0) {
		alpha = -x(0) / x(1);
		beta = -alpha * d(1) / d(0);
		double gamma = -alpha * beta;
		if (gamma <= 1) {
			type = true;
			double tau = d(0);
			d(0) = (1. + gamma) * d(1);
			d(1) = (1. + gamma) * tau;
		} else {
			type = false;
			alpha = 1. / alpha;
			beta = 1. / beta;
			gamma = 1. / gamma;
			d = (1. + gamma) * d;
		}
	}
	else {
		type = false;
		alpha = 0; 
		beta = 0;
	}
}

void fast_givens(const Eigen::MatrixXd &mat, Eigen::VectorXd &matD, Eigen::MatrixXd &matM, Eigen::MatrixXd &matR) {
	matR = mat;

	// Set M to identity at the start
	matM.resize(mat.rows(), mat.rows());
	matM.setIdentity();

	// Vector D represents diagonal matrix D
	matD.resize(mat.rows());
	matD.setIdentity();

	clock_t begin, beginn, infork;
	double alpha, beta, gamma, tau;
	bool type;
	std::cout << mat.rows() << ", " << mat.cols() << std::endl;
	for (int c = 0; c < mat.cols(); c++) {
		begin = clock();
		infork = 0;
		//std::cout << "------------- " << c << std::endl;
		for (int r = mat.rows() - 1; r > c; r--) {
			//std::cout << r << ", ";
			// If the entry is nonzero, process
			//if (abs(mat(r, c)) > 1e-16) {
			//if (matR(r, c) != 0) {
				
				// Perform fast givens
				if (matR(r, c) != 0) {
					alpha = -matR(c, c) / matR(r, c);
					beta = -alpha * matD(r) / matD(c);
					gamma = -alpha * beta;
					if (gamma <= 1) {
						tau = matD(c);
						matD(c) = (1. + gamma) * matD(r);
						matD(r) = (1. + gamma) * tau;

						// type == 1
						for (int k = c; k < mat.cols(); k++) {
							double ck = matR(c, k);
							double rk = matR(r, k);
							matR(c, k) = beta * ck + rk;
							matR(r, k) = ck + alpha * rk;
							//matR(c, k) = ck + beta * rk;
							//matR(r, k) = alpha * ck + rk;
						}
					}
					else {
						alpha = 1. / alpha;
						beta = 1. / beta;
						gamma = 1. / gamma;
						matD(c) *= (1. + gamma);
						matD(r) *= (1. + gamma);

						// type == 2
						for (int k = c; k < mat.cols(); k++) {
							double ck = matR(c, k);
							double rk = matR(r, k);
							matR(c, k) = ck + alpha * rk;
							matR(r, k) = beta * ck + rk;
							//matR(c, k) = beta * ck + rk;
							//matR(r, k) = ck + alpha * rk;
						}
					}
				}

				// Apply the givens rotation
				/*
				beginn = clock();
				for (int k = c; k < mat.cols(); k++) {
					double ck = matR(c, k);
					double rk = matR(r, k);
					matR(c, k) = co * ck + si * rk;
					matR(r, k) = -si * ck + co * rk;

					double qck = matQ(c, k);
					double qrk = matQ(r, k);
					matQ(c, k) = co * qck + si * qrk;
					matQ(r, k) = -si * qck + co * qrk;
				}
				infork += clock() - beginn;
				*/

				// Store si and co ... 
				// FixMe: jasvob
		//	}
		}
		std::cout << "for c(" << c << "):   " << double(clock() - begin) / CLOCKS_PER_SEC << "s\n";
		std::cout << " ... in for k:   " << double(infork) / CLOCKS_PER_SEC << "s\n";
		//std::cout << std::endl;
	}
}

//std::vector<Eigen::MatrixXd> Gs;
//Eigen::VectorXd v1 = 
//JacobianType spQ;
void givens(const Eigen::MatrixXd &mat, Eigen::MatrixXcd &sincos, Eigen::MatrixXd &matR) {
	matR = mat;

	sincos = Eigen::MatrixXcd::Zero(mat.rows(), mat.cols());

	//spQ.resize(mat.rows(), mat.rows());
	//spQ.setIdentity();

	//clock_t begin, beginn, infork;
	for (int c = 0; c < mat.cols(); c++) {
	//	begin = clock();
	//	infork = 0;
		//std::cout << "------------- " << c << std::endl;
		/*
		JacobianType G(mat.rows(), mat.rows());
		G.setZero();
		G.reserve(4);
		G.startVec(c);
		G.insertBack(c, c) = co;
		G.insertBack(c, r) = si;
		G.startVec(r);
		G.insertBack(r, c) = -si;
		G.insertBack(r, r) = co;
		G.finalize();
		Gs.push_back(G);
		*/
		for (int r = mat.rows() - 1; r > c; r--) {
		//for (int r = c + 1; r < mat.rows(); r++) {
				//std::cout << r << ", ";
			// If the entry is nonzero, process
			//if (abs(mat(r, c)) > 1e-16) {
			if (matR(r, c) != 0) {
				// Determin sin and cos
				int r2 = c;//r - 1;
				double tmp = std::sqrt(matR(r2, c) * matR(r2, c) + matR(r, c) * matR(r, c));

				if (matR(r, c) < 0) {
					tmp = -tmp;
				}

				double si = matR(r, c) / tmp;
				double co = matR(r2, c) / tmp;

				// Apply the givens rotation
			//	beginn = clock();
				for (int k = c; k < mat.cols(); k++) {
					double ck = matR(r2, k);
					double rk = matR(r, k);
					matR(r2, k) = co * ck + si * rk;
					matR(r, k) = -si * ck + co * rk;
				}
			//	infork += clock() - beginn;

//				Eigen::MatrixXd G(mat.rows()), mat.rows());

				// Store si and co ... 
				// FixMe: jasvob
				sincos(r, c) = std::complex<double>(si, co);
			}

		}
	//	std::cout << "for c(" << c << "):   " << double(clock() - begin) / CLOCKS_PER_SEC << "s\n";
	//	std::cout << " ... in for k:   " << double(infork) / CLOCKS_PER_SEC << "s\n";
		//std::cout << std::endl;
	}
}

void givens_Q(const Eigen::MatrixXcd &sincos, Eigen::MatrixXd &matQ) {

	Index dimQ = std::max(sincos.rows(), sincos.cols());
	matQ = Eigen::MatrixXd::Identity(dimQ, dimQ);
	
	double si, co;
	for (int c = sincos.cols() - 1; c >= 0; c--) {
		for (int r = sincos.rows() - 1; r > c; r--) {

			if (sincos(r, c).real() != 0 || sincos(r, c).imag() != 0) {
				si = sincos(r, c).real();
				co = sincos(r, c).imag();

				for (int k = 0; k < sincos.cols(); k++) {
					double ck = matQ(c, k);
					double rk = matQ(r, k);
					matQ(c, k) = co * ck - si * rk;
					matQ(r, k) = si * ck + co * rk;
				}
			}
		}
	}
}

int main() {
	Logger::createLogger("runtime_log.log");
	Logger::instance()->log(Logger::Info, "QR Bundle Adjustment Test STARTED!");

	clock_t begin;

	Eigen::MatrixXd t1;
	JacobianType t2;

	std::default_random_engine gen;
	std::uniform_real_distribution<double> dist(0.5, 5.0);

	Eigen::Index nPoints = 512;//5120;// 512;//40000;//512;//40000;//1536;//45000;//768 * 7;//50000;//512;
	Eigen::Index nPointDim = 3;
	Eigen::Index nCams = 12;//12;//260;//12;//150;//3 * 2;//224;//35;
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
	JacobianType spJ;
	generate_block_diagonal_matrix(nLeftBlockParams, nPointDim, nRightBlockParams, nCamDim, nResiduals, spJ);
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
//	Logger::instance()->logMatrixCSV(spJ.toDense(), "slvrJ.csv");
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
	*/
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
	slvr.setSparseBlockParams(nResiduals, nLeftBlockParams);

	// 1) Factorization
	begin = clock();
	slvr.compute(spJ);
	std::cout << "Factorization:   " << double(clock() - begin) / CLOCKS_PER_SEC << "s\n";

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
	/*td::cout << "||Qt.T * R - J||_2 = " << (slvrQt.transpose() * slvr.matrixR() - spJRowPerm).norm() << std::endl;
	std::cout << "||Qt   * J - R||_2 = " << (slvrQt * spJRowPerm - slvr.matrixR()).norm() << std::endl;
	//std::cout << "||Q.T  * Q - I||_2 = " << (slvrQ.transpose() * slvrQ - I).norm() << std::endl;
	std::cout << "####################################################" << std::endl;
	//*/
	//*/

	return 0;
}

// Override system assert so one can set a breakpoint in it rather than clicking "Retry" and "Break"
void __cdecl _wassert(_In_z_ wchar_t const* _Message, _In_z_ wchar_t const* _File, _In_ unsigned _Line)
{
	std::wcerr << _File << "(" << _Line << "): ASSERT FAILED [" << _Message << "]\n";

	abort();
}
