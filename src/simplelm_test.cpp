#define _wassert wassert_awf
#include <cassert>

#include <iostream>
#include <iomanip>
#include <ctime>

#include <future>

#include <random>

#include <Eigen/Eigen>
#include <Eigen/SparseQR>

#include <Eigen/SparseCore>

#include "BlockAngularSparseQR_Ext.h"
#include "BlockDiagonalSparseQR_Ext.h"
#include "DenseBlockedThinSparseQR.h"
#include "Optimization/SimpleLM.h"

#include <unsupported/Eigen/MatrixFunctions>

#include "Logger.h"

using namespace Eigen;

typedef int IndexType;

typedef SparseMatrix<Scalar, ColMajor, IndexType> JacobianType;
typedef Matrix<Scalar, Dynamic, Dynamic> MatrixType;

template <typename _Scalar>
struct EllipseFitting : SparseLMFunctor<_Scalar, IndexType>
{
	// Class data: 2xN matrix with each column a 2D point
	Matrix2Xd ellipsePoints;

	// Number of parameters in the model, to which will be added
	// one latent variable per point.
	static const int nParamsModel = 5;

	// Constructor initializes points, and tells the base class how many parameters there are in total
	EllipseFitting(const Matrix2Xd& points) :
		SparseLMFunctor<_Scalar, IndexType>(nParamsModel + points.cols(), points.cols() * 2),
		ellipsePoints(points)
	{
	}

	// Functor functions
	int operator()(const InputType& uv, ValueType& fvec) const {
		// Ellipse parameters are the last 5 entries
		auto params = uv.tail(nParamsModel);
		double a = params[0];
		double b = params[1];
		double x0 = params[2];
		double y0 = params[3];
		double r = params[4];

		// Correspondences (t values) are the first N
		for (int i = 0; i < ellipsePoints.cols(); i++) {
			double t = uv(i);
			double x = a*cos(t)*cos(r) - b*sin(t)*sin(r) + x0;
			double y = a*cos(t)*sin(r) + b*sin(t)*cos(r) + y0;
			fvec(2 * i + 0) = ellipsePoints(0, i) - x;
			fvec(2 * i + 1) = ellipsePoints(1, i) - y;
		}

		return 0;
	}

	// Functor jacobian
	int df(const InputType& uv, JacobianType& fjac) {
		// X_i - (a*cos(t_i) + x0)
		// Y_i - (b*sin(t_i) + y0)
		int npoints = ellipsePoints.cols();
		auto params = uv.tail(nParamsModel);
		double a = params[0];
		double b = params[1];
		double r = params[4];

		TripletArray<JacobianType::Scalar, IndexType> triplets(npoints * 2 * 5); // npoints * rows_per_point * nonzeros_per_row
		for (int i = 0; i < npoints; i++) {
			double t = uv(i);
			triplets.add(2 * i, i, +a*cos(r)*sin(t) + b*sin(r)*cos(t));
			triplets.add(2 * i, npoints + 0, -cos(t)*cos(r));
			triplets.add(2 * i, npoints + 1, +sin(t)*sin(r));
			triplets.add(2 * i, npoints + 2, -1);
			triplets.add(2 * i, npoints + 4, +a*cos(t)*sin(r) + b*sin(t)*cos(r));

			triplets.add(2 * i + 1, i, +a*sin(r)*sin(t) - b*cos(r)*cos(t));
			triplets.add(2 * i + 1, npoints + 0, -cos(t)*sin(r));
			triplets.add(2 * i + 1, npoints + 1, -sin(t)*cos(r));
			triplets.add(2 * i + 1, npoints + 3, -1);
			triplets.add(2 * i + 1, npoints + 4, -a*cos(t)*cos(r) + b*sin(t)*sin(r));
		}

		fjac.setFromTriplets(triplets.begin(), triplets.end());
		return 0;
	}
};

template <typename _Scalar>
struct SparseBlockDiagonalQR_EllipseFitting : public EllipseFitting<_Scalar> {
	// QR for J1 subblocks is 2x1
	typedef ColPivHouseholderQR<Matrix<Scalar, Dynamic, Dynamic> > DenseBlockSolver;
	typedef BlockDiagonalSparseQR_Ext<JacobianType, DenseBlockSolver> LeftSuperBlockSolver;
	//typedef DenseBlockedThinSparseQR<JacobianType, NaturalOrdering<IndexType>, 5, true> RightSuperBlockSolver;
	//typedef HouseholderQR<Matrix<Scalar, Dynamic, Dynamic> > RightSuperBlockSolver;
	typedef DenseBlockSolver RightSuperBlockSolver;
	// QR for J is concatenation of the above.
	typedef BlockAngularSparseQR_Ext<JacobianType, LeftSuperBlockSolver, RightSuperBlockSolver> SchurlikeQRSolver;

	typedef SchurlikeQRSolver QRSolver;


	SparseBlockDiagonalQR_EllipseFitting(const Matrix2Xd& points) :
		EllipseFitting<_Scalar>(points) {
	}

	// And tell the algorithm how to set the QR parameters.
	void initQRSolver(SchurlikeQRSolver &qr) {
		// set block size
		qr.getLeftSolver().setPattern(ellipsePoints.cols() * 2 + ellipsePoints.cols(), ellipsePoints.cols(), 3, 1);
		//qr.getLeftSolver().setSparseBlockParams(16, 8);
		qr.setSparseBlockParams(ellipsePoints.cols() * 2 + ellipsePoints.cols(), ellipsePoints.cols());
	}
};

typedef EllipseFitting<Scalar>::InputType ParamsType;

void printParamsHeader() {
	std::cout << "a \t";
	std::cout << "b \t";
	std::cout << "x0\t";
	std::cout << "y0\t";
	std::cout << "r \t";
	std::cout << "Duration";
	std::cout << std::endl;
}

void printParams(ParamsType &params, int nDataPoints, double duration = -1.) {
	/*
	std::cout << "a=" << params(nDataPoints) << "\t";
	std::cout << "b=" << params(nDataPoints + 1) << "\t";
	std::cout << "x0=" << params(nDataPoints + 2) << "\t";
	std::cout << "y0=" << params(nDataPoints + 3) << "\t";
	std::cout << "r=" << params(nDataPoints + 4)*180. / EIGEN_PI << "\t";
	*/
	std::cout << params(nDataPoints) << "\t";
	std::cout << params(nDataPoints + 1) << "\t";
	std::cout << params(nDataPoints + 2) << "\t";
	std::cout << params(nDataPoints + 3) << "\t";
	std::cout << params(nDataPoints + 4)*180. / EIGEN_PI << "\t";
	if (duration >= 0) {
		std::cout << duration << "s";
	}
	std::cout << std::endl;
}

void initializeParams(int nDataPoints, const Matrix2Xd &ellipsePoints, double incr, ParamsType &params) {
	params.resize(EllipseFitting<Scalar>::nParamsModel + nDataPoints);
	double minX, minY, maxX, maxY;
	minX = maxX = ellipsePoints(0, 0);
	minY = maxY = ellipsePoints(1, 0);
	for (int i = 0; i<ellipsePoints.cols(); i++) {
		minX = (std::min)(minX, ellipsePoints(0, i));
		maxX = (std::max)(maxX, ellipsePoints(0, i));
		minY = (std::min)(minY, ellipsePoints(1, i));
		maxY = (std::max)(maxY, ellipsePoints(1, i));
	}
	params(ellipsePoints.cols()) = 0.5*(maxX - minX);
	params(ellipsePoints.cols() + 1) = 0.5*(maxY - minY);
	params(ellipsePoints.cols() + 2) = 0.5*(maxX + minX);
	params(ellipsePoints.cols() + 3) = 0.5*(maxY + minY);
	params(ellipsePoints.cols() + 4) = 0;
	for (int i = 0; i<ellipsePoints.cols(); i++) {
		params(i) = Scalar(i)*incr;
	}
}

#define LM_VERBOSE 0

const int NumTests = 9;
int NumSamplePoints[NumTests] = { 10, 500, 1000, 2000, 5000, 10000, 50000, 100000, 500000 };

int main() {
	//eigen_assert(false);

	// _CrtSetDbgFlag(_CRTDBG_CHECK_ALWAYS_DF);

	/***************************************************************************/
	std::cout << "###################### Ellipse fitting test #########################" << std::endl;
	std::cout << "#####################################################################" << std::endl;
	std::cout << "N - Number of data points" << std::endl;
	std::cout << "Bl Diag Sp QR - Block Diagonal Sparse QR" << std::endl;
	std::cout << "#####################################################################" << std::endl;
	for (int i = 0; i < NumTests; i++) {
		// Create the ellipse paramteers and data points
		// ELLIPSE PARAMETERS
		double a, b, x0, y0, r;
		a = 7.5;
		b = 2;
		x0 = 17.;
		y0 = 23.;
		r = 0.23;

		std::cout << "N = " << NumSamplePoints[i] << "   \t";
		printParamsHeader();
		std::cout << "=====================================================================" << std::endl;

		// CREATE DATA SAMPLES
		int nDataPoints = NumSamplePoints[i];
		Matrix2Xd ellipsePoints;
		ellipsePoints.resize(2, nDataPoints);
		Scalar incr = 1.3*EIGEN_PI / Scalar(nDataPoints);
		for (int i = 0; i<nDataPoints; i++) {
			Scalar t = Scalar(i)*incr;
			ellipsePoints(0, i) = x0 + a*cos(t)*cos(r) - b*sin(t)*sin(r);
			ellipsePoints(1, i) = y0 + a*cos(t)*sin(r) + b*sin(t)*cos(r);
		}

		// INITIAL PARAMS
		ParamsType params;
		initializeParams(nDataPoints, ellipsePoints, incr, params);

		/***************************************************************************/
		// Run the optimization problem
		std::cout << "Initialization:" << "\t";
		printParams(params, nDataPoints);
		std::cout << "Ground Truth:" << "\t";
		std::cout << a << "\t";
		std::cout << b << "\t";
		std::cout << x0 << "\t";
		std::cout << y0 << "\t";
		std::cout << r*180. / EIGEN_PI << "\t";
		std::cout << std::endl;
		std::cout << "---------------------------------------------------------------------" << std::endl;

		clock_t begin;
		double duration;
		initializeParams(nDataPoints, ellipsePoints, incr, params);
		typedef SparseBlockDiagonalQR_EllipseFitting<Scalar>  SparseBlockDiagonalQRFunctor;
		Eigen::SimpleLMInfo::Status info;
		SparseBlockDiagonalQRFunctor functor(ellipsePoints);
		Eigen::SimpleLM< SparseBlockDiagonalQRFunctor, true > lm(functor);
		//lm3.setVerbose(LM_VERBOSE);
		begin = clock();
		info = lm.minimize(params);
		duration = double(clock() - begin) / CLOCKS_PER_SEC;
		std::cout << "Bl Diag Sp QR:\t";
		printParams(params, nDataPoints, duration);

		std::cout << "#####################################################################" << std::endl;

		std::cout << "LM finished with status: " << SimpleLMInfo::statusToString(info) << std::endl;

	//	break;

		/*
		// check parameters ambiguity before test result
		// a should be bigger than b
		if (fabs(params(ellipsePoints.cols() + 1)) > fabs(params(ellipsePoints.cols()))) {
		std::swap(params(ellipsePoints.cols()), params(ellipsePoints.cols() + 1));
		params(ellipsePoints.cols() + 4) -= 0.5*EIGEN_PI;
		}
		// a and b should be positive
		if (params(ellipsePoints.cols())<0) {
		params(ellipsePoints.cols()) *= -1.;
		params(ellipsePoints.cols() + 1) *= -1.;
		params(ellipsePoints.cols() + 4) += EIGEN_PI;
		}
		// fix rotation angle range
		while (params(ellipsePoints.cols() + 4) < 0) params(ellipsePoints.cols() + 4) += 2.*EIGEN_PI;
		while (params(ellipsePoints.cols() + 4) > EIGEN_PI) params(ellipsePoints.cols() + 4) -= EIGEN_PI;


		eigen_assert(fabs(a - params(ellipsePoints.cols())) < 0.00001);
		eigen_assert(fabs(b - params(ellipsePoints.cols() + 1)) < 0.00001);
		eigen_assert(fabs(x0 - params(ellipsePoints.cols() + 2)) < 0.00001);
		eigen_assert(fabs(y0 - params(ellipsePoints.cols() + 3)) < 0.00001);
		eigen_assert(fabs(r - params(ellipsePoints.cols() + 4)) < 0.00001);
		*/
	}

	return 0;
}

// Override system assert so one can set a breakpoint in it rather than clicking "Retry" and "Break"
void __cdecl _wassert(_In_z_ wchar_t const* _Message, _In_z_ wchar_t const* _File, _In_ unsigned _Line)
{
	std::wcerr << _File << "(" << _Line << "): ASSERT FAILED [" << _Message << "]\n";

	abort();
}
