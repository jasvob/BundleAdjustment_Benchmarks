#define _wassert wassert_awf
#include <cassert>

#include <iostream>
#include <iomanip>

#include <random>
#include <ctime>

#include <Eigen/Eigen>

#include "BATypeUtils.h"
#include "Logger.h"
#include "Utils.h"
#include "MathUtils.h"

#include "Optimization/BAFunctor.h"
#include "CameraMatrix.h"
#include "DistortionFunction.h"

#include <unsupported/Eigen/NumericalDiff>
#include "Eigen_ext/BacktrackLevMarqMore.h"
#include "Eigen_ext/BacktrackLevMarqCholesky.h"
#include "Eigen_ext/BacktrackLevMarqQRChol.h"

enum ReturnCodes {
	Success = 0,
	WrongInputParams = 1,
	WrongInputFile = 2,

};

typedef BAFunctor OptimizationFunctor;

const Scalar AVG_FOCAL_LENGTH = 1.0;
const Scalar INLIER_THRESHOLD = 0.5;

using namespace Eigen;

int main(int argc, char * argv[]) {
	Logger::createLogger("runtime_log.log");
	Logger::instance()->log(Logger::Info, "Computation STARTED!");
		
	/***************** Check input parameters *****************/
	if (argc != 2) {
		std::cerr << "Usage: " << argv[0] << " <sparse reconstruction file>" << std::endl;
		return ReturnCodes::WrongInputParams;
	}

	std::ifstream ifs(argv[1]);
	if (!ifs) {
		std::cerr << "Cannot open " << argv[1] << std::endl;
		return ReturnCodes::WrongInputFile;
	}

	/***************** Read input data from file *****************/
	const Scalar avg_focal_length = AVG_FOCAL_LENGTH;

	int N, M, K;
	ifs >> N >> M >> K;
	std::cout << "N(cameras) = " << N << ", M(points) = " << M << ", K(measurements) = " << K << std::endl;

	// Read image measurements data
	std::cout << "Reading image measurements..." << std::endl;
	Matrix2XX measurements(2, K);
	std::vector<int> correspondingView(K, -1);
	std::vector<int> correspondingPoint(K, -1);
	for (int k = 0; k < K; ++k) {
		ifs >> correspondingView[k];
		ifs >> correspondingPoint[k];
		ifs >> measurements(0, k) >> measurements(1, k);
		measurements.col(k) /= avg_focal_length;
	}
	std::cout << "Done." << std::endl;

	// Read cameras params
	std::cout << "Reading cameras params..." << std::endl;
	CameraMatrix::Vector cams(N);
	DistortionFunction::Vector distortions(N);
	unsigned int accumulatedPoints = 0;
	for (int i = 0; i < N; ++i) {
		Vector3X om, T;
		Scalar f, k1, k2;
		ifs >> om(0) >> om(1) >> om(2);
		ifs >> T(0) >> T(1) >> T(2);
		ifs >> f >> k1 >> k2;

		Matrix3X K = Matrix3X::Identity();
		K(0, 0) = K(1, 1) = -f / avg_focal_length;
		cams[i].setIntrinsic(K);
		cams[i].setTranslation(T);

		Matrix3X R;
		Math::createRotationMatrixRodrigues(om, R);
		cams[i].setRotation(R);

		const Scalar f2 = f * f;
		distortions[i] = DistortionFunction(k1 * f2, k2 * f2 * f2);
	}
	std::cout << "Done." << std::endl;

	// Read the data points
	std::cout << "Reading 3D points..." << std::endl;
	Matrix3XX data(3, M);
	for (int j = 0; j < M; ++j) {
		ifs >> data(0, j) >> data(1, j) >> data(2, j);
	}
	std::cout << "Done." << std::endl;

	/***************** Show statistics before the optimization *****************/
	Utils::showErrorStatistics(avg_focal_length, INLIER_THRESHOLD, cams, distortions,
		data, measurements, correspondingView, correspondingPoint);
	Utils::showObjective(avg_focal_length, INLIER_THRESHOLD, cams, distortions,
		data, measurements, correspondingView, correspondingPoint);

	/***************** Setup and run the optimization *****************/
	VectorXX weights = VectorXX::Ones(measurements.cols());

	// Set initial optimization parameters
	OptimizationFunctor::InputType params;
	params.cams = cams;
	params.distortions = distortions;
	params.data_points = data;
	params.weights = weights;

	// Craete optimization functor
	OptimizationFunctor functor(data.cols(), cams.size(), measurements, correspondingView, correspondingPoint, INLIER_THRESHOLD);
	
  // Craete the LM optimizer 
#ifdef QRKIT
  Eigen::BacktrackLevMarq< OptimizationFunctor, true > lm(functor);
  // Run optimization
  clock_t begin = clock();
  Eigen::BacktrackLevMarqInfo::Status info = lm.minimize(params);
  std::cout << "lm.minimize(params) ... " << double(clock() - begin) / CLOCKS_PER_SEC << "s" << std::endl;
  std::cout << "LM finished with status: " << Eigen::BacktrackLevMarqInfo::statusToString(info) << std::endl;
#elif QRCHOL
  Eigen::BacktrackLevMarqQRCHol< OptimizationFunctor, true > lm(functor);
  // Run optimization
  clock_t begin = clock();
  Eigen::BacktrackLevMarqQRCHolInfo::Status info = lm.minimize(params);
  std::cout << "lm.minimize(params) ... " << double(clock() - begin) / CLOCKS_PER_SEC << "s" << std::endl;
  std::cout << "LM finished with status: " << Eigen::BacktrackLevMarqQRCHolInfo::statusToString(info) << std::endl;
#elif MOREQR
  Eigen::BacktrackLevMarqMore< OptimizationFunctor, true > lm(functor);
  // Run optimization
  clock_t begin = clock();
  Eigen::BacktrackLevMarqMoreInfo::Status info = lm.minimize(params);
  std::cout << "lm.minimize(params) ... " << double(clock() - begin) / CLOCKS_PER_SEC << "s" << std::endl;
  std::cout << "LM finished with status: " << Eigen::BacktrackLevMarqMoreInfo::statusToString(info) << std::endl;
#elif QRSPQR
  Eigen::BacktrackLevMarq< OptimizationFunctor, true > lm(functor);
  // Run optimization
  clock_t begin = clock();
  Eigen::BacktrackLevMarqInfo::Status info = lm.minimize(params);
  std::cout << "lm.minimize(params) ... " << Scalar(clock() - begin) / CLOCKS_PER_SEC << "s" << std::endl;
  std::cout << "LM finished with status: " << Eigen::BacktrackLevMarqInfo::statusToString(info) << std::endl;
#elif CHOLESKY
  Eigen::BacktrackLevMarqCholesky< OptimizationFunctor, true > lm(functor);
  // Run optimization
  clock_t begin = clock();
  Eigen::BacktrackLevMarqCholeskyInfo::Status info = lm.minimize(params);
  std::cout << "lm.minimize(params) ... " << double(clock() - begin) / CLOCKS_PER_SEC << "s" << std::endl;
  std::cout << "LM finished with status: " << Eigen::BacktrackLevMarqCholeskyInfo::statusToString(info) << std::endl;
#endif

	/***************** Show statistics after the optimization *****************/
	Utils::showErrorStatistics(avg_focal_length, INLIER_THRESHOLD, params.cams, params.distortions,
		params.data_points, measurements, correspondingView, correspondingPoint);
	Utils::showObjective(avg_focal_length, INLIER_THRESHOLD, params.cams, params.distortions,
		params.data_points, measurements, correspondingView, correspondingPoint);

	Logger::instance()->log(Logger::Info, "Computation DONE!");

	return ReturnCodes::Success;
}

// Override system assert so one can set a breakpoint in it rather than clicking "Retry" and "Break"
void __cdecl _wassert(_In_z_ wchar_t const* _Message, _In_z_ wchar_t const* _File, _In_ unsigned _Line)
{
	std::wcerr << _File << "(" << _Line << "): ASSERT FAILED [" << _Message << "]\n";

	abort();
}
