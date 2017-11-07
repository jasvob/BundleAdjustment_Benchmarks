#define _wassert wassert_awf
#include <cassert>

#include <iostream>
#include <iomanip>

#include <random>
#include <ctime>

#include <Eigen/Eigen>

#include "Logger.h"

#include "Optimization/BAFunctor.h"
#include "CameraMatrix.h"
#include "DistortionFunction.h"

#include "Utils.h"
#include "MathUtils.h"

enum ReturnCodes {
	Success = 0,
	WrongInputParams = 1,
	WrongInputFile = 2,

};

typedef BAFunctor OptimizationFunctor;

const double AVG_FOCAL_LENGTH = 1.0;
const double INLIER_THRESHOLD = 0.5;

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
	const double avg_focal_length = AVG_FOCAL_LENGTH;

	int N, M, K;
	ifs >> N >> M >> K;
	std::cout << "N(cameras) = " << N << ", M(points) = " << M << ", K(measurements) = " << K << std::endl;

	// Read image measurements data
	std::cout << "Reading image measurements..." << std::endl;
	Eigen::Matrix2Xd measurements(2, K);
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
		Eigen::Vector3d om, T;
		double f, k1, k2;
		ifs >> om(0) >> om(1) >> om(2);
		ifs >> T(0) >> T(1) >> T(2);
		ifs >> f >> k1 >> k2;

		Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
		K(0, 0) = K(1, 1) = -f / avg_focal_length;
		cams[i].setIntrinsic(K);
		cams[i].setTranslation(T);

		Eigen::Matrix3d R;
		Math::createRotationMatrixRodrigues(om, R);
		cams[i].setRotation(R);

		const double f2 = f * f;
		distortions[i] = DistortionFunction(k1 * f2, k2 * f2 * f2);
	}
	std::cout << "Done." << std::endl;

	// Read the data points
	std::cout << "Reading 3D points..." << std::endl;
	Eigen::Matrix3Xd data(3, M);
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
	Eigen::VectorXd weights = Eigen::VectorXd::Ones(measurements.cols());
		
	// Set initial optimization parameters
	OptimizationFunctor::InputType params;
	params.cams = cams;
	params.distortions = distortions;
	params.data_points = data;
	params.weights = weights;

	// Craete optimization functor
	OptimizationFunctor functor(data.cols(), cams.size(), measurements, correspondingView, correspondingPoint, INLIER_THRESHOLD);
	



	// Check Jacobian
	/*
	std::cout << "Testing Jacobian ..." << std::endl;
	for (float eps = 1e-8f; eps < 1.1e-3f; eps *= 10.f) {
		std::cout << "Eps = " << eps << std::endl;
		NumericalDiff<OptimizationFunctor> fd{ functor, OptimizationFunctor::Scalar(eps) };
		OptimizationFunctor::JacobianType J;
		OptimizationFunctor::JacobianType J_fd;
		std::cout << "Compute J ..." << std::endl;
		functor.df(params, J);
		std::cout << "Compute J_fd ..." << std::endl;
		fd.df(params, J_fd);
		std::cout << "Compute diff ..." << std::endl;
		double diff = (J - J_fd).norm();

		Logger::instance()->logMatrixCSV(J.block(0, 0, 1024, 1024).toDense(), "J_pts.csv");
		Logger::instance()->logMatrixCSV(J.block(0, data.cols() * 3, 1024, cams.size() * 9).toDense(), "J_cams.csv");
		Logger::instance()->logMatrixCSV(J_fd.block(0, 0, 1024, 1024).toDense(), "J_fd_pts.csv");
		Logger::instance()->logMatrixCSV(J_fd.block(0, data.cols() * 3, 1024, cams.size() * 9).toDense(), "J_fd_cams.csv");

		if (diff > 0) {
			std::stringstream ss;
			ss << "Jacobian diff(eps=" << eps << "), = " << diff;
			std::cout << ss.str() << std::endl;
		}

		if (diff > 10.0) {
			std::cout << "Test Jacobian - ERROR TOO BIG, exitting..." << std::endl;
			return 0;
		}
	}
	std::cout << "Test Jacobian - DONE, exitting..." << std::endl;
	//*/




	// Craete the LM optimizer
	Eigen::LevenbergMarquardt< OptimizationFunctor > lm(functor);
	lm.setVerbose(true);
	lm.setMaxfev(40);
	//lm.setExternalScaling(1e-3);
	//lm.setFactor(1e-3);
	
	// Run optimization
	clock_t begin = clock();
	Eigen::LevenbergMarquardtSpace::Status info = lm.minimize(params);
	std::cout << "lm.minimize(params) ... " << double(clock() - begin) / CLOCKS_PER_SEC << "s" << std::endl;
	
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
