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

#include <unsupported/Eigen/BacktrackLevMarq>
#include <unsupported/Eigen/NumericalDiff>

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
		ifs >> correspondingPoint[k];
    ifs >> correspondingView[k];
    correspondingView[k] -= 1;
    correspondingPoint[k] -= 1;
		ifs >> measurements(0, k) >> measurements(1, k);
		measurements.col(k) /= avg_focal_length;
	}
	std::cout << "Done." << std::endl;

	// Read cameras params
	std::cout << "Reading cameras params..." << std::endl;
	CameraMatrix::Vector cams(N);
	DistortionFunction::Vector distortions(N);
	for (int i = 0; i < N; ++i) {
    Matrix3x4d P = Matrix3x4d::Zero();
		Eigen::Vector3d om, T;
		double f, k1, k2;
		ifs >> P(0, 0) >> P(0, 1) >> P(0, 2) >> P(0, 3);
    ifs >> P(1, 0) >> P(1, 1) >> P(1, 2) >> P(1, 3);
    ifs >> P(2, 0) >> P(2, 1) >> P(2, 2) >> P(2, 3);
    //ifs >> P(3, 0) >> P(3, 1) >> P(3, 2) >> P(3, 3);

    cams[i] = CameraMatrix(P);

		distortions[i] = DistortionFunction(0, 0);
	}
	std::cout << "Done." << std::endl;

	// Read the data points
  std::cout << "Reading 3D points..." << std::endl;
	Eigen::Matrix3Xd data = Eigen::Matrix3Xd::Random(3, M);  
  /*
  for (int j = 0; j < M; ++j) {
		ifs >> data(0, j) >> data(1, j) >> data(2, j);
	}
  */
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
	Eigen::BacktrackLevMarq< OptimizationFunctor, true > lm(functor);

	//lm.setExternalScaling(1e-3);
	//lm.setFactor(1e-3);
	
	// Run optimization
	clock_t begin = clock();
	Eigen::BacktrackLevMarqInfo::Status info = lm.minimize(params);
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
