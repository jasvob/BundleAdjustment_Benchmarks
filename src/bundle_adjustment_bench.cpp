#define _wassert wassert_awf
#include <cassert>

#include <iostream>
#include <iomanip>

#include <random>

#include <Eigen/Eigen>

#include "Logger.h"

#include "Optimization/BAFunctor.h"
#include "CameraMatrix.h"
#include "DistortionFunction.h"

#include "Utils.h"

enum ReturnCodes {
	Success = 0,
	WrongInputParams = 1,
	WrongInputFile = 2,

};

typedef BAFunctor OptimizationFunctor;

const double AVG_FOCAL_LENGTH = 1.0;
const double INLIER_THRESHOLD = 0.5;

using namespace Eigen;

double copysign(double x, double y) {
	return (y < 0) ? -std::abs(x) : std::abs(x);
}

void makeCrossProductMatrix(const Eigen::Vector3d& v, Eigen::Matrix3d &m) {
	assert(v.size() == 3);
	assert(m.rows() == m.cols() == 3);

	m(0, 0) = 0; m(0, 1) = -v(2); m(0, 2) = v(1);
	m(1, 0) = v(2); m(1, 1) = 0; m(1, 2) = -v(0);
	m(2, 0) = -v(1); m(2, 1) = v(0); m(2, 2) = 0;
}

void createQuaternionFromRotationMatrix(const Eigen::Matrix3d& R, Eigen::Vector4d& q) {
	assert(R.rows() == 3);
	assert(R.cols() == 3);
	assert(q.size() == 4);

	double const m00 = R(0, 0); double const m01 = R(0, 1); double const m02 = R(0, 2);
	double const m10 = R(1, 0); double const m11 = R(1, 1); double const m12 = R(1, 2);
	double const m20 = R(2, 0); double const m21 = R(1, 2); double const m22 = R(2, 2);

	q(3) = sqrt(std::max(0.0, 1.0 + m00 + m11 + m22)) / 2;
	q(0) = sqrt(std::max(0.0, 1.0 + m00 - m11 - m22)) / 2;
	q(1) = sqrt(std::max(0.0, 1.0 - m00 + m11 - m22)) / 2;
	q(2) = sqrt(std::max(0.0, 1.0 - m00 - m11 + m22)) / 2;

	q(0) = copysign(q(0), m21 - m12);
	q(1) = copysign(q(1), m02 - m20);
	q(2) = copysign(q(2), m10 - m01);
}

void createRotationMatrixFromQuaternion(const Eigen::Vector4d& q, Eigen::Matrix3d& R) {
	assert(R.rows() == 3);
	assert(R.cols() == 3);
	assert(q.size() == 4);

	double x = q(0);
	double y = q(1);
	double z = q(2);
	double w = q(3);

	double const len = sqrt(x*x + y*y + z*z + w*w);
	double const s = (len > 0.0) ? (1.0 / len) : 0.0;

	x *= s; y *= s; z *= s; w *= s;

	double const wx = 2 * w*x; double const wy = 2 * w*y; double const wz = 2 * w*z;
	double const xx = 2 * x*x; double const xy = 2 * x*y; double const xz = 2 * x*z;
	double const yy = 2 * y*y; double const yz = 2 * y*z; double const zz = 2 * z*z;

	R(0, 0) = 1.0 - (yy + zz); R(0, 1) = xy - wz;         R(0, 2) = xz + wy;
	R(1, 0) = xy + wz;         R(1, 1) = 1.0 - (xx + zz); R(1, 2) = yz - wx;
	R(2, 0) = xz - wy;         R(2, 1) = yz + wx;         R(2, 2) = 1.0 - (xx + yy);
}

void createRotationMatrixRodrigues(const Eigen::Vector3d &omega, Eigen::Matrix3d &R) {
	assert(omega.size() == 3);
	assert(R.rows() == R.cols() == 3);

	const double theta = omega.norm();
	R.setIdentity();

	if (fabs(theta) > 1e-6) {
		Eigen::Matrix3d J, J2;
		makeCrossProductMatrix(omega, J);
		J2 = J * J;
		const double c1 = sin(theta) / theta;
		const double c2 = (1.0 - cos(theta)) / (theta * theta);
		R = R + c1 * J + c2 * J2;
	}
}

void createRodriguesParamFromRotationMatrix(const Eigen::Matrix3d &R, Eigen::Vector3d &omega) {
	assert(omega.size() == 3);
	assert(R.rows() == R.cols() == 3);

	Eigen::Vector4d q;
	createQuaternionFromRotationMatrix(R, q);
	omega = q.segment<3>(0);
	omega.normalize();
	omega *= (2.0 * acos(q(3)));
}

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
		createRotationMatrixRodrigues(om, R);
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


	return ReturnCodes::Success;
	
	/***************** Setup and run the optimization *****************/
	Matrix3X control_vertices_gt;
	//Matrix3X data;

	// INITIAL PARAMS
	OptimizationFunctor::InputType params;
	params.control_vertices = control_vertices_gt;
	
	OptimizationFunctor functor(data);
	
	// Set-up the optimization
	Eigen::LevenbergMarquardt< OptimizationFunctor > lm(functor);
	lm.setVerbose(true);
	lm.setMaxfev(40);
	
	Eigen::LevenbergMarquardtSpace::Status info = lm.minimize(params);

	/***************** Show statistics after the optimization *****************/



	Logger::instance()->log(Logger::Info, "Computation DONE!");

	return ReturnCodes::Success;
}

// Override system assert so one can set a breakpoint in it rather than clicking "Retry" and "Break"
void __cdecl _wassert(_In_z_ wchar_t const* _Message, _In_z_ wchar_t const* _File, _In_ unsigned _Line)
{
	std::wcerr << _File << "(" << _Line << "): ASSERT FAILED [" << _Message << "]\n";

	abort();
}
