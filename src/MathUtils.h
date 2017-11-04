#ifndef MATH_UTILS_H
#define MATH_UTILS_H

#include <Eigen/Eigen>

namespace Math {
	inline double copysign(double x, double y) {
		return (y < 0) ? -std::abs(x) : std::abs(x);
	}

	inline void makeCrossProductMatrix(const Eigen::Vector3d& v, Eigen::Matrix3d &m) {
		assert(v.size() == 3);
		assert(m.rows() == 3);
		assert(m.cols() == 3);

		m(0, 0) = 0; m(0, 1) = -v(2); m(0, 2) = v(1);
		m(1, 0) = v(2); m(1, 1) = 0; m(1, 2) = -v(0);
		m(2, 0) = -v(1); m(2, 1) = v(0); m(2, 2) = 0;
	}

	inline void createQuaternionFromRotationMatrix(const Eigen::Matrix3d& R, Eigen::Vector4d& q) {
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

	inline void createRotationMatrixFromQuaternion(const Eigen::Vector4d& q, Eigen::Matrix3d& R) {
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

	inline void createRotationMatrixRodrigues(const Eigen::Vector3d &omega, Eigen::Matrix3d &R) {
		assert(omega.size() == 3);
		assert(R.rows() == 3);
		assert(R.cols() == 3);

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

	inline void createRodriguesParamFromRotationMatrix(const Eigen::Matrix3d &R, Eigen::Vector3d &omega) {
		assert(omega.size() == 3);
		assert(R.rows() == 3);
		assert(R.cols() == 3);

		Eigen::Vector4d q;
		createQuaternionFromRotationMatrix(R, q);
		omega = q.segment<3>(0);
		omega.normalize();
		omega *= (2.0 * acos(q(3)));
	}
}

#endif
