#ifndef MATH_UTILS_H
#define MATH_UTILS_H

#include <Eigen/Eigen>

#include "BATypeUtils.h"

namespace Math {
	inline Scalar copysign(Scalar x, Scalar y) {
		return (y < 0) ? -std::abs(x) : std::abs(x);
	}

	inline void makeCrossProductMatrix(const Vector3X& v, Matrix3X &m) {
		assert(v.size() == 3);
		assert(m.rows() == 3);
		assert(m.cols() == 3);

		m(0, 0) = 0; m(0, 1) = -v(2); m(0, 2) = v(1);
		m(1, 0) = v(2); m(1, 1) = 0; m(1, 2) = -v(0);
		m(2, 0) = -v(1); m(2, 1) = v(0); m(2, 2) = 0;
	}

	inline void createQuaternionFromRotationMatrix(const Matrix3X& R, Vector4X& q) {
		assert(R.rows() == 3);
		assert(R.cols() == 3);
		assert(q.size() == 4);

		Scalar const m00 = R(0, 0); Scalar const m01 = R(0, 1); Scalar const m02 = R(0, 2);
		Scalar const m10 = R(1, 0); Scalar const m11 = R(1, 1); Scalar const m12 = R(1, 2);
		Scalar const m20 = R(2, 0); Scalar const m21 = R(1, 2); Scalar const m22 = R(2, 2);

		q(3) = sqrt(std::max(Scalar(0.0), Scalar(1.0) + m00 + m11 + m22)) / 2;
		q(0) = sqrt(std::max(Scalar(0.0), Scalar(1.0) + m00 - m11 - m22)) / 2;
		q(1) = sqrt(std::max(Scalar(0.0), Scalar(1.0) - m00 + m11 - m22)) / 2;
		q(2) = sqrt(std::max(Scalar(0.0), Scalar(1.0) - m00 - m11 + m22)) / 2;

		q(0) = copysign(q(0), m21 - m12);
		q(1) = copysign(q(1), m02 - m20);
		q(2) = copysign(q(2), m10 - m01);
	}

	inline void createRotationMatrixFromQuaternion(const Vector4X& q, Matrix3X& R) {
		assert(R.rows() == 3);
		assert(R.cols() == 3);
		assert(q.size() == 4);

		Scalar x = q(0);
		Scalar y = q(1);
		Scalar z = q(2);
		Scalar w = q(3);

		Scalar const len = sqrt(x*x + y*y + z*z + w*w);
		Scalar const s = (len > Scalar(0.0)) ? (Scalar(1.0) / len) : Scalar(0.0);

		x *= s; y *= s; z *= s; w *= s;

		Scalar const wx = 2 * w*x; Scalar const wy = 2 * w*y; Scalar const wz = 2 * w*z;
		Scalar const xx = 2 * x*x; Scalar const xy = 2 * x*y; Scalar const xz = 2 * x*z;
		Scalar const yy = 2 * y*y; Scalar const yz = 2 * y*z; Scalar const zz = 2 * z*z;

		R(0, 0) = Scalar(1.0) - (yy + zz); R(0, 1) = xy - wz;         R(0, 2) = xz + wy;
		R(1, 0) = xy + wz;         R(1, 1) = Scalar(1.0) - (xx + zz); R(1, 2) = yz - wx;
		R(2, 0) = xz - wy;         R(2, 1) = yz + wx;         R(2, 2) = Scalar(1.0) - (xx + yy);
	}

	inline void createRotationMatrixRodrigues(const Vector3X &omega, Matrix3X &R) {
		assert(omega.size() == 3);
		assert(R.rows() == 3);
		assert(R.cols() == 3);

		const Scalar theta = omega.norm();
		R.setIdentity();

		if (std::abs(theta) > Scalar(1e-6)) {
			Matrix3X J, J2;
			makeCrossProductMatrix(omega, J);
			J2 = J * J;
			const Scalar c1 = sin(theta) / theta;
			const Scalar c2 = (Scalar(1.0) - cos(theta)) / (theta * theta);
			R = R + c1 * J + c2 * J2;
		}
	}

	inline void createRodriguesParamFromRotationMatrix(const Matrix3X &R, Vector3X &omega) {
		assert(omega.size() == 3);
		assert(R.rows() == 3);
		assert(R.cols() == 3);

		Vector4X q;
		createQuaternionFromRotationMatrix(R, q);
		omega = q.segment<3>(0);
		omega.normalize();
		omega *= (2.0 * acos(q(3)));
	}
}

#endif
