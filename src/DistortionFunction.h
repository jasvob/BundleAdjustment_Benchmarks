#ifndef DISTORTION_FUNCTION_H
#define DISTORTION_FUNCTION_H

#include <vector>
#include <Eigen/Eigen>

#include "BATypeUtils.h"

class DistortionFunction {
public:
	typedef std::vector<DistortionFunction> Vector;

	DistortionFunction();
	DistortionFunction(Scalar k1, Scalar k2);
	~DistortionFunction();

	Vector2X operator()(const Vector2X &xu) const;

	Matrix2X derivativeWrtRadialParameters(const Vector2X &xu) const;
  Matrix2X derivativeWrtUndistortedPoint(const Vector2X &xu) const;

	Scalar getK1() const {
		return this->m_k1;
	}
  Scalar getK2() const {
		return this->m_k2;
	}

  Scalar& k1() {
		return this->m_k1;
	}
  Scalar& k2() {
		return this->m_k2;
	}

private:
  Scalar m_k1, m_k2;
};

#endif

