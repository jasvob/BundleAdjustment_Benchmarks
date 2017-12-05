#include "DistortionFunction.h"

DistortionFunction::DistortionFunction()
	: m_k1(1.0), m_k2(1.0) {
}

DistortionFunction::DistortionFunction(Scalar k1, Scalar k2)
	: m_k1(k1), m_k2(k2) {
}

DistortionFunction::~DistortionFunction() {
}

Vector2X DistortionFunction::operator()(const Vector2X &xu) const {
	const Scalar r2 = xu(0) * xu(0) + xu(1) * xu(1);
	const Scalar r4 = r2 * r2;
	const Scalar kr = 1 + m_k1 * r2 + m_k2 * r4;

	return Vector2X(
		kr * xu(0),
		kr * xu(1)
	);
}

Matrix2X DistortionFunction::derivativeWrtRadialParameters(const Vector2X &xu) const {
	const Scalar r2 = xu(0) * xu(0) + xu(1) * xu(1);
	const Scalar r4 = r2 * r2;

	Matrix2X deriv;
	deriv(0, 0) = xu(0) * r2;	// d xd / d k1
	deriv(0, 1) = xu(0) * r4;	// d xd / d k2
	deriv(1, 0) = xu(1) * r2;	// d yd / d k1
	deriv(1, 1) = xu(1) * r4;	// d yd / d k2

	return deriv;
}

Matrix2X DistortionFunction::derivativeWrtUndistortedPoint(const Vector2X &xu) const {
	const Scalar r2 = xu(0) * xu(0) + xu(1) * xu(1);
	const Scalar r4 = r2 * r2;
	const Scalar kr = 1 + m_k1 * r2 + m_k2 * r4;
	const Scalar dkr = 2 * m_k1 + 4 * m_k2 * r2;

	Matrix2X deriv;
	deriv(0, 0) = kr + xu(0) * xu(0) * dkr;	// d xd / d xu
	deriv(0, 1) =      xu(0) * xu(1) * dkr; // d xd / d yu
	deriv(1, 0) = deriv(0, 1);				// d yd / d xu
	deriv(1, 1) = kr + xu(1) * xu(1) * dkr; // d yd / d yu
	
	return deriv;
}