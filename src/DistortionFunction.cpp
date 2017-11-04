#include "DistortionFunction.h"

DistortionFunction::DistortionFunction()
	: m_k1(1.0), m_k2(1.0) {
}

DistortionFunction::DistortionFunction(double k1, double k2) 
	: m_k1(k1), m_k2(k2) {
}

DistortionFunction::~DistortionFunction() {
}

Eigen::Vector2d DistortionFunction::operator()(const Eigen::Vector2d &xu) const {
	const double r2 = xu(0) * xu(0) + xu(1) * xu(1);
	const double r4 = r2 * r2;
	const double kr = 1 + m_k1 * r2 + m_k2 * r4;

	return Eigen::Vector2d(
		kr * xu(0),
		kr * xu(1)
	);
}

Eigen::Matrix2d DistortionFunction::derivativeWrtRadialParameters(const Eigen::Vector2d &xu) const {
	const double r2 = xu(0) * xu(0) + xu(1) * xu(1);
	const double r4 = r2 * r2;

	Eigen::Matrix2d deriv;
	deriv(0, 0) = xu(0) * r2;	// d xd / d k1
	deriv(0, 1) = xu(0) * r4;	// d xd / d k2
	deriv(1, 0) = xu(1) * r2;	// d yd / d k1
	deriv(1, 1) = xu(1) * r4;	// d yd / d k2

	return deriv;
}

Eigen::Matrix2d DistortionFunction::derivativeWrtUndistortedPoint(const Eigen::Vector2d &xu) const {
	const double r2 = xu(0) * xu(0) + xu(1) * xu(1);
	const double r4 = r2 * r2;
	const double kr = 1 + m_k1 * r2 + m_k2 * r4;
	const double dkr = 2 * m_k1 + 4 * m_k2 * r2;

	Eigen::Matrix2d deriv;
	deriv(0, 0) = kr + xu(0) * xu(0) * dkr;	// d xd / d xu
	deriv(0, 1) =      xu(0) * xu(1) * dkr; // d xd / d yu
	deriv(1, 0) = deriv(0, 1);				// d yd / d xu
	deriv(1, 1) = kr + xu(1) * xu(1) * dkr; // d yd / d yu
	
	return deriv;
}