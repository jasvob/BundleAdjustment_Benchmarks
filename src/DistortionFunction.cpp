#include "DistortionFunction.h"

DistortionFunction::DistortionFunction()
	: k1(1.0), k2(1.0) {
}

DistortionFunction::DistortionFunction(double k1, double k2) 
	: k1(k1), k2(k2) {
}

DistortionFunction::~DistortionFunction() {
}

Eigen::Vector2d DistortionFunction::operator()(const Eigen::Vector2d &xu) const {
	const double r2 = xu(0) * xu(0) + xu(1) * xu(1);
	const double r4 = r2 * r2;
	const double kr = 1 + k1 * r2 + k2 * r4;

	return Eigen::Vector2d(
		kr * xu(0),
		kr * xu(1)
	);
}