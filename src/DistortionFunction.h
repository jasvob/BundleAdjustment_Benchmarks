#ifndef DISTORTION_FUNCTION_H
#define DISTORTION_FUNCTION_H

#include <vector>
#include <Eigen/Eigen>

class DistortionFunction {
public:
	typedef std::vector<DistortionFunction> Vector;

	DistortionFunction();
	DistortionFunction(double k1, double k2);
	~DistortionFunction();

	Eigen::Vector2d operator()(const Eigen::Vector2d &xu) const;

private:
	double k1, k2;
};

#endif

