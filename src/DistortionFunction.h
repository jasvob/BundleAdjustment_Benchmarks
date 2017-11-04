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

	Eigen::Matrix2d derivativeWrtRadialParameters(const Eigen::Vector2d &xu) const;
	Eigen::Matrix2d derivativeWrtUndistortedPoint(const Eigen::Vector2d &xu) const;

	double getK1() const {
		return this->m_k1;
	}
	double getK2() const {
		return this->m_k2;
	}

	double& k1() {
		return this->m_k1;
	}
	double& k2() {
		return this->m_k2;
	}

private:
	double m_k1, m_k2;
};

#endif

