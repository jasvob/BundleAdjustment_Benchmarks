#ifndef UTILS_H
#define UTILS_H

#include <Eigen/Eigen>
#include "CameraMatrix.h"
#include "DistortionFunction.h"

namespace Utils {
	inline double psi(double const tau2, double const r2) { return (r2 < tau2) ? r2*(2.0 - r2 / tau2) / 4.0f : tau2 / 4; }
	inline double psi_weight(double const tau2, double const r2) { return std::max(0.0, 1.0 - r2 / tau2); }
	inline double psi_hat(double const tau2, double const r2, double const w2) { return w2*r2 + tau2 / 2.0*(w2 - 1)*(w2 - 1); }

	double showErrorStatistics(double const avg_focal_length, const double inlierThreshold, 
		const CameraMatrix::Vector &cams, const DistortionFunction::Vector &distortions,
		const Eigen::Matrix3Xd& data, const Eigen::Matrix2Xd& measurements,
		const std::vector<int>& correspondingView, const std::vector<int>& correspondingPoint) {

		const int K = measurements.cols();
		int nInliers = 0;

		double meanReprojectionError = 0.0;
		double inlierReprojectionError = 0.0;
		for (int k = 0; k < K; ++k) {
			const int i = correspondingView[k];
			const int j = correspondingPoint[k];
			Eigen::Vector2d p = cams[i].projectPoint(distortions[i], data.col(j));

			double reprojectionError = (avg_focal_length * (p - measurements.col(k))).norm();
			meanReprojectionError += reprojectionError;

			if (reprojectionError <= inlierThreshold) {
				++nInliers;
				inlierReprojectionError += reprojectionError;
			}
		}

		std::cout << "Mean reprojection error: " << meanReprojectionError / K << std::endl;
		std::cout << "Inlier mean reprojection error: " << inlierReprojectionError / nInliers << " (" << nInliers << " / " << K << " inliers)" << std::endl;
	
		return double(nInliers) / K;
	}

	double showObjective(double const avg_focal_length, const double inlierThreshold,
		const CameraMatrix::Vector &cams, const DistortionFunction::Vector &distortions,
		const Eigen::Matrix3Xd& data, const Eigen::Matrix2Xd& measurements,
		const std::vector<int>& correspondingView, const std::vector<int>& correspondingPoint) {
		
		const int K = measurements.cols();

		const double tau2 = inlierThreshold * inlierThreshold;
		const double avg_focal_length2 = avg_focal_length * avg_focal_length;

		double obj = 0.0;
		for (int k = 0; k < K; ++k) {
			const int i = correspondingView[k];
			const int j = correspondingPoint[k];
			Eigen::Vector2d p = cams[i].projectPoint(distortions[i], data.col(j));

			const double r2 = (avg_focal_length2 * (p - measurements.col(k))).norm();
			obj += psi(tau2, r2);
		}

		std::cout << "True objective: " << obj << std::endl;

		return obj;
	}
}

#endif
