#ifndef UTILS_H
#define UTILS_H

#include <Eigen/Eigen>
#include "CameraMatrix.h"
#include "DistortionFunction.h"
#include "BATypeUtils.h"

namespace Utils {
	inline Scalar psi(Scalar const tau2, Scalar const r2) {
		Scalar const r4 = r2*r2, tau4 = tau2*tau2;
		return (r2 < tau2) ? r2*(Scalar(3.0) - Scalar(3.0) * r2 / tau2 + r4 / tau4) / Scalar(6.0) : tau2 / Scalar(6.0);
	}

	Scalar showErrorStatistics(Scalar const avg_focal_length, const Scalar inlierThreshold, 
		const CameraMatrix::Vector &cams, const DistortionFunction::Vector &distortions,
		const Matrix3XX& data, const Matrix2XX& measurements,
		const std::vector<int>& correspondingView, const std::vector<int>& correspondingPoint) {

		const int K = measurements.cols();
		int nInliers = 0;

		Scalar meanReprojectionError = 0.0;
		Scalar inlierReprojectionError = 0.0;
		for (int k = 0; k < K; ++k) {
			const int i = correspondingView[k];
			const int j = correspondingPoint[k];
			Vector2X p = cams[i].projectPoint(distortions[i], data.col(j));

			Scalar reprojectionError = (avg_focal_length * (p - measurements.col(k))).norm();
			meanReprojectionError += reprojectionError;

			if (reprojectionError <= inlierThreshold) {
				++nInliers;
				inlierReprojectionError += reprojectionError;
			}
		}

		std::cout << "Mean reprojection error: " << meanReprojectionError / K << std::endl;
		std::cout << "Inlier mean reprojection error: " << inlierReprojectionError / nInliers << " (" << nInliers << " / " << K << " inliers)" << std::endl;
	
		return Scalar(nInliers) / K;
	}

	Scalar showObjective(Scalar const avg_focal_length, const Scalar inlierThreshold,
		const CameraMatrix::Vector &cams, const DistortionFunction::Vector &distortions,
		const Matrix3XX& data, const Matrix2XX& measurements,
		const std::vector<int>& correspondingView, const std::vector<int>& correspondingPoint) {
		
		const int K = measurements.cols();

		const Scalar tau2 = inlierThreshold * inlierThreshold;
		const Scalar avg_focal_length2 = avg_focal_length * avg_focal_length;

		Scalar obj = 0.0;
		for (int k = 0; k < K; ++k) {
			const int i = correspondingView[k];
			const int j = correspondingPoint[k];
			Vector2X p = cams[i].projectPoint(distortions[i], data.col(j));

			const Scalar r2 = (avg_focal_length2 * (p - measurements.col(k))).norm();
			obj += psi(tau2, r2);
		}

		std::cout << "True objective: " << obj << std::endl;

		return obj;
	}
}

#endif
