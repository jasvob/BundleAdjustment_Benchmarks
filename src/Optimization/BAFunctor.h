#ifndef BA_FUNCTOR_H
#define BA_FUNCTOR_H

#include <Eigen/Eigen>

#include <Eigen/SparseCore>

#include <unsupported/Eigen/MatrixFunctions>
#include <unsupported/Eigen/BacktrackLevMarq>
#include <unsupported/Eigen/SparseExtra>
#include <unsupported/Eigen/SparseQRExtra>

#include <Eigen/SparseCore>
#include <Eigen/src/Core/util/DisableStupidWarnings.h>
#include <suitesparse/SuiteSparseQR.hpp>
#include <Eigen/src/CholmodSupport/CholmodSupport.h>
#include <Eigen/src/SPQRSupport/SuiteSparseQRSupport.h>

#include "../CameraMatrix.h"
#include "../DistortionFunction.h"

#include "../MathUtils.h"

typedef Eigen::Matrix<double, 2, 3> Matrix2x3d;
typedef Eigen::Matrix<double, 2, 6> Matrix2x6d;
typedef Eigen::Matrix<double, 3, 4> Matrix3x4d;
typedef Eigen::Matrix<double, 3, 6> Matrix3x6d;

using namespace Eigen;

//typedef Index SparseDataType;
//typedef SuiteSparse_long SparseDataType;
typedef int SparseDataType;
typedef double Scalar;
typedef Matrix<Scalar, Dynamic, 1> VectorX;

struct BAFunctor : Eigen::SparseFunctor<Scalar, SparseDataType> {
	typedef Eigen::SparseFunctor<Scalar, SparseDataType> Base;
	typedef typename Base::JacobianType JacobianType;

	// Variables for optimization live in InputType
	struct InputType {
		CameraMatrix::Vector cams;
		DistortionFunction::Vector distortions;
		Eigen::Matrix3Xd data_points;
		Eigen::VectorXd weights;

		Index nCameras() const { return cams.size(); }
		Index nDistortions() const { return distortions.size(); }
		Index nDataPoints() const { return data_points.cols(); }
		Index nWeights() const { return weights.size(); }

		typedef SparseDataType Index;
	};

	// And the optimization steps are computed using VectorType.
	// For subdivs (see xx), the correspondences are of type (int, Vec2) while the updates are of type (Vec2).
	// The interactions between InputType and VectorType are restricted to:
	//   The Jacobian computation takes an InputType, and its rows must easily convert to VectorType
	//   The increment_in_place operation takes InputType and StepType. 
	typedef VectorX VectorType;
		
	// Functor constructor
	BAFunctor(const Eigen::Index numPoints, const Eigen::Index numCameras, const Eigen::Matrix2Xd &measurements,
		const std::vector<int> &correspondingView, const std::vector<int> &correspondingPoint, 
		double inlierThreshold);

	// Functor functions
	// 1. Evaluate the residuals at x
	int operator()(const InputType& x, ValueType& fvec);
	virtual void f_impl(const InputType& x, ValueType& fvec);

	// 2. Evaluate jacobian at x
	int df(const InputType& x, JacobianType& fjac);
	virtual void df_impl(const InputType& x, Eigen::TripletArray<Scalar, typename JacobianType::Index>& jvals);

	// Update function
	void increment_in_place(InputType* x, StepType const& p);
	virtual void increment_in_place_impl(InputType* x, StepType const& p);
	
	// Input data
	Eigen::Matrix2Xd measurements;
	std::vector<int> correspondingView;
	std::vector<int> correspondingPoint;
	const double inlierThreshold;
	
	Eigen::Index nMeasurements() const {
		return measurements.cols();
	}

	const Index numParameters;
	const Index numResiduals;
	const Index numJacobianNonzeros;
	const Index numPointParams;
	
	// Workspace initialization
	virtual void initWorkspace();

	Scalar estimateNorm(InputType const& x, StepType const& diag);

	// Describe the QR solvers
	// QR for J1 is block diagonal
//	typedef BandedBlockedSparseQR<JacobianType, NaturalOrdering<SparseDataType>, 8, false> LeftSuperBlockSolver;
	//typedef SPQR<JacobianType> LeftSuperBlockSolver;
	// QR for J1'J2 is general dense (faster than general sparse by about 1.5x for n=500K)
	//typedef ColPivHouseholderQR<Matrix<Scalar, Dynamic, Dynamic> > RightSuperBlockSolver;
//	typedef BandedBlockedSparseQR<JacobianType, NaturalOrdering<SparseDataType>, 8, false> RightSuperBlockSolver;
	//typedef SPQR<JacobianType> RightSuperBlockSolver;
	//typedef ColPivHouseholderQR<Matrix<Scalar, Dynamic, Dynamic> > RightSuperBlockSolver;
	// QR solver is concatenation of the above.
	//typedef BlockAngularSparseQR<JacobianType, LeftSuperBlockSolver, RightSuperBlockSolver> SchurlikeQRSolver;
	typedef ColPivHouseholderQR<Matrix<Scalar, Dynamic, Dynamic> > DenseBlockSolver;
	typedef BlockDiagonalSparseQR<JacobianType, DenseBlockSolver> LeftSuperBlockSolver;
	typedef DenseBlockedThinSparseQR<JacobianType, NaturalOrdering<SparseDataType>, 10, true> RightSuperBlockSolver;
	typedef BlockAngularSparseQR<JacobianType, LeftSuperBlockSolver, RightSuperBlockSolver> SchurlikeQRSolver;
	//typedef SPQR<JacobianType> SchurlikeQRSolver;

	//typedef BlockAngularSparseQR<JacobianType, LeftSuperBlockSolver, RightSuperBlockSolver> SchurlikeQRSolver;
	typedef SchurlikeQRSolver QRSolver;

	// And tell the algorithm how to set the QR parameters.
	virtual void initQRSolver(SchurlikeQRSolver &qr);

	/************ UTILITY FUNCTIONS FOR ENERGIES ************/
	void poseDerivatives(const CameraMatrix &cam, const Eigen::Vector3d &pt, Eigen::Vector3d &XX, Matrix3x6d &d_dRT, Eigen::Matrix3d &d_dX) {
		XX = cam.transformPointIntoCameraSpace(pt);

		// See Frank Dellaerts bundle adjustment tutorial.
		// d(dR * R0 * X + t) / d omega = -[R0 * X]_x
		Eigen::Matrix3d J;
		Math::makeCrossProductMatrix(XX - cam.getTranslation(), J);
		J *= -1;

		// Now the transformation from world coords into camera space is xx = Rx + T
		// Hence the derivative of x wrt. T is just the identity matrix
		d_dRT.setIdentity();
		d_dRT.rightCols(3) = J;

		// The derivative of Rx + t wrt x is just R
		d_dX = cam.getRotation();
	}

	/************ ENERGIES, GRADIENTS and UPDATES ************/
	/************ ENERGIES ************/
	// Residual wrt to 3d points
	inline double psi(double const tau2, double const r2) { return (r2 < tau2) ? r2*(2.0 - r2 / tau2) / 4.0f : tau2 / 4; }
	inline double psi_weight(double const tau2, double const r2) { return std::max(0.0, 1.0 - r2 / tau2); }
	inline double psi_hat(double const tau2, double const r2, double const w2) { return w2*r2 + tau2 / 2.0*(w2 - 1)*(w2 - 1); }

	Vector2d projectPoint(const CameraMatrix &cam, const DistortionFunction &distortion, const Vector3d &X) {
		Vector3d XX = cam.transformPointIntoCameraSpace(X);
		Vector2d xu(XX(0) / XX(2), XX(1) / XX(2));
		Vector2d xd = distortion(xu);
		return cam.getFocalLength() * xd;
	}

	const double eps_psi_residual = 1e-20;
	void E_pos(const InputType& x, const Eigen::Matrix2Xd& measurements, const std::vector<int> &correspondingView, const std::vector<int> &correspondingPoint, ValueType& fvec) {
		double sqrInlierThreshold = this->inlierThreshold * this->inlierThreshold;

		for (int i = 0; i < this->nMeasurements(); i++) {
			int view = correspondingView[i];
			int point = correspondingPoint[i];

			// Project 3D point into corresponding camera view
			//Eigen::Vector2d q = x.cams[view].projectPoint(x.distortions[view], x.data_points.col(point));
			Eigen::Vector2d q = this->projectPoint(x.cams[view], x.distortions[view], x.data_points.col(point));
			Eigen::Vector2d r = (q - measurements.col(i));

			double sqrt_psi = sqrt(psi(sqrInlierThreshold, r.squaredNorm()));
			double rnorm_r = 1.0 / std::max(eps_psi_residual, r.norm());

			// Compute residual for the point
			fvec(i * 2 + 0) = r(0) * sqrt_psi * rnorm_r;
			fvec(i * 2 + 1) = r(1) * sqrt_psi * rnorm_r;
			//fvec.segment<2>(i * 2) = r.cwiseProduct(Eigen::Vector2d(sqrt_psi * rnorm_r, sqrt_psi * rnorm_r));
			//fvec.segment<2>(point * x.nCameras() + view) = q - measurements.col(i);

			/*if (i == 0) {
				std::cout << "q: \n" << q << std::endl;
				std::cout << "r: \n" << r << std::endl;
				std::cout << "sqrt_psi, rnorm_r: " << sqrt_psi << ", " << rnorm_r << std::endl;
				std::cout << "fvec: " << fvec.segment<2>(i * 2) << std::endl;
			}*/
		}
	}
	
	/************ GRADIENTS ************/
	void dE_pos(const InputType& x, Eigen::TripletArray<Scalar, typename JacobianType::Index>& jvals) {

		Index pt_base = 0;
		Index cam_base = x.nDataPoints() * 3;

		// Camera parameters ordering
		Index numPointCoords = 3;	// xyz
		Index numCamParams = 9;
		Index transfParamsOffset = 0;
		Index focalLengthOffset = 6;
		Index radialParamsOffset = 7;
		
		Eigen::Vector3d XX;
		Matrix3x6d dXX_dRT;
		Eigen::Matrix3d dXX_dX;
		int view, point;
		for (int i = 0; i < this->nMeasurements(); i++) {
			XX.setZero();
			dXX_dRT.setZero();
			dXX_dX.setZero();

			view = correspondingView[i];
			point = correspondingPoint[i];

			/*if (i == 0) {
				std::cout << "Point/View: " << point << "/" << view << std::endl;
			}*/
		
			this->poseDerivatives(x.cams[view], x.data_points.col(point), XX, dXX_dRT, dXX_dX);
			/*if (i == 0) {
				std::cout << "Pose deriv: \n";
				std::cout << "dXX_dRT: \n" << dXX_dRT << std::endl;
				std::cout << "dXX_dX: \n" << dXX_dX << std::endl;
				std::cout << "XX: \n" << XX << std::endl;
			}*/
		
			Eigen::Vector2d xu;	// Undistorted image point
			xu(0) = XX(0) / XX(2);
			xu(1) = XX(1) / XX(2);
			/*if (i == 0) {
				std::cout << "xu: \n" << xu << std::endl;
			}*/

			Eigen::Vector2d xd = x.distortions[view](xu);	// Distorted image point
			/*if (i == 0) {
				std::cout << "xd: \n" << xd << std::endl;
			}*/

			double focalLength = x.cams[view].getFocalLength();

			Eigen::Matrix2d dp_dxd = Eigen::Matrix2d::Zero();
			dp_dxd(0, 0) = focalLength;
			dp_dxd(1, 1) = focalLength;
			/*if (i == 0) {
				std::cout << "dp_dxd: \n" << dp_dxd << std::endl;
			}*/

			Matrix2x3d dxu_dXX = Matrix2x3d::Zero();
			dxu_dXX(0, 0) = 1.0 / XX(2);							  dxu_dXX(0, 2) = -XX(0) / (XX(2) * XX(2));
										 dxu_dXX(1, 1) = 1.0 / XX(2); dxu_dXX(1, 2) = -XX(1) / (XX(2) * XX(2));
			/*if (i == 0) {
				std::cout << "dxu_dXX: \n" << dxu_dXX << std::endl;
			}*/

			Eigen::Matrix2d dxd_dxu = x.distortions[view].derivativeWrtUndistortedPoint(xu);
			Eigen::Matrix2d dp_dxu = dp_dxd * dxd_dxu;
			Matrix2x3d dp_dXX = dp_dxu * dxu_dXX;
			/*if (i == 0) {
				std::cout << "dxd_dxu: \n" << dxd_dxu << std::endl;
				std::cout << "dp_dxu: \n" << dp_dxu << std::endl;
				std::cout << "dp_dXX: \n" << dp_dXX << std::endl;
			}*/

			// Compute outer derivative from psi residual
			double sqrInlierThreshold = this->inlierThreshold * this->inlierThreshold;
			//const Vector2d q = x.cams[view].projectPoint(x.distortions[view], x.data_points.col(point));
			const Vector2d q = this->projectPoint(x.cams[view], x.distortions[view], x.data_points.col(point));
			const Vector2d r = q - measurements.col(i);

    /*  if (i == 0)
      {
        std::cout << "pt = " << x.data_points.col(point) << std::endl;
        std::cout << "q = " << q << std::endl;
        std::cout << "measurements.col(i) = " << measurements.col(i) << std::endl;
      }*/
			const double r2 = r.squaredNorm();
			const double W = psi_weight(sqrInlierThreshold, r2);
			const double sqrt_psi = sqrt(psi(sqrInlierThreshold, r2));
			const double rsqrt_psi = 1.0 / std::max(eps_psi_residual, sqrt_psi);

			Matrix2d outer_deriv, r_rt, rI;
			const double rcp_r2 = 1.0 / std::max(eps_psi_residual, r2);
			const double rnorm_r = 1.0 / std::max(eps_psi_residual, sqrt(r2));
			r_rt = r * r.transpose(); r_rt *= rnorm_r;
			rI.setIdentity(); rI *= sqrt(r2);
			outer_deriv = W / 2.0 * rsqrt_psi * r_rt + sqrt_psi * rcp_r2 * (rI - r_rt);

			// Prepare Jacobian block
			Eigen::MatrixXd Jblock = Eigen::MatrixXd::Zero(2, 9 + 3);	// 9 camera params + xyz 3d point coords

			// Set deriv wrt camera parameters
			Eigen::Matrix2d dxd_dk1k2 = x.distortions[view].derivativeWrtRadialParameters(xu);
			Eigen::Matrix2d d_dk1k2 = dp_dxd * dxd_dk1k2;
			Jblock.block<2, 2>(0, 7) = d_dk1k2; 
			/*if (i == 0) {
				std::cout << "dxd_dk1k2: \n" << dxd_dk1k2 << std::endl;
				std::cout << "d_dk1k2: \n" << d_dk1k2 << std::endl;
			}*/
			//jvals.add(i * 2 + 0, cam_base + view * numCamParams + radialParamsOffset + 0, d_dk1k2(0, 0));
			//jvals.add(i * 2 + 0, cam_base + view * numCamParams + radialParamsOffset + 1, d_dk1k2(0, 1));
			//jvals.add(i * 2 + 1, cam_base + view * numCamParams + radialParamsOffset + 0, d_dk1k2(1, 0));
			//jvals.add(i * 2 + 1, cam_base + view * numCamParams + radialParamsOffset + 1, d_dk1k2(1, 1));

			Jblock.col(6) = xd;
			//jvals.add(i * 2 + 0, cam_base + view * numCamParams + focalLengthOffset, xd(0));
			//jvals.add(i * 2 + 1, cam_base + view * numCamParams + focalLengthOffset, xd(1));

			Matrix2x6d dp_dRT = dp_dXX * dXX_dRT;
			Jblock.block<2, 6>(0, 0) = dp_dRT;
			/*if (i == 0) {
				std::cout << "dp_dRT: \n" << dp_dRT << std::endl;
			}*/
			//jvals.add(i * 2 + 0, cam_base + view * numCamParams + transfParamsOffset + 0, dp_dRT(0, 0));
			//jvals.add(i * 2 + 0, cam_base + view * numCamParams + transfParamsOffset + 1, dp_dRT(0, 1));
			//jvals.add(i * 2 + 0, cam_base + view * numCamParams + transfParamsOffset + 2, dp_dRT(0, 2));
			//jvals.add(i * 2 + 0, cam_base + view * numCamParams + transfParamsOffset + 3, dp_dRT(0, 3));
			//jvals.add(i * 2 + 0, cam_base + view * numCamParams + transfParamsOffset + 4, dp_dRT(0, 4));
			//jvals.add(i * 2 + 0, cam_base + view * numCamParams + transfParamsOffset + 5, dp_dRT(0, 5));
			//jvals.add(i * 2 + 1, cam_base + view * numCamParams + transfParamsOffset + 0, dp_dRT(1, 0));
			//jvals.add(i * 2 + 1, cam_base + view * numCamParams + transfParamsOffset + 1, dp_dRT(1, 1));
			//jvals.add(i * 2 + 1, cam_base + view * numCamParams + transfParamsOffset + 2, dp_dRT(1, 2));
			//jvals.add(i * 2 + 1, cam_base + view * numCamParams + transfParamsOffset + 3, dp_dRT(1, 3));
			//jvals.add(i * 2 + 1, cam_base + view * numCamParams + transfParamsOffset + 4, dp_dRT(1, 4));
			//jvals.add(i * 2 + 1, cam_base + view * numCamParams + transfParamsOffset + 5, dp_dRT(1, 5));

			// Set deriv wrt 3d points
			Jblock.block<2, 3>(0, 9) = dp_dXX * dXX_dX;
			/*if (i == 0) {
				std::cout << "dp_dX: \n" << Jblock.block<2, 3>(0, 9) << std::endl;
			}*/
			//Matrix2x3d dp_dX = dp_dXX * dXX_dX;
			//jvals.add(i * 2 + 0, pt_base + point * numPointCoords + 0, dp_dX(0, 0));
			//jvals.add(i * 2 + 0, pt_base + point * numPointCoords + 1, dp_dX(0, 1));
			//jvals.add(i * 2 + 0, pt_base + point * numPointCoords + 2, dp_dX(0, 2));
			//jvals.add(i * 2 + 1, pt_base + point * numPointCoords + 0, dp_dX(1, 0));
			//jvals.add(i * 2 + 1, pt_base + point * numPointCoords + 1, dp_dX(1, 1));
			//jvals.add(i * 2 + 1, pt_base + point * numPointCoords + 2, dp_dX(1, 2));

			//std::cout << i * 2 << ", " << pt_base + point * numPointCoords << ":\n" << dp_dX << "----" << std::endl;
	
			// Multiply block with outer deriv

    /*  if (i == 0) {
        std::cout << "Bef" << std::endl;
        std::cout << Jblock << std::endl;
      }*/
			Jblock = outer_deriv * Jblock;

	/*		if (i == 0)
			{
				std::cout << "r2 = " << r2 << " W = " << W << " sqrt_psi = " << sqrt_psi << std::endl;
				std::cout << "outer_deriv = " << outer_deriv << std::endl;
			}

			if (i == 0) {
				std::cout << Jblock << std::endl;
			}
      */
			// Fill into the Jacobian
			// Set deriv wrt camera parameters
			jvals.add(i * 2 + 0, cam_base + view * numCamParams + radialParamsOffset + 0, Jblock(0, 7));
			jvals.add(i * 2 + 0, cam_base + view * numCamParams + radialParamsOffset + 1, Jblock(0, 8));
			jvals.add(i * 2 + 1, cam_base + view * numCamParams + radialParamsOffset + 0, Jblock(1, 7));
			jvals.add(i * 2 + 1, cam_base + view * numCamParams + radialParamsOffset + 1, Jblock(1, 8));

			jvals.add(i * 2 + 0, cam_base + view * numCamParams + focalLengthOffset, Jblock(0, 6));
			jvals.add(i * 2 + 1, cam_base + view * numCamParams + focalLengthOffset, Jblock(1, 6));

			jvals.add(i * 2 + 0, cam_base + view * numCamParams + transfParamsOffset + 0, Jblock(0, 0));
			jvals.add(i * 2 + 0, cam_base + view * numCamParams + transfParamsOffset + 1, Jblock(0, 1));
			jvals.add(i * 2 + 0, cam_base + view * numCamParams + transfParamsOffset + 2, Jblock(0, 2));
			jvals.add(i * 2 + 0, cam_base + view * numCamParams + transfParamsOffset + 3, Jblock(0, 3));
			jvals.add(i * 2 + 0, cam_base + view * numCamParams + transfParamsOffset + 4, Jblock(0, 4));
			jvals.add(i * 2 + 0, cam_base + view * numCamParams + transfParamsOffset + 5, Jblock(0, 5));
			jvals.add(i * 2 + 1, cam_base + view * numCamParams + transfParamsOffset + 0, Jblock(1, 0));
			jvals.add(i * 2 + 1, cam_base + view * numCamParams + transfParamsOffset + 1, Jblock(1, 1));
			jvals.add(i * 2 + 1, cam_base + view * numCamParams + transfParamsOffset + 2, Jblock(1, 2));
			jvals.add(i * 2 + 1, cam_base + view * numCamParams + transfParamsOffset + 3, Jblock(1, 3));
			jvals.add(i * 2 + 1, cam_base + view * numCamParams + transfParamsOffset + 4, Jblock(1, 4));
			jvals.add(i * 2 + 1, cam_base + view * numCamParams + transfParamsOffset + 5, Jblock(1, 5));

			// Set deriv wrt 3d points
			jvals.add(i * 2 + 0, pt_base + point * numPointCoords + 0, Jblock(0, 9));
			jvals.add(i * 2 + 0, pt_base + point * numPointCoords + 1, Jblock(0, 10));
			jvals.add(i * 2 + 0, pt_base + point * numPointCoords + 2, Jblock(0, 11));
			jvals.add(i * 2 + 1, pt_base + point * numPointCoords + 0, Jblock(1, 9));
			jvals.add(i * 2 + 1, pt_base + point * numPointCoords + 1, Jblock(1, 10));
			jvals.add(i * 2 + 1, pt_base + point * numPointCoords + 2, Jblock(1, 11));


			/*
			jvals.add(point * x.nCameras() + view + 0, cam_base + view * numCamParams + radialParamsOffset + 0, d_dk1k2(0, 0));
			jvals.add(point * x.nCameras() + view + 0, cam_base + view * numCamParams + radialParamsOffset + 1, d_dk1k2(0, 1));
			jvals.add(point * x.nCameras() + view + 1, cam_base + view * numCamParams + radialParamsOffset + 0, d_dk1k2(1, 0));
			jvals.add(point * x.nCameras() + view + 1, cam_base + view * numCamParams + radialParamsOffset + 1, d_dk1k2(1, 1));

			//Jblock.col(6) = xd;
			jvals.add(point * x.nCameras() + view + 0, cam_base + view * numCamParams + focalLengthOffset, xd(0));
			jvals.add(point * x.nCameras() + view + 1, cam_base + view * numCamParams + focalLengthOffset, xd(1));

			Matrix2x6d dp_dRT = dp_dXX * dXX_dRT;
			//Jblock.block<2, 6>(0, 0) = dp_dRT;
			jvals.add(point * x.nCameras() + view + 0, cam_base + view * numCamParams + transfParamsOffset + 0, dp_dRT(0, 0));
			jvals.add(point * x.nCameras() + view + 0, cam_base + view * numCamParams + transfParamsOffset + 1, dp_dRT(0, 1));
			jvals.add(point * x.nCameras() + view + 0, cam_base + view * numCamParams + transfParamsOffset + 2, dp_dRT(0, 2));
			jvals.add(point * x.nCameras() + view + 0, cam_base + view * numCamParams + transfParamsOffset + 3, dp_dRT(0, 3));
			jvals.add(point * x.nCameras() + view + 0, cam_base + view * numCamParams + transfParamsOffset + 4, dp_dRT(0, 4));
			jvals.add(point * x.nCameras() + view + 0, cam_base + view * numCamParams + transfParamsOffset + 5, dp_dRT(0, 5));
			jvals.add(point * x.nCameras() + view + 1, cam_base + view * numCamParams + transfParamsOffset + 0, dp_dRT(1, 0));
			jvals.add(point * x.nCameras() + view + 1, cam_base + view * numCamParams + transfParamsOffset + 1, dp_dRT(1, 1));
			jvals.add(point * x.nCameras() + view + 1, cam_base + view * numCamParams + transfParamsOffset + 2, dp_dRT(1, 2));
			jvals.add(point * x.nCameras() + view + 1, cam_base + view * numCamParams + transfParamsOffset + 3, dp_dRT(1, 3));
			jvals.add(point * x.nCameras() + view + 1, cam_base + view * numCamParams + transfParamsOffset + 4, dp_dRT(1, 4));
			jvals.add(point * x.nCameras() + view + 1, cam_base + view * numCamParams + transfParamsOffset + 5, dp_dRT(1, 5));

			// Set deriv wrt 3d points
			//Jblock.block<2, 3>(0, 10) = dp_dXX * dXX_dX;
			Matrix2x3d dp_dX = dp_dXX * dXX_dX;
			jvals.add(point * x.nCameras() + view + 0, pt_base + point * numPointCoords + 0, dp_dX(0, 0));
			jvals.add(point * x.nCameras() + view + 0, pt_base + point * numPointCoords + 1, dp_dX(0, 1));
			jvals.add(point * x.nCameras() + view + 0, pt_base + point * numPointCoords + 2, dp_dX(0, 2));
			jvals.add(point * x.nCameras() + view + 1, pt_base + point * numPointCoords + 0, dp_dX(1, 0));
			jvals.add(point * x.nCameras() + view + 1, pt_base + point * numPointCoords + 1, dp_dX(1, 1));
			jvals.add(point * x.nCameras() + view + 1, pt_base + point * numPointCoords + 2, dp_dX(1, 2));
			*/

			// Set deriv wrt weights
			// FixMe: Not used for now
		}
	}
	/************ UPDATES ************/
	void update_params(InputType* x, StepType const& p) {
		Index pt_base = 0;
		Index cam_base = x->nDataPoints() * 3;

		// Camera parameters ordering
		Index numPointCoords = 3;	// xyz
		Index numCamParams = 9;
		Index translationOffset = 0;
		Index rotationOffset = 3;
		Index focalLengthOffset = 6;
		Index radialParamsOffset = 7;

		// Update camera parameters
		Eigen::Vector3d T, omega;
		Eigen::Matrix3d R0, dR;
		for (int i = 0; i < x->nCameras(); i++) {
			T = x->cams[i].getTranslation();
			T += p.segment<3>(cam_base + i * numCamParams);
			x->cams[i].setTranslation(T);

			// Create incremental rotation using Rodriguez formula
			R0 = x->cams[i].getRotation();
			omega = p.segment<3>(cam_base + i * numCamParams + rotationOffset);
			Math::createRotationMatrixRodrigues(omega, dR);
			x->cams[i].setRotation(dR * R0);

			x->distortions[i].k1() += p[cam_base + i * numCamParams + radialParamsOffset];
			x->distortions[i].k2() += p[cam_base + i * numCamParams + radialParamsOffset + 1];
		
			Eigen::Matrix3d K = x->cams[i].getIntrinsic();
			K(0, 0) += p[cam_base + i * numCamParams + focalLengthOffset];
			K(1, 1) += p[cam_base + i * numCamParams + focalLengthOffset];
			x->cams[i].setIntrinsic(K);
		}


		// Update 3d point positions
		for (int i = 0; i < x->nDataPoints(); i++) {
			x->data_points.col(i) += p.segment<3>(pt_base + i * numPointCoords);
		}

		// Update weights
		// FixMe: Not used for now
	}
};

#endif

