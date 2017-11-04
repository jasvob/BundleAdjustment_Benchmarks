#ifndef BA_FUNCTOR_H
#define BA_FUNCTOR_H

#include <Eigen/Eigen>

#include <Eigen/SparseCore>

#include <unsupported/Eigen/MatrixFunctions>
//#include <unsupported/Eigen/LevenbergMarquardt>
#include "../Eigen_ext/LevenbergMarquardt/LevenbergMarquardt.h"
#include "../Eigen_ext/LevenbergMarquardt/LMqrsolv.h"
#include "../Eigen_ext/LevenbergMarquardt/LMpar.h"
#include "../Eigen_ext/LevenbergMarquardt/LMcovar.h"
#include "../Eigen_ext/LevenbergMarquardt/LMonestep.h"
#include <unsupported/Eigen/SparseExtra>
#include <unsupported/Eigen/SparseQRExtra>

#include <Eigen/SparseCore>
#include <Eigen/src/Core/util/DisableStupidWarnings.h>
#include <suitesparse/SuiteSparseQR.hpp>
#include <Eigen/src/CholmodSupport/CholmodSupport.h>
#include <Eigen/src/SPQRSupport/SuiteSparseQRSupport.h>

#include "../BlockDiagonalSparseQR_Ext.h"
#include "../DenseBlockedThinSparseQR.h"

#include "../BlockAngularSparseQR_Ext.h"

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
	typedef BlockDiagonalSparseQR_Ext<JacobianType, DenseBlockSolver> LeftSuperBlockSolver;
	typedef DenseBlockedThinSparseQR<JacobianType, NaturalOrdering<SparseDataType>, 4, true> RightSuperBlockSolver;
	typedef BlockAngularSparseQR_Ext<JacobianType, LeftSuperBlockSolver, RightSuperBlockSolver> SchurlikeQRSolver;
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
		d_dRT.leftCols(3) = J;

		// The derivative of Rx + t wrt x is just R
		d_dX = cam.getRotation();
	}

	/************ ENERGIES, GRADIENTS and UPDATES ************/
	/************ ENERGIES ************/
	// Residual wrt to 3d points
	void E_pos(const InputType& x, const Eigen::Matrix2Xd& measurements, const std::vector<int> &correspondingView, const std::vector<int> &correspondingPoint, ValueType& fvec) {
		
		for (int i = 0; i < this->nMeasurements(); i++) {
			int view = correspondingView[i];
			int point = correspondingPoint[i];

			// Project 3D point into corresponding camera view
			Eigen::Vector2d q = x.cams[view].projectPoint(x.distortions[view], x.data_points.col(point));

			// Compute residual for the point
			fvec.segment<2>(i * 2) = q - measurements.col(i);
			//fvec.segment<2>(point * x.nCameras() + view) = q - measurements.col(i);
		}
	}
	
	/************ GRADIENTS ************/
	void dE_pos(const InputType& x, Eigen::TripletArray<Scalar, typename JacobianType::Index>& jvals) {

		Index pt_base = 0;
		Index cam_base = x.nDataPoints() * 3;

		// Camera parameters ordering
		Index numPointCoords = 3;	// xyz
		Index numCamParams = 10;
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
		
			this->poseDerivatives(x.cams[view], x.data_points.col(point), XX, dXX_dRT, dXX_dX);
		
			Eigen::Vector2d xu;	// Undistorted image point
			xu(0) = XX(0) / XX(2);
			xu(1) = XX(1) / XX(2);

			Eigen::Vector2d xd = x.distortions[view](xu);	// Distorted image point

			double focalLength = x.cams[view].getFocalLength();

			Eigen::Matrix2d dp_dxd = Eigen::Matrix2d::Zero();;
			dp_dxd(0, 0) = focalLength;
			dp_dxd(1, 1) = focalLength;

			Matrix2x3d dxu_dXX = Matrix2x3d::Zero();
			dxu_dXX(0, 0) = 1.0 / XX(2);							  dxu_dXX(0, 2) = -XX(0) / (XX(2) * XX(2));
										 dxu_dXX(1, 1) = 1.0 / XX(2); dxu_dXX(1, 2) = -XX(1) / (XX(2) * XX(2));

			Eigen::Matrix2d dxd_dxu = x.distortions[view].derivativeWrtUndistortedPoint(xu);
			Eigen::Matrix2d dp_dxu = dp_dxd * dxd_dxu;
			Matrix2x3d dp_dXX = dp_dxu * dxu_dXX;

			// Prepare Jacobian block
			//Eigen::MatrixXd Jblock = Eigen::MatrixXd::Zero(2, 10 + 3);	// 10 camera params + xyz 3d point coords

			// Set deriv wrt camera parameters
			Eigen::Matrix2d dxd_dk1k2 = x.distortions[view].derivativeWrtRadialParameters(xu);
			Eigen::Matrix2d d_dk1k2 = dp_dxd * dxd_dk1k2;
			//Jblock.block<2, 2>(0, 7) = d_dk1k2;
			jvals.add(i * 2 + 0, cam_base + view * numCamParams + radialParamsOffset + 0, d_dk1k2(0, 0));
			jvals.add(i * 2 + 0, cam_base + view * numCamParams + radialParamsOffset + 1, d_dk1k2(0, 1));
			jvals.add(i * 2 + 1, cam_base + view * numCamParams + radialParamsOffset + 0, d_dk1k2(1, 0));
			jvals.add(i * 2 + 1, cam_base + view * numCamParams + radialParamsOffset + 1, d_dk1k2(1, 1));

			//Jblock.col(6) = xd;
			jvals.add(i * 2 + 0, cam_base + view * numCamParams + focalLengthOffset, xd(0));
			jvals.add(i * 2 + 1, cam_base + view * numCamParams + focalLengthOffset, xd(1));

			Matrix2x6d dp_dRT = dp_dXX * dXX_dRT;
			//Jblock.block<2, 6>(0, 0) = dp_dRT;
			jvals.add(i * 2 + 0, cam_base + view * numCamParams + transfParamsOffset + 0, dp_dRT(0, 0));
			jvals.add(i * 2 + 0, cam_base + view * numCamParams + transfParamsOffset + 1, dp_dRT(0, 1));
			jvals.add(i * 2 + 0, cam_base + view * numCamParams + transfParamsOffset + 2, dp_dRT(0, 2));
			jvals.add(i * 2 + 0, cam_base + view * numCamParams + transfParamsOffset + 3, dp_dRT(0, 3));
			jvals.add(i * 2 + 0, cam_base + view * numCamParams + transfParamsOffset + 4, dp_dRT(0, 4));
			jvals.add(i * 2 + 0, cam_base + view * numCamParams + transfParamsOffset + 5, dp_dRT(0, 5));
			jvals.add(i * 2 + 1, cam_base + view * numCamParams + transfParamsOffset + 0, dp_dRT(1, 0));
			jvals.add(i * 2 + 1, cam_base + view * numCamParams + transfParamsOffset + 1, dp_dRT(1, 1));
			jvals.add(i * 2 + 1, cam_base + view * numCamParams + transfParamsOffset + 2, dp_dRT(1, 2));
			jvals.add(i * 2 + 1, cam_base + view * numCamParams + transfParamsOffset + 3, dp_dRT(1, 3));
			jvals.add(i * 2 + 1, cam_base + view * numCamParams + transfParamsOffset + 4, dp_dRT(1, 4));
			jvals.add(i * 2 + 1, cam_base + view * numCamParams + transfParamsOffset + 5, dp_dRT(1, 5));

			// Set deriv wrt 3d points
			//Jblock.block<2, 3>(0, 10) = dp_dXX * dXX_dX;
			Matrix2x3d dp_dX = dp_dXX * dXX_dX;
			jvals.add(i * 2 + 0, pt_base + point * numPointCoords + 0, dp_dX(0, 0));
			jvals.add(i * 2 + 0, pt_base + point * numPointCoords + 1, dp_dX(0, 1));
			jvals.add(i * 2 + 0, pt_base + point * numPointCoords + 2, dp_dX(0, 2));
			jvals.add(i * 2 + 1, pt_base + point * numPointCoords + 0, dp_dX(1, 0));
			jvals.add(i * 2 + 1, pt_base + point * numPointCoords + 1, dp_dX(1, 1));
			jvals.add(i * 2 + 1, pt_base + point * numPointCoords + 2, dp_dX(1, 2));
			//std::cout << i * 2 << ", " << pt_base + point * numPointCoords << ":\n" << dp_dX << "----" << std::endl;

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
		Index numCamParams = 10;
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

