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
#include <Eigen/SparseCholesky>
#include <unsupported/Eigen/src/SparseQRExtra/BlockAngularSparseQRPartial.h>

#include "../CameraMatrix.h"
#include "../DistortionFunction.h"

#include "../MathUtils.h"

using namespace Eigen;

#ifdef QRSPQR
typedef SuiteSparse_long SparseDataType;
#else
typedef int SparseDataType;
#endif

struct BAFunctor : Eigen::SparseFunctor<Scalar, SparseDataType> {
	typedef Eigen::SparseFunctor<Scalar, SparseDataType> Base;
	typedef typename Base::JacobianType JacobianType;

	// Variables for optimization live in InputType
	struct InputType {
		CameraMatrix::Vector cams;
		DistortionFunction::Vector distortions;
		Matrix3XX data_points;
		VectorXX weights;

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
	typedef VectorXX VectorType;
		
	// Functor constructor
	BAFunctor(const Eigen::Index numPoints, const Eigen::Index numCameras, const Matrix2XX &measurements,
		const std::vector<int> &correspondingView, const std::vector<int> &correspondingPoint, 
		Scalar inlierThreshold);

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
	Matrix2XX measurements;
	std::vector<int> correspondingView;
	std::vector<int> correspondingPoint;
	const Scalar inlierThreshold;
	
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

#ifdef QRKIT
  typedef ColPivHouseholderQR<Matrix<Scalar, Dynamic, Dynamic> > DenseBlockSolver;
  typedef BlockDiagonalSparseQR<JacobianType, DenseBlockSolver> LeftSuperBlockSolver;
  typedef DenseBlockedThinQR<Matrix<Scalar, Dynamic, Dynamic>, NaturalOrdering<SparseDataType>, 4, true > RightSuperBlockSolver;
  typedef BlockAngularSparseQR<JacobianType, LeftSuperBlockSolver, RightSuperBlockSolver> SchurlikeQRSolver;
#elif QRCHOL
  typedef ColPivHouseholderQR<Matrix<Scalar, Dynamic, Dynamic> > DenseBlockSolver;
  typedef BlockDiagonalSparseQR<JacobianType, DenseBlockSolver> LeftSuperBlockSolver;
  typedef Eigen::SimplicialLDLT<JacobianType, Eigen::Lower> RightSuperBlockSolver;
  typedef BlockAngularSparseQRPartial<JacobianType, LeftSuperBlockSolver, RightSuperBlockSolver> SchurlikeQRSolver;
#elif MOREQR
  typedef ColPivHouseholderQR<Matrix<Scalar, Dynamic, Dynamic> > DenseBlockSolver;
  typedef BlockDiagonalSparseQR<JacobianType, DenseBlockSolver> LeftSuperBlockSolver;
  typedef DenseBlockedThinQR<Matrix<Scalar, Dynamic, Dynamic>, NaturalOrdering<SparseDataType>, 4, true > RightSuperBlockSolver;
  typedef BlockAngularSparseQR<JacobianType, LeftSuperBlockSolver, RightSuperBlockSolver> SchurlikeQRSolver;
#elif CHOLESKY
  typedef SPQR<JacobianType> SchurlikeQRSolver;
#elif QRSPQR
  typedef SPQR<JacobianType> SchurlikeQRSolver;
#endif

	typedef SchurlikeQRSolver QRSolver;

	// And tell the algorithm how to set the QR parameters.
	virtual void initQRSolver(SchurlikeQRSolver &qr);
  virtual void initQRSolverInner(SchurlikeQRSolver &qr);

	/************ UTILITY FUNCTIONS FOR ENERGIES ************/
	void poseDerivatives(const CameraMatrix &cam, const Vector3X &pt, Vector3X &XX, Matrix3x6X &d_dRT, Matrix3X &d_dX) {
		XX = cam.transformPointIntoCameraSpace(pt);

		// See Frank Dellaerts bundle adjustment tutorial.
		// d(dR * R0 * X + t) / d omega = -[R0 * X]_x
		Matrix3X J;
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
	inline Scalar psi(Scalar const tau2, Scalar const r2) { return (r2 < tau2) ? r2*(Scalar(2.0) - r2 / tau2) / Scalar(4.0) : tau2 / Scalar(4.0); }
	inline Scalar psi_weight(Scalar const tau2, Scalar const r2) { return std::max(Scalar(0.0), Scalar(1.0) - r2 / tau2); }
	inline Scalar psi_hat(Scalar const tau2, Scalar const r2, Scalar const w2) { return w2*r2 + tau2 / Scalar(2.0) * (w2 - Scalar(1.0)) * (w2 - Scalar(1.0)); }

	Vector2X projectPoint(const CameraMatrix &cam, const DistortionFunction &distortion, const Vector3X &X) {
		Vector3X XX = cam.transformPointIntoCameraSpace(X);
		Vector2X xu(XX(0) / XX(2), XX(1) / XX(2));
		Vector2X xd = distortion(xu);
		return cam.getFocalLength() * xd;
	}

	//const double eps_psi_residual = 1e-20;
  const Scalar eps_psi_residual = Scalar(1e-15);
	void E_pos(const InputType& x, const Matrix2XX& measurements, const std::vector<int> &correspondingView, const std::vector<int> &correspondingPoint, ValueType& fvec) {
		Scalar sqrInlierThreshold = this->inlierThreshold * this->inlierThreshold;

		for (int i = 0; i < this->nMeasurements(); i++) {
			int view = correspondingView[i];
			int point = correspondingPoint[i];

			// Project 3D point into corresponding camera view
			Vector2X q = this->projectPoint(x.cams[view], x.distortions[view], x.data_points.col(point));
			Vector2X r = (q - measurements.col(i));

			Scalar sqrt_psi = sqrt(psi(sqrInlierThreshold, r.squaredNorm()));
			Scalar rnorm_r = Scalar(1.0) / std::max(eps_psi_residual, r.norm());

			// Compute residual for the point
			fvec(i * 2 + 0) = r(0) * sqrt_psi * rnorm_r;
			fvec(i * 2 + 1) = r(1) * sqrt_psi * rnorm_r;
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
		
		Vector3X XX;
		Matrix3x6X dXX_dRT;
		Matrix3X dXX_dX;
		int view, point;
		for (int i = 0; i < this->nMeasurements(); i++) {
			XX.setZero();
			dXX_dRT.setZero();
			dXX_dX.setZero();

			view = correspondingView[i];
			point = correspondingPoint[i];

			this->poseDerivatives(x.cams[view], x.data_points.col(point), XX, dXX_dRT, dXX_dX);
		
			Vector2X xu;	// Undistorted image point
			xu(0) = XX(0) / XX(2);
			xu(1) = XX(1) / XX(2);

			Vector2X xd = x.distortions[view](xu);	// Distorted image point

			Scalar focalLength = x.cams[view].getFocalLength();

			Matrix2X dp_dxd = Matrix2X::Zero();
			dp_dxd(0, 0) = focalLength;
			dp_dxd(1, 1) = focalLength;

			Matrix2x3X dxu_dXX = Matrix2x3X::Zero();
			dxu_dXX(0, 0) = Scalar(1.0) / XX(2);							  dxu_dXX(0, 2) = -XX(0) / (XX(2) * XX(2));
										 dxu_dXX(1, 1) = Scalar(1.0) / XX(2); dxu_dXX(1, 2) = -XX(1) / (XX(2) * XX(2));

			Matrix2X dxd_dxu = x.distortions[view].derivativeWrtUndistortedPoint(xu);
			Matrix2X dp_dxu = dp_dxd * dxd_dxu;
			Matrix2x3X dp_dXX = dp_dxu * dxu_dXX;

			// Compute outer derivative from psi residual
			Scalar sqrInlierThreshold = this->inlierThreshold * this->inlierThreshold;
			const Vector2X q = this->projectPoint(x.cams[view], x.distortions[view], x.data_points.col(point));
			const Vector2X r = q - measurements.col(i);

			const Scalar r2 = r.squaredNorm();
			const Scalar W = psi_weight(sqrInlierThreshold, r2);
			const Scalar sqrt_psi = sqrt(psi(sqrInlierThreshold, r2));
			const Scalar rsqrt_psi = Scalar(1.0) / std::max(eps_psi_residual, sqrt_psi);

			Matrix2X outer_deriv, r_rt, rI;
			const Scalar rcp_r2 = Scalar(1.0) / std::max(eps_psi_residual, r2);
			const Scalar rnorm_r = Scalar(1.0) / std::max(eps_psi_residual, sqrt(r2));
			r_rt = r * r.transpose(); r_rt *= rnorm_r;
			rI.setIdentity(); rI *= sqrt(r2);
			outer_deriv = W / Scalar(2.0) * rsqrt_psi * r_rt + sqrt_psi * rcp_r2 * (rI - r_rt);

			// Prepare Jacobian block
			MatrixXX Jblock = MatrixXX::Zero(2, 9 + 3);	// 9 camera params + xyz 3d point coords

			// Set deriv wrt camera parameters
			Matrix2X dxd_dk1k2 = x.distortions[view].derivativeWrtRadialParameters(xu);
			Matrix2X d_dk1k2 = dp_dxd * dxd_dk1k2;
			Jblock.block<2, 2>(0, 7) = d_dk1k2; 

			Jblock.col(6) = xd;

			Matrix2x6X dp_dRT = dp_dXX * dXX_dRT;
			Jblock.block<2, 6>(0, 0) = dp_dRT;

			// Set deriv wrt 3d points
			Jblock.block<2, 3>(0, 9) = dp_dXX * dXX_dX;
	
			// Multiply block with outer deriv
			Jblock = outer_deriv * Jblock;

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
		Vector3X T, omega;
		Matrix3X R0, dR;
		for (int i = 0; i < x->nCameras(); i++) {
      T = x->cams[i].getTranslation();
			T += p.segment<3>(cam_base + i * numCamParams);
			x->cams[i].setTranslation(T);

			// Create incremental rotation using Rodriguez formula
			R0 = x->cams[i].getRotation();
			omega = p.segment<3>(cam_base + i * numCamParams + rotationOffset);
			Math::createRotationMatrixRodrigues(omega, dR);
			x->cams[i].setRotation(dR * R0);

			x->distortions[i].k1() += p(cam_base + i * numCamParams + radialParamsOffset);
			x->distortions[i].k2() += p(cam_base + i * numCamParams + radialParamsOffset + 1);
		
			Matrix3X K = x->cams[i].getIntrinsic();
			K(0, 0) += p(cam_base + i * numCamParams + focalLengthOffset);
			K(1, 1) += p(cam_base + i * numCamParams + focalLengthOffset);
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

