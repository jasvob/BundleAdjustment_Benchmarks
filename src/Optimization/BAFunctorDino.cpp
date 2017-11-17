#include "BAFunctorDino.h"

#include <iostream>

BAFunctorDino::BAFunctorDino(const Eigen::Index numPoints, const Eigen::Index numCameras, const Eigen::Matrix2Xd &measurements,
	const std::vector<int> &correspondingView, const std::vector<int> &correspondingPoint,
	double inlierThreshold) 
	: Base(numPoints * 3 + numCameras * 9, measurements.cols() * 2),
	measurements(measurements),
	correspondingView(correspondingView),
	correspondingPoint(correspondingPoint),
	inlierThreshold(inlierThreshold),
	numParameters(numPoints * 3 + numCameras * 9),
	numPointParams(numPoints * 3),
	numResiduals(measurements.cols() * 2),
	numJacobianNonzeros(measurements.cols() * 2 * 3 + 9 * numCameras) {

	initWorkspace();
}

void BAFunctorDino::initWorkspace() {

}

Scalar BAFunctorDino::estimateNorm(InputType const& x, StepType const& diag) {
	Index pt_base = 0;
	Index cam_base = x.nDataPoints() * 3;

	// Camera parameters ordering
	Index numPointCoords = 3;	// xyz
	Index numCamParams = 9;
	Index translationOffset = 0;
	Index rotationOffset = 3;
	Index focalLengthOffset = 6;
	Index radialParamsOffset = 7;

	double total = 0.0;
	Eigen::Vector3d T, omega;
	Eigen::Vector2d k12;
	double focalLength = 0.0;
	Eigen::Matrix3d R;
	for (int i = 0; i < x.nCameras(); i++) {
		T = x.cams[i].getTranslation();
		R = x.cams[i].getRotation();
		Math::createRodriguesParamFromRotationMatrix(R, omega);
		k12 = Eigen::Vector2d(x.distortions[i].getK1(), x.distortions[i].getK2());
		focalLength = x.cams[i].getFocalLength();

		total += T.cwiseProduct(diag.segment<3>(cam_base + i * numCamParams)).stableNorm();
		total += omega.cwiseProduct(diag.segment<3>(cam_base + i * numCamParams + rotationOffset)).stableNorm();
		total += k12.cwiseProduct(diag.segment<2>(cam_base + i * numCamParams + radialParamsOffset)).stableNorm();
		total += sqrt((focalLength * diag[cam_base + i * numCamParams + focalLengthOffset]) * (focalLength * diag[cam_base + i * numCamParams + focalLengthOffset]));
	}
	total = total * total;


	Map<VectorXd> xtop{ (double*)x.data_points.data(), x.nDataPoints() * 3 };
	total += xtop.cwiseProduct(diag.head(x.nDataPoints() * 3)).squaredNorm();

	return Scalar(sqrt(total));

	/*
	Eigen::Vector3d pt;
	for (int i = 0; i < x.nDataPoints(); i++) {
		pt = 
	}
	*/
}

// And tell the algorithm how to set the QR parameters.
void BAFunctorDino::initQRSolver(SchurlikeQRSolver &qr) {
	// set block size
	//int blkRows = 2;
	//int blkCols = 3;
	//int blockOverlap = 0;
	qr.setSparseBlockParams(this->measurements.cols() * 2 + this->numPointParams, this->numPointParams);
	//qr.getLeftSolver().setPattern(data_points.cols() * blkRows, data_points.cols() * (blkCols - blockOverlap), blkRows, blkCols, blockOverlap);
}

// Functor functions
// 1. Evaluate the residuals at x
int BAFunctorDino::operator()(const InputType& x, ValueType& fvec) {
	this->f_impl(x, fvec);

	return 0;
}

// 2. Evaluate jacobian at x
int BAFunctorDino::df(const InputType& x, JacobianType& fjac) {
	// Fill Jacobian columns.  
	Eigen::TripletArray<Scalar, typename JacobianType::Index> jvals(this->numJacobianNonzeros);

	this->df_impl(x, jvals);

	fjac.resize(this->numResiduals, this->numParameters);
	// Do not redefine the functor treating duplicate entries!!! The implementation expects to sum them up as done by default.
	fjac.setFromTriplets(jvals.begin(), jvals.end());
	fjac.makeCompressed();

	return 0;
}

void BAFunctorDino::increment_in_place(InputType* x, StepType const& p) {


	this->increment_in_place_impl(x, p);
}

// Functor functions
// 1. Evaluate the residuals at x
void BAFunctorDino::f_impl(const InputType& x, ValueType& fvec) {
	this->E_pos(x, this->measurements, this->correspondingView, this->correspondingPoint, fvec);
}

// 2. Evaluate jacobian at x
void BAFunctorDino::df_impl(const InputType& x, Eigen::TripletArray<Scalar, typename JacobianType::Index>& jvals) {
	this->dE_pos(x, jvals);
}

void BAFunctorDino::increment_in_place_impl(InputType* x, StepType const& p) {
	// The parameters are passed to the optimization as:
	this->update_params(x, p);
}