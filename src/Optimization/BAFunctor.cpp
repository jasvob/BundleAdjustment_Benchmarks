#include "BAFunctor.h"

#include <iostream>

BAFunctor::BAFunctor(const Matrix3X& data_points) :
	Base(data_points.cols() * 2 + 9, data_points.cols() * 3),
	data_points(data_points),
	numParameters(data_points.cols() * 2 + 9),
	numResiduals(data_points.cols() * 3),
	numJacobianNonzeros(data_points.cols() * 3 * 9 + data_points.cols() * 6) {

	initWorkspace();
}

void BAFunctor::initWorkspace() {
	Index nPoints = data_points.cols();

}

Scalar BAFunctor::estimateNorm(InputType const& x, StepType const& diag) {
	return 0;	// FixMe: Implement norm est.
}

// And tell the algorithm how to set the QR parameters.
void BAFunctor::initQRSolver(SchurlikeQRSolver &qr) {
	// set block size
	/*
	int blkRows = 3;
	int blkCols = 2;
	int blockOverlap = 0;
	qr.setSparseBlockParams(data_points.cols() * blkRows, data_points.cols() * (blkCols - blockOverlap));
	qr.getLeftSolver().setPattern(data_points.cols() * blkRows, data_points.cols() * (blkCols - blockOverlap), blkRows, blkCols, blockOverlap);
	*/
}

// Functor functions
// 1. Evaluate the residuals at x
int BAFunctor::operator()(const InputType& x, ValueType& fvec) {
	this->f_impl(x, fvec);

	return 0;
}

// 2. Evaluate jacobian at x
int BAFunctor::df(const InputType& x, JacobianType& fjac) {
	// Fill Jacobian columns.  
	Eigen::TripletArray<Scalar, typename JacobianType::Index> jvals(this->numJacobianNonzeros);

	this->df_impl(x, jvals);

	fjac.resize(this->numResiduals, this->numParameters);
	// Do not redefine the functor treating duplicate entries!!! The implementation expects to sum them up as done by default.
	fjac.setFromTriplets(jvals.begin(), jvals.end());
	fjac.makeCompressed();

	return 0;
}

void BAFunctor::increment_in_place(InputType* x, StepType const& p) {
	Index nPoints = data_points.cols();

	this->increment_in_place_impl(x, p);
}

// Functor functions
// 1. Evaluate the residuals at x
void BAFunctor::f_impl(const InputType& x, ValueType& fvec) {

}

// 2. Evaluate jacobian at x
void BAFunctor::df_impl(const InputType& x, Eigen::TripletArray<Scalar, typename JacobianType::Index>& jvals) {

}

void BAFunctor::increment_in_place_impl(InputType* x, StepType const& p) {

}