#ifndef BA_FUNCTOR_H
#define BA_FUNCTOR_H

#include <Eigen/Eigen>

#include <Eigen/SparseCore>

#include <unsupported/Eigen/MatrixFunctions>
#include <unsupported/Eigen/LevenbergMarquardt>
#include <unsupported/Eigen/SparseExtra>
#include <unsupported/Eigen/SparseQRExtra>

using namespace Eigen;

typedef Index SparseDataType;
//typedef SuiteSparse_long SparseDataType;

struct BAFunctor : Eigen::SparseFunctor<Scalar, SparseDataType> {
	typedef Eigen::SparseFunctor<Scalar, SparseDataType> Base;
	typedef typename Base::JacobianType JacobianType;

	// Variables for optimization live in InputType
	struct InputType {
		Matrix3X control_vertices;
		
		Index nVertices() const { return control_vertices.cols(); }
	};

	// And the optimization steps are computed using VectorType.
	// For subdivs (see xx), the correspondences are of type (int, Vec2) while the updates are of type (Vec2).
	// The interactions between InputType and VectorType are restricted to:
	//   The Jacobian computation takes an InputType, and its rows must easily convert to VectorType
	//   The increment_in_place operation takes InputType and StepType. 
	typedef VectorX VectorType;

	// Functor constructor
	BAFunctor(const Matrix3X& data_points);

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
	Matrix3X data_points;

	Eigen::Index nDataPoints() const {
		return data_points.cols();
	}

	const Index numParameters;
	const Index numResiduals;
	const Index numJacobianNonzeros;
	
	// Workspace initialization
	virtual void initWorkspace();

	Scalar estimateNorm(InputType const& x, StepType const& diag);

	// Describe the QR solvers
	// QR for J1 is block diagonal
	typedef BandedBlockedSparseQR<JacobianType, NaturalOrdering<SparseDataType>, 8> LeftSuperBlockSolver;
	//typedef SPQR<JacobianType> LeftSuperBlockSolver;
	// QR for J1'J2 is general dense (faster than general sparse by about 1.5x for n=500K)
	typedef ColPivHouseholderQR<Matrix<Scalar, Dynamic, Dynamic> > RightSuperBlockSolver;
	// QR solver is concatenation of the above.
	typedef BlockAngularSparseQR<JacobianType, LeftSuperBlockSolver, RightSuperBlockSolver> SchurlikeQRSolver;
	typedef SchurlikeQRSolver QRSolver;

	// And tell the algorithm how to set the QR parameters.
	virtual void initQRSolver(SchurlikeQRSolver &qr);

	/************ UTILITY FUNCTIONS FOR ENERGIES ************/

	/************ ENERGIES, GRADIENTS and UPDATES ************/
	/************ ENERGIES ************/
	
	/************ GRADIENTS ************/
	
	/************ UPDATES ************/

};

#endif

