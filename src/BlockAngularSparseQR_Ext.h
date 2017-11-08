// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2017 Jan Svoboda <jan.svoboda@usi.ch>
// Copyright (C) 2016 Andrew Fitzgibbon <awf@microsoft.com>
// Copyright (C) 2016 Sergio Garrido Jurado <>
// Copyright (C) 2012-2013 Desire Nuentsa <desire.nuentsa_wakam@inria.fr>
// Copyright (C) 2012-2014 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_BLOCK_ANGULAR_SPARSE_QR_EXT_H
#define EIGEN_BLOCK_ANGULAR_SPARSE_QR_EXT_H

#include "Logger.h"
#include <algorithm>
#include <ctime>
#include "SparseQR_Ext.h"
#include "BandedBlockedSparseQR_Ext.h"

template <typename Derived>
void logMatrixCSV2(Eigen::MatrixBase<Derived> const& J, char const* filename) {
	std::ofstream f(filename);
	if (!f.good()) {
		std::stringstream ss;
		ss << "Failed to open [" << filename << "] for writing";
		return;
	}

	std::stringstream ss;
	std::cout << "Writing " << J.rows() << "x" << J.cols() << " dense to \"" << filename << "\" in CSV format." << std::endl;
	f << J.format(Logger::CSVFormat);
}

namespace Eigen {

	//std::type_info StorageKindDense = typeid(SparseMatrix<Scalar>::StorageKind);
	//std::type_info StorageKindSparse = typeid(Matrix<Scalar, Dynamic, Dynamic>::StorageKind);

	template < typename MatrixType, typename LeftSolver, typename RightSolver > class BlockAngularSparseQR_Ext;
	template<typename SparseQRType> struct BlockAngularSparseQR_ExtMatrixQReturnType;
	template<typename SparseQRType> struct BlockAngularSparseQR_ExtMatrixQTransposeReturnType;
	template<typename SparseQRType, typename Derived> struct BlockAngularSparseQR_Ext_QProduct;

	namespace internal {

		// traits<SparseQRMatrixQ[Transpose]>
		template <typename SparseQRType> struct traits<BlockAngularSparseQR_ExtMatrixQReturnType<SparseQRType> >
		{
			typedef typename SparseQRType::MatrixType ReturnType;
			typedef typename ReturnType::StorageIndex StorageIndex;
			typedef typename ReturnType::StorageKind StorageKind;
			enum {
				RowsAtCompileTime = Dynamic,
				ColsAtCompileTime = Dynamic
			};
		};

		template <typename SparseQRType> struct traits<BlockAngularSparseQR_ExtMatrixQTransposeReturnType<SparseQRType> >
		{
			typedef typename SparseQRType::MatrixType ReturnType;
		};

		template <typename SparseQRType, typename Derived> struct traits<BlockAngularSparseQR_Ext_QProduct<SparseQRType, Derived> >
		{
			typedef typename Derived::PlainObject ReturnType;
		};
	} // End namespace internal


	  /**
	  * \ingroup SparseQR_Module
	  * \class BlockAngularSparseQR_Ext
	  * \brief QR factorization of block matrix, specifying subblock solvers
	  *
	  * This implementation is restricted to 1x2 block structure, factorizing
	  * matrix A = [A1 A2].
	  *
	  * \tparam _BlockQRSolverLeft The type of the QR solver which will factorize A1
	  * \tparam _BlockQRSolverRight The type of the QR solver which will factorize Q1'*A2
	  *
	  * \implsparsesolverconcept
	  *
	  */

	template<typename _MatrixType, typename _BlockQRSolverLeft, typename _BlockQRSolverRight>
	class BlockAngularSparseQR_Ext : public SparseSolverBase<BlockAngularSparseQR_Ext<_MatrixType, _BlockQRSolverLeft, _BlockQRSolverRight> >
	{
	protected:
		typedef BlockAngularSparseQR_Ext<_MatrixType, _BlockQRSolverLeft, _BlockQRSolverRight> this_t;
		typedef SparseSolverBase<BlockAngularSparseQR_Ext<_MatrixType, _BlockQRSolverLeft, _BlockQRSolverRight> > Base;
		using Base::m_isInitialized;
	public:
		using Base::_solve_impl;
		typedef _MatrixType MatrixType;
		typedef _BlockQRSolverLeft BlockQRSolverLeft;
		typedef _BlockQRSolverRight BlockQRSolverRight;
		typedef typename BlockQRSolverLeft::MatrixType LeftBlockMatrixType;
		typedef typename BlockQRSolverRight::MatrixType RightBlockMatrixType;
		typedef typename BlockQRSolverLeft::MatrixQType LeftBlockMatrixQType;
		typedef typename BlockQRSolverRight::MatrixQType RightBlockMatrixQType;
		typedef typename MatrixType::Scalar Scalar;
		typedef typename MatrixType::RealScalar RealScalar;
		typedef typename MatrixType::StorageIndex StorageIndex;
		typedef typename MatrixType::Index Index;
		typedef Matrix<StorageIndex, Dynamic, 1> IndexVector;
		typedef Matrix<Scalar, Dynamic, 1> ScalarVector;

		typedef BlockAngularSparseQR_ExtMatrixQReturnType<BlockAngularSparseQR_Ext> MatrixQType;
		typedef SparseMatrix<Scalar, ColMajor, StorageIndex> MatrixRType;
		typedef PermutationMatrix<Dynamic, Dynamic, StorageIndex> PermutationMatrixType;
		typedef PermutationMatrix<Dynamic, Dynamic, StorageIndex> PermutationType;

		enum {
			ColsAtCompileTime = MatrixType::ColsAtCompileTime,
			MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime
		};

//#define RIGHT_BLOCK_SPARSE (typeid(BlockAngularSparseQR_Ext::RightBlockMatrixType::StorageKind).hash_code() == typeid(SparseMatrix<Scalar>::StorageKind).hash_code())

	public:
		BlockAngularSparseQR_Ext() : m_analysisIsok(false), m_lastError(""), m_isQSorted(false), m_blockCols(1)
		{ }

		/** Construct a QR factorization of the matrix \a mat.
		*
		* \warning The matrix \a mat must be in compressed mode (see SparseMatrix::makeCompressed()).
		*
		* \sa compute()
		*/
		explicit BlockAngularSparseQR_Ext(const MatrixType& mat) : m_analysisIsok(false), m_lastError(""), m_isQSorted(false), m_blockCols(1)
		{
			compute(mat);
		}

		/** Computes the QR factorization of the sparse matrix \a mat.
		*
		* \warning The matrix \a mat must be in compressed mode (see SparseMatrix::makeCompressed()).
		*
		* \sa analyzePattern(), factorize()
		*/
		void compute(const MatrixType& mat)
		{
			analyzePattern(mat);
			factorize(mat);
		}
		void analyzePattern(const MatrixType& mat);
		void factorize(const MatrixType& mat);

		/** \returns the number of rows of the represented matrix.
		*/
		inline Index rows() const { return m_R.rows(); }

		/** \returns the number of columns of the represented matrix.
		*/
		inline Index cols() const { return m_R.cols(); }

		/** \returns a const reference to the \b sparse upper triangular matrix R of the QR factorization.
		* \warning The entries of the returned matrix are not sorted. This means that using it in algorithms
		*          expecting sorted entries will fail. This include random coefficient accesses (SpaseMatrix::coeff()),
		*          and coefficient-wise operations. Matrix products and triangular solves are fine though.
		*
		* To sort the entries, you can assign it to a row-major matrix, and if a column-major matrix
		* is required, you can copy it again:
		* \code
		* SparseMatrix<double>          R  = qr.matrixR();  // column-major, not sorted!
		* SparseMatrix<double,RowMajor> Rr = qr.matrixR();  // row-major, sorted
		* SparseMatrix<double>          Rc = Rr;            // column-major, sorted
		* \endcode
		*/
		const MatrixRType& matrixR() const { return m_R; }

		/** \returns the number of non linearly dependent columns as determined by the pivoting threshold.
		*
		* \sa setPivotThreshold()
		*/
		Index rank() const
		{
			eigen_assert(m_isInitialized && "The factorization should be called first, use compute()");
			return m_nonzeropivots;
		}

		/** \returns the matrix Q
		*/
		BlockAngularSparseQR_ExtMatrixQReturnType<BlockAngularSparseQR_Ext> matrixQ() const
		{
			return BlockAngularSparseQR_ExtMatrixQReturnType<BlockAngularSparseQR_Ext>(*this); // xxawf pass pointer not ref
		}

		/** \returns a const reference to the column permutation P that was applied to A such that A*P = Q*R
		* It is the combination of the fill-in reducing permutation and numerical column pivoting.
		*/
		const PermutationType& colsPermutation() const
		{
			eigen_assert(m_isInitialized && "Decomposition is not initialized.");
			return m_outputPerm_c;
		}

		const PermutationType& rowsPermutation() const {
			eigen_assert(m_isInitialized && "Decomposition is not initialized.");
			return this->m_rowPerm;
		}

		const PermutationType& rowperm1() const {
			return this->rp1;
		}
		const PermutationType& rowperm2() const {
			return this->rp2;
		}
		/** \returns A string describing the type of error.
		* This method is provided to ease debugging, not to handle errors.
		*/
		std::string lastErrorMessage() const { return m_lastError; }

		/** \internal */
		template<typename Rhs, typename Dest>
		bool _solve_impl(const MatrixBase<Rhs> &B, MatrixBase<Dest> &dest) const
		{
			eigen_assert(m_isInitialized && "The factorization should be called first, use compute()");
			eigen_assert(this->rows() == B.rows() && "SparseQR::solve() : invalid number of rows in the right hand side matrix");

			Index rank = this->rank();

			// Compute Q^T * b;
			typename Dest::PlainObject y, b;
			y = this->matrixQ().transpose() * B;
			b = y;

			// Solve with the triangular matrix R
			y.resize((std::max<Index>)(cols(), y.rows()), y.cols());
			y.topRows(rank) = this->matrixR().topLeftCorner(rank, rank).template triangularView<Upper>().solve(b.topRows(rank));
			y.bottomRows(y.rows() - rank).setZero();

			// Apply the column permutation
			if (colsPermutation().size() > 0)
				dest = colsPermutation() * y.topRows(cols());
			else
				dest = y.topRows(cols());

			m_info = Success;
			return true;
		}

		/** Sets the threshold that is used to determine linearly dependent columns during the factorization.
		*
		* In practice, if during the factorization the norm of the column that has to be eliminated is below
		* this threshold, then the entire column is treated as zero, and it is moved at the end.
		*/
		void setPivotThreshold(const RealScalar& threshold)
		{
			// No pivoting ...
		}

		/** \returns the solution X of \f$ A X = B \f$ using the current decomposition of A.
		*
		* \sa compute()
		*/
		template<typename Rhs>
		inline const Solve<BlockAngularSparseQR_Ext, Rhs> solve(const MatrixBase<Rhs>& B) const
		{
			eigen_assert(m_isInitialized && "The factorization should be called first, use compute()");
			eigen_assert(this->rows() == B.rows() && "SparseQR::solve() : invalid number of rows in the right hand side matrix");
			return Solve<BlockAngularSparseQR_Ext, Rhs>(*this, B.derived());
		}
		template<typename Rhs>
		inline const Solve<BlockAngularSparseQR_Ext, Rhs> solve(const SparseMatrixBase<Rhs>& B) const
		{
			eigen_assert(m_isInitialized && "The factorization should be called first, use compute()");
			eigen_assert(this->rows() == B.rows() && "SparseQR::solve() : invalid number of rows in the right hand side matrix");
			return Solve<BlockAngularSparseQR_Ext, Rhs>(*this, B.derived());
		}

		/** \brief Reports whether previous computation was successful.
		*
		* \returns \c Success if computation was successful,
		*          \c NumericalIssue if the QR factorization reports a numerical problem
		*          \c InvalidInput if the input matrix is invalid
		*
		* \sa iparm()
		*/
		ComputationInfo info() const
		{
			eigen_assert(m_isInitialized && "Decomposition is not initialized.");
			return m_info;
		}


		/** \internal */
		inline void _sort_matrix_Q()
		{
			if (this->m_isQSorted) return;
			// The matrix Q is sorted during the transposition
			SparseMatrix<Scalar, RowMajor, Index> mQrm(this->m_Q);
			this->m_Q = mQrm;
			this->m_isQSorted = true;
		}

		void setSparseBlockParams(Index blockRows, Index blockCols) {
			m_blockRows = blockRows;
			m_blockCols = blockCols;
		}
		
		Index leftBlockRows() const {
			return this->m_blockRows;
		}
		Index leftBlockCols() const {
			return this->m_blockCols;
		}
		
		BlockQRSolverLeft& getLeftSolver() { return m_leftSolver; }
		BlockQRSolverRight& getRightSolver() { return m_rightSolver; }


	protected:
		bool m_analysisIsok;
		bool m_factorizationIsok;
		mutable ComputationInfo m_info;
		std::string m_lastError;

		MatrixRType m_R;                // The triangular factor matrix
		typename BlockQRSolverLeft::MatrixQType m_Q1;

		PermutationType m_outputPerm_c; // The final column permutation
		PermutationType m_rowPerm;		// The final row permutation
		Index m_nonzeropivots;          // Number of non zero pivots found
		IndexVector m_etree;            // Column elimination tree
		IndexVector m_firstRowElt;      // First element in each row
		bool m_isQSorted;               // whether Q is sorted or not

		Index m_blockCols;                // Cols of first block
		Index m_blockRows;				  // Rows of the first block
										  // Every row below the first block is treated as a part of already upper triangular block)
		BlockQRSolverLeft m_leftSolver;
		BlockQRSolverRight m_rightSolver;

		template <typename, typename > friend struct BlockAngularSparseQR_Ext_QProduct;

	};

	/** \brief Preprocessing step of a QR factorization
	*
	* \warning The matrix \a mat must be in compressed mode (see SparseMatrix::makeCompressed()).
	*
	* In this step, the fill-reducing permutation is computed and applied to the columns of A
	* and the column elimination tree is computed as well. Only the sparsity pattern of \a mat is exploited.
	*
	* \note In this step it is assumed that there is no empty row in the matrix \a mat.
	*/
	template <typename MatrixType, typename BlockQRSolverLeft, typename BlockQRSolverRight>
	void BlockAngularSparseQR_Ext<MatrixType, BlockQRSolverLeft, BlockQRSolverRight>::analyzePattern(const MatrixType& mat)
	{
		eigen_assert(mat.isCompressed() && "SparseQR requires a sparse matrix in compressed mode. Call .makeCompressed() before passing it to SparseQR");

		StorageIndex n = mat.cols();
		m_outputPerm_c.resize(n);
		m_outputPerm_c.indices().setLinSpaced(n, 0, StorageIndex(n - 1));

		StorageIndex m = mat.rows();
		m_rowPerm.resize(m);
		m_rowPerm.indices().setLinSpaced(m, 0, StorageIndex(m - 1));
	}

	/*********************************************************************************************************/
	template <typename RightBlockSolverType, typename StorageIndex, typename SrcType>
	void makeR(const int m1, const int m2, const int n1, const int n2, const SrcType &R2, const SrcType &J2, const SparseMatrix<Scalar, ColMajor, StorageIndex> &R1, const RightBlockSolverType &rightSolver, SparseMatrix<Scalar, ColMajor, StorageIndex> &Rout) {
		{
			Rout.resize(n1 + n2, m1 + m2);
			Rout.reserve(R1.nonZeros() + n1 * m2 + ((m2 - 1) * m2) / 2);
			// Columns of R1
			for (int c = 0; c < m1; c++) {
				Rout.startVec(c);
				for (typename SparseMatrix<Scalar, ColMajor, StorageIndex>::InnerIterator colIt(R1, c); colIt; ++colIt) {
					Rout.insertBack(colIt.index(), c) = colIt.value();
				}
			}
			// Columns of J2top combined with R2 + the desired column permutation on J2top
			for (int c = 0; c < m2; c++) {
				Rout.startVec(m1 + c);
				for (int r = 0; r < m1; r++) {
					Rout.insertBack(r, m1 + c) = J2(r, rightSolver.colsPermutation().indices()(c));
				}
				for (int r = m1; r <= m1 + c; r++) {
					Rout.insertBack(r, m1 + c) = R2(r - m1, c);
				}
			}
			Rout.finalize();
		}
	}
	template <typename RightBlockSolverType, typename StorageIndex>
	void makeR(const int m1, const int m2, const int n1, const int n2, const SparseMatrix<Scalar, ColMajor, StorageIndex> &R2, const SparseMatrix<Scalar, ColMajor, StorageIndex> &J2, const SparseMatrix<Scalar, ColMajor, StorageIndex> &R1, const RightBlockSolverType &rightSolver, SparseMatrix<Scalar, ColMajor, StorageIndex> &Rout) {
		{
			Eigen::MatrixXd J2top = J2.topRows(m1);

			Rout.resize(n1 + n2, m1 + m2);
			Rout.reserve(R1.nonZeros() + n1 * m2 + R2.nonZeros());
			// Columns of R1
			for (int c = 0; c < m1; c++) {
				Rout.startVec(c);
				for (typename SparseMatrix<Scalar, ColMajor, StorageIndex>::InnerIterator colIt(R1, c); colIt; ++colIt) {
					Rout.insertBack(colIt.index(), c) = colIt.value();
				}
			}
			// Columns of J2top combined with R2 + the desired column permutation on J2top
			for (int c = 0; c < m2; c++) {
				Rout.startVec(m1 + c);
				/*for (MatrixType::InnerIterator colIt(J2top, rightSolver.colsPermutation().indices()(c)); colIt; ++colIt) {
					Rout.insertBack(colIt.index(), m1 + c) = colIt.value();
				}*/
				for (int r = 0; r < m1; r++) {
					Rout.insertBack(r, m1 + c) = J2top(r, rightSolver.colsPermutation().indices()(c));
				}
				for (typename SparseMatrix<Scalar, ColMajor, StorageIndex>::InnerIterator colIt(R2, c); colIt; ++colIt) {
					Rout.insertBack(m1 + colIt.index(), m1 + c) = colIt.value();
				}
			}
			Rout.finalize();
		}
	}

	/*********************************************************************************************************/
	template <typename RightBlockSolver, typename LeftBlockSolver, typename StorageIndex, typename MatType>
	void solveRightBlock(const int m1, const int m2, const int n1, const int n2, MatType &J2, const SparseMatrix<Scalar, RowMajor, StorageIndex> &mat, RightBlockSolver &rightSolver, LeftBlockSolver &leftSolver) {
		MatType J2toprows = mat.block(0, m1, n1, m2).toDense();
		J2.topRows(n1).noalias() = leftSolver.matrixQ().transpose() * J2toprows;
		J2.bottomRows(n2) = mat.block(n1, m1, n2, m2);

		rightSolver.compute(J2.bottomRows(n1 + n2 - m1));

	//	MatType J2botPerm = rightSolver.rowsPermutation() * J2.bottomRows(n1 + n2 - m1);
	//	J2botPerm = J2botPerm * rightSolver.colsPermutation();
	//	SparseMatrix<Scalar> J2botPermSp = J2botPerm.sparseView();
	//	SparseMatrix<Scalar> R2sp = rightSolver.matrixR().sparseView();
	//	std::cout << "Q * R - J = " << (rightSolver.matrixQ() * R2sp - J2botPermSp).norm() << std::endl;
	//	std::cout << "Q.T * J - R = " << (rightSolver.matrixQ().transpose() * J2botPermSp - R2sp).norm() << std::endl;
	}

	template <typename RightBlockSolver, typename LeftBlockSolver, typename StorageIndex>
	void solveRightBlock(const int m1, const int m2, const int n1, const int n2, SparseMatrix<Scalar, ColMajor, StorageIndex> &J2, const SparseMatrix<Scalar, ColMajor, StorageIndex> &mat, RightBlockSolver &rightSolver, LeftBlockSolver &leftSolver) {
		J2 = mat.rightCols(m2);
		SparseMatrix<Scalar, RowMajor, StorageIndex> rmJ2(J2);
		rmJ2.topRows(n1) = leftSolver.matrixQ().transpose() * rmJ2.topRows(n1);
		J2 = SparseMatrix<Scalar, ColMajor, StorageIndex>(rmJ2);
		
		SparseMatrix<Scalar> J2bot = J2.bottomRows(n1 + n2 - m1);
		rightSolver.compute(J2bot);

		// Verify correctness of the qr decomposition
		//SparseMatrix<Scalar> J2botPerm = rightSolver.rowsPermutation() * J2bot;
		//J2botPerm = J2botPerm * rightSolver.colsPermutation();
		//std::cout << "Q * R - J = " << (rightSolver.matrixQ() * rightSolver.matrixR() - J2botPerm).norm() << std::endl;
		//std::cout << "Q.T * J - R = " << (rightSolver.matrixQ().transpose() * J2botPerm - rightSolver.matrixR()).norm() << std::endl;
	}
	/*********************************************************************************************************/

	/** \brief Performs the numerical QR factorization of the input matrix
	*
	* The function SparseQR::analyzePattern(const MatrixType&) must have been called beforehand with
	* a matrix having the same sparsity pattern than \a mat.
	*
	* \param mat The sparse column-major matrix
	*/

//#define OUTPUT_MAT 1
//#define OUTPUT_TIMING 1
	template <typename MatrixType, typename BlockQRSolverLeft, typename BlockQRSolverRight>
	void BlockAngularSparseQR_Ext<MatrixType, BlockQRSolverLeft, BlockQRSolverRight>::factorize(const MatrixType& mat)
	{
		clock_t begin = clock();

		typedef Matrix<Scalar, Dynamic, Dynamic> DenseMatrix;
		typedef typename MatrixType::Index Index;
		Index m1 = m_blockCols;
		Index m2 = mat.cols() - m_blockCols;
		Index n1 = m_blockRows;
		Index n2 = mat.rows() - m_blockRows;

		/// mat = | J1 J2 |
		/// J1 has m1 cols 

#ifdef OUTPUT_MAT
		Logger::instance()->logMatrixCSV(mat.leftCols(m1).toDense(), "J1.csv");
		Logger::instance()->logMatrixCSV(mat.rightCols(m2).toDense(), "J2.csv");
#endif

		// 1) Solve already block diagonal left block to get Q1 and R1
		begin = clock();
		m_leftSolver.compute(mat.block(0, 0, n1, m1));
#ifdef OUTPUT_TIMING
		std::cout << "m_leftSolver.compute(J1) ... " << double(clock() - begin) / CLOCKS_PER_SEC << "s" << std::endl;
#endif
		eigen_assert(m_leftSolver.info() == Success);		

		typename BlockQRSolverLeft::MatrixRType R1 = m_leftSolver.matrixR();

#ifdef OUTPUT_MAT
		Logger::instance()->logMatrixCSV(m_leftSolver.matrixQ().toDense(), "Q1.csv");
		Logger::instance()->logMatrixCSV(R1.toDense(), "R1.csv");
#endif

		RightBlockMatrixType J2(n1 + n2, m2);
		J2.setZero();
		begin = clock();
		solveRightBlock<BlockQRSolverRight, BlockQRSolverLeft, StorageIndex>(m1, m2, n1, n2, J2, mat, m_rightSolver, m_leftSolver);
#ifdef OUTPUT_TIMING
		std::cout << "solveRightBlock ... " << double(clock() - begin) / CLOCKS_PER_SEC << "s" << std::endl;
#endif
		typename BlockQRSolverRight::PermutationType cp2 = m_rightSolver.colsPermutation();
		typename BlockQRSolverRight::PermutationType rp2 = m_rightSolver.rowsPermutation();
		const typename BlockQRSolverRight::MatrixRType& R2 = m_rightSolver.matrixR();

	//	Eigen::MatrixXd Rut = m_rightSolver.matrixR().block(0, 0, m2, m2);
	//	Logger::instance()->logMatrixCSV(Rut, "R2.csv");

#ifdef OUTPUT_MAT
//		Logger::instance()->logMatrixCSV(R2.toDense(), "R2.csv");
//		Logger::instance()->logMatrixCSV(cp2.indices(), "cp2.csv");
//		Logger::instance()->logMatrixCSV(cp2.inverse().eval().indices(), "cp2Inv.csv");
//		Logger::instance()->logMatrixCSV(rp2.indices(), "rp2.csv");
//		Logger::instance()->logMatrixCSV(rp2.inverse().eval().indices(), "rp2Inv.csv");
#endif
/*
		MatrixType I(J2.rows(), J2.rows());
		I.setIdentity();
		MatrixType Q2(J2.rows(), J2.rows());
		Q2 = m_rightSolver.matrixQ() * I;
*/
#ifdef OUTPUT_MAT
//		Logger::instance()->logMatrixCSV(J2bot.toDense(), "J2bot.csv");
//		Logger::instance()->logMatrixCSV(J2.toDense(), "J2.csv");
//		Logger::instance()->logMatrixCSV(Q2.toDense(), "Q2.csv");
//		Logger::instance()->logMatrixCSV(R22.toDense(), "R2.csv");
#endif

		/// Compute R
		/// R Matrix
		/// R = | head(R1,m1) Atop*P2  |      m1 rows
		///     | 0           R2       |
		makeR<BlockQRSolverRight, StorageIndex>(m1, m2, n1, n2, R2, J2, R1, m_rightSolver, m_R);

#ifdef OUTPUT_MAT
		Logger::instance()->logMatrixCSV(m_R.toDense(), "R_final.csv");
#endif

		// fill cols permutation
		for (Index j = 0; j < m1; j++)
			m_outputPerm_c.indices()(j, 0) = m_leftSolver.colsPermutation().indices()(j, 0);
		for (Index j = m1; j < mat.cols(); j++)
			m_outputPerm_c.indices()(j, 0) = Index(m1 + m_rightSolver.colsPermutation().indices()(j - m1, 0));

		// fill rows permutation
		// Top block will use row permutation from the left solver
		if (m_leftSolver.hasRowPermutation()) {
			for (Index j = 0; j < n1; j++) {
				m_rowPerm.indices()(j, 0) = m_leftSolver.rowsPermutation().indices()(j, 0);
			}
			for (Index j = n1; j < n1 + n2; j++) {
				m_rowPerm.indices()(j, 0) = m_rightSolver.rowsPermutation().indices()(j - n1, 0);
			}
			// Add bottom block row permutation
		}

		m_nonzeropivots = m_leftSolver.rank() + m_rightSolver.rank();
		m_isInitialized = true;
		m_info = Success;

	}

	template <typename SparseQRType, typename Derived>
	struct BlockAngularSparseQR_Ext_QProduct : ReturnByValue<BlockAngularSparseQR_Ext_QProduct<SparseQRType, Derived> >
	{
		typedef typename SparseQRType::MatrixType MatrixType;
		typedef typename SparseQRType::Scalar Scalar;

		// Get the references 
		BlockAngularSparseQR_Ext_QProduct(const SparseQRType& qr, const Derived& other, bool transpose) :
			m_qr(qr), m_other(other), m_transpose(transpose) {}

		inline Index rows() const { return m_transpose ? m_qr.rows() : m_qr.cols(); }

		inline Index cols() const { return m_other.cols(); }

		// Assign to a vector
		template<typename DesType>
		void evalTo(DesType& res) const
		{
			Index n = m_qr.rows();
			Index m1 = m_qr.m_blockCols;
			Index n1 = m_qr.m_blockRows;

			eigen_assert(n == m_other.rows() && "Non conforming object sizes");

			if (m_transpose)
			{
				/// Q' Matrix
				/// Q = | I 0   | * Q1'    | m1xm1    0              | * n x n 
				///     | 0 Q2' |          |     0   (n-m1)x(n-m1)   |           

				/// Q v = | I 0   | * Q1' * v   = | I 0   | * [ Q1tv1 ]  = [ Q1tv1       ]
				///       | 0 Q2' |               | 0 Q2' |   [ Q1tv2 ]    [ Q2' * Q1tv2 ]    

				res = m_other;
				// jasvob FixMe: The multipliation has to be split on 3 lines like this in order for the Eigen type inference to work well. 
				MatrixType otherTopRows = m_other.topRows(n1);
				MatrixType resTopRows = m_qr.m_leftSolver.matrixQ().transpose() * otherTopRows;
				res.topRows(n1) = resTopRows;
				MatrixType Q2v2;
				MatrixType resBottomRows = m_qr.m_rightSolver.rowsPermutation() * res.bottomRows(n - m1);	// Don't forget to rowpermute
				Q2v2 = m_qr.m_rightSolver.matrixQ().transpose() * resBottomRows;
				res.bottomRows(n - m1) = Q2v2;
			}
			else
			{
				/// Q Matrix 
				/// Q = Q1 * | I 0  |     n x n * | m1xm1    0            |
				///          | 0 Q2 |             |     0   (n-m1)x(n-m1) |

				/// Q v = Q1 * | I 0  | * | v1 | =  Q1 * | v1      | 
				///            | 0 Q2 |   | v2 |         | Q2 * v2 | 

				res = m_other;
				MatrixType Q2v2;
				MatrixType resBottomRows = res.bottomRows(n - m1);
				Q2v2 = m_qr.m_rightSolver.matrixQ() * resBottomRows;
				res.bottomRows(n - m1) = m_qr.m_rightSolver.rowsPermutation().inverse() * Q2v2;	// Don't forget to back-rowpermute
				res = m_qr.m_leftSolver.matrixQ() * res;
			}
		}

		const SparseQRType& m_qr;
		const Derived& m_other;
		bool m_transpose;
	};

	template <typename SparseQRType>
	struct BlockAngularSparseQR_Ext_QProduct<SparseQRType, VectorX> : ReturnByValue<BlockAngularSparseQR_Ext_QProduct<SparseQRType, VectorX> >
	{
		typedef typename SparseQRType::MatrixType MatrixType;
		typedef typename SparseQRType::Scalar Scalar;

		// Get the references 
		BlockAngularSparseQR_Ext_QProduct(const SparseQRType& qr, const VectorX& other, bool transpose) :
			m_qr(qr), m_other(other), m_transpose(transpose) {}

		inline Index rows() const { return m_transpose ? m_qr.rows() : m_qr.cols(); }

		inline Index cols() const { return m_other.cols(); }

		// Assign to a vector
		template<typename DesType>
		void evalTo(DesType& res) const
		{
			Index n = m_qr.rows();
			Index m1 = m_qr.m_blockCols;
			Index n1 = m_qr.m_blockRows;

			eigen_assert(n == m_other.rows() && "Non conforming object sizes");

			if (m_transpose)
			{
				/// Q' Matrix
				/// Q = | I 0   | * Q1'    | m1xm1    0              | * n x n 
				///     | 0 Q2' |          |     0   (n-m1)x(n-m1)   |           

				/// Q v = | I 0   | * Q1' * v   = | I 0   | * [ Q1tv1 ]  = [ Q1tv1       ]
				///       | 0 Q2' |               | 0 Q2' |   [ Q1tv2 ]    [ Q2' * Q1tv2 ]    

				res = m_other;
				// jasvob FixMe: The multipliation has to be split on 3 lines like this in order for the Eigen type inference to work well. 
				VectorX otherTopRows = m_other.topRows(n1);
				VectorX resTopRows = m_qr.m_leftSolver.matrixQ().transpose() * otherTopRows;
				res.topRows(n1) = resTopRows;
				VectorX Q2v2;
				VectorX resBottomRows = m_qr.m_rightSolver.rowsPermutation() * res.bottomRows(n - m1);	// Don't forget to rowpermute
				Q2v2 = m_qr.m_rightSolver.matrixQ().transpose() * resBottomRows;
				res.bottomRows(n - m1) = Q2v2;
			}
			else
			{
				/// Q Matrix 
				/// Q = Q1 * | I 0  |     n x n * | m1xm1    0            |
				///          | 0 Q2 |             |     0   (n-m1)x(n-m1) |

				/// Q v = Q1 * | I 0  | * | v1 | =  Q1 * | v1      | 
				///            | 0 Q2 |   | v2 |         | Q2 * v2 | 

				res = m_other;
				VectorX Q2v2;
				VectorX resBottomRows = res.bottomRows(n - m1);
				Q2v2 = m_qr.m_rightSolver.matrixQ() * resBottomRows;
				res.bottomRows(n - m1) = m_qr.m_rightSolver.rowsPermutation().inverse() * Q2v2;	// Don't forget to back-rowpermute
				res = m_qr.m_leftSolver.matrixQ() * res;
			}
		}

		const SparseQRType& m_qr;
		const VectorX& m_other;
		bool m_transpose;
	};

	template<typename SparseQRType>
	struct BlockAngularSparseQR_ExtMatrixQReturnType : public EigenBase<BlockAngularSparseQR_ExtMatrixQReturnType<SparseQRType> >
	{
		typedef typename SparseQRType::Scalar Scalar;
		typedef Matrix<Scalar, Dynamic, Dynamic> DenseMatrix;
		enum {
			RowsAtCompileTime = Dynamic,
			ColsAtCompileTime = Dynamic
		};
		explicit BlockAngularSparseQR_ExtMatrixQReturnType(const SparseQRType& qr) : m_qr(qr) {}
		template<typename Derived>
		BlockAngularSparseQR_Ext_QProduct<SparseQRType, Derived> operator*(const MatrixBase<Derived>& other)
		{
			return BlockAngularSparseQR_Ext_QProduct<SparseQRType, Derived>(m_qr, other.derived(), false);
		}
		template<typename _Scalar, int _Options, typename _Index>
		BlockAngularSparseQR_Ext_QProduct<SparseQRType, SparseMatrix<_Scalar, _Options, _Index>> operator*(const SparseMatrix<_Scalar, _Options, _Index>& other)
		{
			return BlockAngularSparseQR_Ext_QProduct<SparseQRType, SparseMatrix<_Scalar, _Options, _Index>>(m_qr, other, false);
		}
		BlockAngularSparseQR_ExtMatrixQTransposeReturnType<SparseQRType> adjoint() const
		{
			return BlockAngularSparseQR_ExtMatrixQTransposeReturnType<SparseQRType>(m_qr);
		}
		inline Index rows() const { return m_qr.rows(); }
		inline Index cols() const { return m_qr.rows(); }
		// To use for operations with the transpose of Q
		BlockAngularSparseQR_ExtMatrixQTransposeReturnType<SparseQRType> transpose() const
		{
			return BlockAngularSparseQR_ExtMatrixQTransposeReturnType<SparseQRType>(m_qr);
		}

		const SparseQRType& m_qr;
	};

	template<typename SparseQRType>
	struct BlockAngularSparseQR_ExtMatrixQTransposeReturnType
	{
		explicit BlockAngularSparseQR_ExtMatrixQTransposeReturnType(const SparseQRType& qr) : m_qr(qr) {}
		template<typename Derived>
		BlockAngularSparseQR_Ext_QProduct<SparseQRType, Derived> operator*(const MatrixBase<Derived>& other)
		{
			return BlockAngularSparseQR_Ext_QProduct<SparseQRType, Derived>(m_qr, other.derived(), true);
		}
		template<typename _Scalar, int _Options, typename _Index>
		BlockAngularSparseQR_Ext_QProduct<SparseQRType, SparseMatrix<_Scalar, _Options, _Index>> operator*(const SparseMatrix<_Scalar, _Options, _Index>& other)
		{
			return BlockAngularSparseQR_Ext_QProduct<SparseQRType, SparseMatrix<_Scalar, _Options, _Index>>(m_qr, other, true);
		}
		const SparseQRType& m_qr;
	};

	namespace internal {

		template<typename SparseQRType>
		struct evaluator_traits<BlockAngularSparseQR_ExtMatrixQReturnType<SparseQRType> >
		{
			typedef typename SparseQRType::MatrixType MatrixType;
			typedef typename storage_kind_to_evaluator_kind<typename MatrixType::StorageKind>::Kind Kind;
			typedef SparseShape Shape;
		};

		template< typename DstXprType, typename SparseQRType>
		struct Assignment<DstXprType, BlockAngularSparseQR_ExtMatrixQReturnType<SparseQRType>, internal::assign_op<typename DstXprType::Scalar, typename BlockAngularSparseQR_ExtMatrixQReturnType<SparseQRType>::Scalar>, Sparse2Sparse>
		{
			typedef BlockAngularSparseQR_ExtMatrixQReturnType<SparseQRType> SrcXprType;
			typedef typename DstXprType::Scalar Scalar;
			typedef typename DstXprType::StorageIndex StorageIndex;
			static void run(DstXprType &dst, const SrcXprType &src, const internal::assign_op<Scalar, typename SrcXprType::Scalar> &/*func*/)
			{
				typename DstXprType::PlainObject idMat(src.m_qr.rows(), src.m_qr.rows());
				idMat.setIdentity();
				// Sort the sparse householder reflectors if needed
				const_cast<SparseQRType *>(&src.m_qr)->_sort_matrix_Q();
				dst = BlockAngularSparseQR_Ext_QProduct<SparseQRType, DstXprType>(src.m_qr, idMat, false);
			}
		};

		template< typename DstXprType, typename SparseQRType>
		struct Assignment<DstXprType, BlockAngularSparseQR_ExtMatrixQReturnType<SparseQRType>, internal::assign_op<typename DstXprType::Scalar, typename BlockAngularSparseQR_ExtMatrixQReturnType<SparseQRType>::Scalar>, Sparse2Dense>
		{
			typedef BlockAngularSparseQR_ExtMatrixQReturnType<SparseQRType> SrcXprType;
			typedef typename DstXprType::Scalar Scalar;
			typedef typename DstXprType::StorageIndex StorageIndex;
			static void run(DstXprType &dst, const SrcXprType &src, const internal::assign_op<Scalar, typename SrcXprType::Scalar> &/*func*/)
			{
				dst = src.m_qr.matrixQ() * DstXprType::Identity(src.m_qr.rows(), src.m_qr.rows());
			}
		};

	} // end namespace internal



} // end namespace Eigen

#endif

