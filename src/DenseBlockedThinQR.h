// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2017 Jan Svoboda <jan.svoboda@usi.ch>
// Copyright (C) 2016 Andrew Fitzgibbon <awf@microsoft.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_DENSE_BLOCKED_THIN_QR_H
#define EIGEN_DENSE_BLOCKED_THIN_QR_H

#include <ctime>
#include <typeinfo>
#include <shared_mutex>
#include "unsupported/Eigen/src/SparseQRExtra/SparseBlockCOO.h"
#include "unsupported/Eigen/src/SparseQRExtra/eigen_extras.h"

namespace Eigen {

	template<typename MatrixType, typename OrderingType, int SuggestedBlockCols, bool MultiThreading> class DenseBlockedThinQR;
	template<typename DenseBlockedThinQRType> struct DenseBlockedThinQRMatrixQReturnType;
	template<typename DenseBlockedThinQRType> struct DenseBlockedThinQRMatrixQTransposeReturnType;
	template<typename DenseBlockedThinQRType, typename Derived> struct DenseBlockedThinQR_QProduct;
	namespace internal {

		// traits<DenseBlockedThinQRMatrixQ[Transpose]>
		template <typename DenseBlockedThinQRType> struct traits<DenseBlockedThinQRMatrixQReturnType<DenseBlockedThinQRType> >
		{
			typedef typename DenseBlockedThinQRType::MatrixType ReturnType;
			typedef typename ReturnType::StorageIndex StorageIndex;
			typedef typename ReturnType::StorageKind StorageKind;
			enum {
				RowsAtCompileTime = Dynamic,
				ColsAtCompileTime = Dynamic
			};
		};

		template <typename DenseBlockedThinQRType> struct traits<DenseBlockedThinQRMatrixQTransposeReturnType<DenseBlockedThinQRType> >
		{
			typedef typename DenseBlockedThinQRType::MatrixType ReturnType;
		};

		template <typename DenseBlockedThinQRType, typename Derived> struct traits<DenseBlockedThinQR_QProduct<DenseBlockedThinQRType, Derived> >
		{
			typedef typename Derived::PlainObject ReturnType;
		};

		// DenseBlockedThinQR_traits
		template <typename T> struct DenseBlockedThinQR_traits {  };
		template <class T, int Rows, int Cols, int Options> struct DenseBlockedThinQR_traits<Matrix<T, Rows, Cols, Options>> {
			typedef Matrix<T, Rows, 1, Options> Vector;
		};
		template <class Scalar, int Options, typename Index> struct DenseBlockedThinQR_traits<SparseMatrix<Scalar, Options, Index>> {
			typedef SparseVector<Scalar, Options> Vector;
		};
	} // End namespace internal

	  /**
	  * \ingroup DenseBlockedThinQR_Module
	  * \class DenseBlockedThinQR
	  * \brief Sparse Householder QR Factorization for banded matrices
	  * This implementation is not rank revealing and uses Eigen::HouseholderQR for solving the dense blocks.
	  *
	  * Q is the orthogonal matrix represented as products of Householder reflectors.
	  * Use matrixQ() to get an expression and matrixQ().transpose() to get the transpose.
	  * You can then apply it to a vector.
	  *
	  * R is the sparse triangular or trapezoidal matrix. The later occurs when A is rank-deficient.
	  * matrixR().topLeftCorner(rank(), rank()) always returns a triangular factor of full rank.
	  *
	  * \tparam _MatrixType The type of the sparse matrix A, must be a column-major SparseMatrix<>
	  * \tparam _OrderingType The fill-reducing ordering method. See the \link OrderingMethods_Module
	  *  OrderingMethods \endlink module for the list of built-in and external ordering methods.
	  *
	  * \implsparsesolverconcept
	  *
	  * \warning The input sparse matrix A must be in compressed mode (see SparseMatrix::makeCompressed()).
	  *
	  */
	template<typename _MatrixType, typename _OrderingType, int _SuggestedBlockCols = 2, bool _MultiThreading = false>
	class DenseBlockedThinQR : public SparseSolverBase<DenseBlockedThinQR<_MatrixType, _OrderingType, _SuggestedBlockCols, _MultiThreading> >
	{
	protected:
		typedef SparseSolverBase<DenseBlockedThinQR<_MatrixType, _OrderingType, _SuggestedBlockCols, _MultiThreading> > Base;
		using Base::m_isInitialized;
	public:
		using Base::_solve_impl;
		typedef _MatrixType MatrixType;
		typedef _OrderingType OrderingType;
		typedef typename MatrixType::Scalar Scalar;
		typedef typename MatrixType::RealScalar RealScalar;
		typedef typename MatrixType::StorageIndex StorageIndex;
		typedef Matrix<StorageIndex, Dynamic, 1> IndexVector;
		typedef Matrix<Scalar, Dynamic, 1> ScalarVector;

		typedef DenseBlockedThinQRMatrixQReturnType<DenseBlockedThinQR> MatrixQType;
		typedef Matrix<Scalar, Dynamic, Dynamic> MatrixRType;
		typedef PermutationMatrix<Dynamic, Dynamic, StorageIndex> PermutationType;
		typedef Matrix<Scalar, Dynamic, Dynamic> DenseMatrixType;

		/*
		* Stores information about a dense block in a block sparse matrix.
		* Holds the position of the block (row index, column index) and its size (number of rows, number of columns).
		*/
		template <typename IndexType>
		struct BlockInfo {
			IndexType idxDiag;
			IndexType numRows;
			IndexType numCols;

			BlockInfo()
				: idxDiag(0), numRows(0), numCols(0) {
			}

			BlockInfo(const IndexType &diagIdx, const IndexType &nr, const IndexType &nc)
				: idxDiag(diagIdx), numRows(nr), numCols(nc) {
			}
		};

		typedef BlockInfo<StorageIndex> MatrixBlockInfo;
		typedef std::map<StorageIndex, MatrixBlockInfo> BlockInfoMap;
		typedef std::vector<StorageIndex> BlockInfoMapOrder;

		enum {
			ColsAtCompileTime = MatrixType::ColsAtCompileTime,
			MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime
		};

	public:
		DenseBlockedThinQR() : m_analysisIsok(false), m_lastError(""), m_useDefaultThreshold(true), m_useMultiThreading(_MultiThreading), m_hasRowPermutation(false)
		{ }

		/** Construct a QR factorization of the matrix \a mat.
		*
		* \warning The matrix \a mat must be in compressed mode (see SparseMatrix::makeCompressed()).
		*
		* \sa compute()
		*/
		explicit DenseBlockedThinQR(const MatrixType& mat) : m_analysisIsok(false), m_lastError(""), m_useDefaultThreshold(true), m_useMultiThreading(_MultiThreading), m_hasRowPermutation(false)
		{
			compute(mat);
		}

		/** Computes the QR factorization of the sparse matrix \a mat.
		*
		* \warning The matrix \a mat must be in compressed mode (see SparseMatrix::makeCompressed()).
		*
		* If input pattern analysis has been successfully performed before, it won't be run again by default.
		* forcePatternAnalysis - if true, forces reruning pattern analysis of the input matrix
		* \sa analyzePattern(), factorize()
		*/
		void compute(const DenseMatrixType& mat)
		{
			// Reset variables in case this method is called multiple times
			m_isInitialized = false;
			m_factorizationIsok = false;
			m_blocksYT.clear();

			clock_t beginFull = clock();

			// Analyze input matrix pattern and perform row and column permutations
			// Stores input matrix to m_pmat
			analyzePattern(mat);
	//		std::cout << "Analp: " << double(clock() - beginFull) / CLOCKS_PER_SEC << "s\n";

			// Create dense version of the already permuted input matrix
			// It is much faster to do the permutations on the sparse version
			beginFull = clock();
			this->m_R = mat;// .toDense();

	//		std::cout << "todense: " << double(clock() - beginFull) / CLOCKS_PER_SEC << "s\n";
			// And start factorizing block-by-block
			Index solvedCols = 0;
			Index cntr = 0;
			// As long as there are some unsolved columns
			while (solvedCols < this->m_R.cols()) {
				clock_t begin = clock();

				// Get next block info
				this->updateBlockInfo(solvedCols, this->m_R, _SuggestedBlockCols);
				
				// Factorize current block
				factorize(this->m_R);
				solvedCols += this->denseBlockInfo.numCols;

		//		std::cout << "Fact_" << cntr++ << ": " << double(clock() - begin) / CLOCKS_PER_SEC << "s\n";
				//if (solvedCols > 1000)
				//	break;
			}
//			std::cout << "Fact_total: " << double(clock() - beginFull) / CLOCKS_PER_SEC << "s\n";

			// m_pmatDense is now upper triangular (it is in fact the R)
			//RowMajorMatrixType rmR(m_pmatDense.rows(), m_pmatDense.cols());
			//rmR.setZero();
			//rmR.topRows(rmR.cols()) = m_pmatDense.topRows(rmR.cols()).sparseView();
			//this->m_R = MatrixType(rmR).template triangularView<Upper>();

			this->m_nonzeropivots = this->m_R.cols();
			m_isInitialized = true;
		}
		void analyzePattern(const DenseMatrixType& mat, bool rowPerm = true, bool colPerm = true);
		void updateBlockInfo(const Index solvedCols, const DenseMatrixType& mat, const Index blockCols = -1);
		void factorize(DenseMatrixType& mat);

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

		/** \returns an expression of the matrix Q as products of sparse Householder reflectors.
		* The common usage of this function is to apply it to a dense matrix or vector
		* \code
		* VectorXd B1, B2;
		* // Initialize B1
		* B2 = matrixQ() * B1;
		* \endcode
		*
		* To get a plain SparseMatrix representation of Q:
		* \code
		* SparseMatrix<double> Q;
		* Q = DenseBlockedThinQR<SparseMatrix<double> >(A).matrixQ();
		* \endcode
		* Internally, this call simply performs a sparse product between the matrix Q
		* and a sparse identity matrix. However, due to the fact that the sparse
		* reflectors are stored unsorted, two transpositions are needed to sort
		* them before performing the product.
		*/
		DenseBlockedThinQRMatrixQReturnType<DenseBlockedThinQR> matrixQ() const
		{
			return DenseBlockedThinQRMatrixQReturnType<DenseBlockedThinQR>(*this);
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
			//eigen_assert(m_isInitialized && "Decomposition is not initialized.");
			return this->m_rowPerm;
		}

		/**
		* \returns a flag indicating whether the factorization introduced some row permutations
		* It is determined during the input pattern analysis step.
		*/
		bool hasRowPermutation() const {
			return this->m_hasRowPermutation;
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
			eigen_assert(this->rows() == B.rows() && "DenseBlockedThinQR::solve() : invalid number of rows in the right hand side matrix");

			Index rank = this->rank();

			// Compute Q^T * b;
			typename Dest::PlainObject y, b;
			y = this->matrixQ().transpose() * B;
			b = y;

			// Solve with the triangular matrix R
			y.resize((std::max<Index>)(cols(), y.rows()), y.cols());
			y.topRows(rank) = this->matrixR().topLeftCorner(rank, rank).template triangularView<Upper>().solve(b.topRows(rank));
			y.bottomRows(y.rows() - rank).setZero();

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
			m_useDefaultThreshold = false;
			m_threshold = threshold;
		}

		void setDenseBlockStartRow(const Index &ri) {
			this->m_denseStartRow = ri;
		}

		/** \returns the solution X of \f$ A X = B \f$ using the current decomposition of A.
		*
		* \sa compute()
		*/
		template<typename Rhs>
		inline const Solve<DenseBlockedThinQR, Rhs> solve(const MatrixBase<Rhs>& B) const
		{
			eigen_assert(m_isInitialized && "The factorization should be called first, use compute()");
			eigen_assert(this->rows() == B.rows() && "DenseBlockedThinQR::solve() : invalid number of rows in the right hand side matrix");
			return Solve<DenseBlockedThinQR, Rhs>(*this, B.derived());
		}
		template<typename Rhs>
		inline const Solve<DenseBlockedThinQR, Rhs> solve(const SparseMatrixBase<Rhs>& B) const
		{
			eigen_assert(m_isInitialized && "The factorization should be called first, use compute()");
			eigen_assert(this->rows() == B.rows() && "DenseBlockedThinQR::solve() : invalid number of rows in the right hand side matrix");
			return Solve<DenseBlockedThinQR, Rhs>(*this, B.derived());
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

	protected:
		typedef SparseMatrix<Scalar, ColMajor, StorageIndex> MatrixQStorageType;
		typedef SparseBlockCOO<BlockYTY<Scalar, StorageIndex>, StorageIndex> SparseBlockYTY;
		typedef SparseMatrix<Scalar, RowMajor, StorageIndex> RowMajorMatrixType;

		bool m_analysisIsok;
		bool m_factorizationIsok;
		mutable ComputationInfo m_info;
		std::string m_lastError;
			
		DenseMatrixType m_R;                // The triangular factor matrix
		SparseBlockYTY m_blocksYT;		// Sparse block matrix storage holding the dense YTY blocks of the blocked representation of Householder reflectors.

		PermutationType m_outputPerm_c; // The final column permutation (for compatibility here, set to identity)
		PermutationType m_rowPerm;

		RealScalar m_threshold;         // Threshold to determine null Householder reflections
		bool m_useDefaultThreshold;     // Use default threshold
		Index m_nonzeropivots;          // Number of non zero pivots found
		bool m_useMultiThreading;		// Use multithreaded implementation of Householder product evaluation
		bool m_hasRowPermutation;		// Row permutation performed during the factorization

										/*
										* Structures filled during sparse matrix pattern analysis.
										*/

		MatrixBlockInfo denseBlockInfo;

		template <typename, typename > friend struct DenseBlockedThinQR_QProduct;

		void updateMat(const Index &fromIdx, const Index &toIdx, DenseMatrixType &mat, const Index &blockK = -1);
	};

	/** \brief Preprocessing step of a QR factorization
	*
	* \warning The matrix \a mat must be in compressed mode (see SparseMatrix::makeCompressed()).
	*
	* In this step, row-reordering permutation of A is computed and matrix banded structure is analyzed.
	* This is neccessary preprocessing step before the matrix factorization is carried out.
	*
	* This step assumes there is some sort of banded structure in the matrix.
	*
	* \note In this step it is assumed that there is no empty row in the matrix \a mat.
	*/
	struct IndexValue {
		Index index;
		Scalar value;

		IndexValue() : index(0), value(0) {
		}
		IndexValue(const Index &idx, const Scalar &val) : index(idx), value(val) {
		}
	};
	template <typename MatrixType, typename OrderingType, int SuggestedBlockCols, bool MultiThreading>
	void DenseBlockedThinQR<MatrixType, OrderingType, SuggestedBlockCols, MultiThreading>::analyzePattern(const DenseMatrixType& mat, bool rowPerm, bool colPerm)
	{
		typedef ColumnCount<MatrixType::StorageIndex> MatrixColCount;
		typedef RowRange<MatrixType::StorageIndex> MatrixRowRange;
		typedef std::map<MatrixType::StorageIndex, MatrixType::StorageIndex> BlockBandSize;

		Index n = mat.cols();
		Index m = mat.rows();
		Index diagSize = (std::min)(m, n);

		// No column permutation here
		this->m_outputPerm_c.setIdentity(mat.cols());

		// And no row permutatio neither
		this->m_rowPerm.setIdentity(mat.rows());
		
		m_analysisIsok = true;
	}

	template <typename MatrixType, typename OrderingType, int SuggestedBlockCols, bool MultiThreading>
	void DenseBlockedThinQR<MatrixType, OrderingType, SuggestedBlockCols, MultiThreading>::updateBlockInfo(const Index solvedCols, const DenseMatrixType& mat, const Index blockCols) {
		Index newCols = (blockCols > 0) ? blockCols : SuggestedBlockCols;
		Index colIdx = solvedCols + newCols;
		Index numRows = mat.rows() - colIdx;
		if (colIdx >= mat.cols()) {
			colIdx = mat.cols() - 1;
			newCols = mat.cols() - solvedCols;
			numRows = mat.rows() - solvedCols;
		} 

		this->denseBlockInfo = MatrixBlockInfo(solvedCols, numRows, newCols);

//		std::cout << "Solving Dense block: " << this->denseBlockInfo.idxDiag << ", " << this->denseBlockInfo.numRows << ", " << this->denseBlockInfo.numCols
//			<< " of matrix size " << this->m_pmat.rows() - solvedCols << "x" << this->m_pmat.cols() - solvedCols << std::endl;
	}

	template <typename MatrixType, typename OrderingType, int SuggestedBlockCols, bool MultiThreading>
	void DenseBlockedThinQR<MatrixType, OrderingType, SuggestedBlockCols, MultiThreading>::updateMat(const Index &fromIdx, const Index &toIdx, DenseMatrixType &mat, const Index &blockK) {
		// Now update the unsolved rest of m_pmat
		SparseBlockYTY::Element blockYTY = this->m_blocksYT[this->m_blocksYT.size() - 1];
	
		Index blockRows = this->m_blocksYT[blockK].value.rows();
		///*
		const size_t nloop = toIdx - fromIdx;
		const size_t nthreads = std::thread::hardware_concurrency();
		{
			std::vector<std::thread> threads(nthreads);
			std::mutex critical;
			for (int t = 0; t < nthreads; t++)
			{
				threads[t] = std::thread(std::bind(
					[&](const int bi, const int ei, const int t)
				{
					// loop over all items
					for (int j = bi; j < ei; j++)
					{
						// inner loop
						{
							mat.middleRows(this->denseBlockInfo.idxDiag, this->denseBlockInfo.numRows).col(fromIdx + j).noalias()
								+= (this->m_blocksYT[blockK].value.Y() * (this->m_blocksYT[blockK].value.T().transpose() * (this->m_blocksYT[blockK].value.Y().transpose() * mat.middleRows(this->denseBlockInfo.idxDiag, this->denseBlockInfo.numRows).col(fromIdx + j))));
						}
					}

				}, t*nloop / nthreads, (t + 1) == nthreads ? nloop : (t + 1)*nloop / nthreads, t));
			}
			std::for_each(threads.begin(), threads.end(), [](std::thread& x) {x.join(); });
		}
		//*/
		/*
			for (int ci = fromIdx; ci < toIdx; ci++) {
				// We can afford noalias() in this case
				mat.middleRows(this->denseBlockInfo.idxDiag, this->denseBlockInfo.numRows).col(ci).noalias()
					+= (this->m_blocksYT[blockK].value.Y() * (this->m_blocksYT[blockK].value.T().transpose() * (this->m_blocksYT[blockK].value.Y().transpose() * mat.middleRows(this->denseBlockInfo.idxDiag, this->denseBlockInfo.numRows).col(ci))));
			}
		//*/
	}

	/** \brief Performs the numerical QR factorization of the input matrix
	*
	* The function DenseBlockedThinQR::analyzePattern(const MatrixType&) must have been called beforehand with
	* a matrix having the same sparsity pattern than \a mat.
	*
	* \param mat The sparse column-major matrix
	*/
	template <typename MatrixType, typename OrderingType, int SuggestedBlockCols, bool MultiThreading>
	void DenseBlockedThinQR<MatrixType, OrderingType, SuggestedBlockCols, MultiThreading>::factorize(DenseMatrixType& mat)
	{
		// Triplet array for the matrix R
		//		Eigen::TripletArray<Scalar, typename MatrixType::Index> Rvals(2 * mat.nonZeros());

		// Dense QR solver used for each dense block 
		Eigen::HouseholderQR<DenseMatrixType> houseqr;

		// Prepare the first block
		//DenseMatrixType Ji = mat.block(this->denseBlockInfo.idxDiag, this->denseBlockInfo.idxDiag, this->denseBlockInfo.numRows, this->denseBlockInfo.numCols);

		/*********** Process the block ***********/
		// 1) Factorize the block
		houseqr.compute(mat.block(this->denseBlockInfo.idxDiag, this->denseBlockInfo.idxDiag, this->denseBlockInfo.numRows, this->denseBlockInfo.numCols));

		// 2) Create matrices T and Y
		Index numRows = this->denseBlockInfo.numRows;
		Index numCols = this->denseBlockInfo.numCols;
		MatrixXd T = MatrixXd::Zero(numCols, numCols);
		MatrixXd Y = MatrixXd::Zero(numRows, numCols);
		VectorXd v = VectorXd::Zero(numRows);
		VectorXd z = VectorXd::Zero(numRows);
		v(0) = 1.0;
		v.segment(1, numRows - 1) = houseqr.householderQ().essentialVector(0).segment(0, numRows - 1);
		Y.col(0) = v;
		T(0, 0) = -houseqr.hCoeffs()(0);
		for (MatrixType::StorageIndex bc = 1; bc < numCols; bc++) {
			v.setZero();
			v(bc) = 1.0;
			v.segment(bc + 1, numRows - bc - 1) = houseqr.householderQ().essentialVector(bc).segment(0, numRows - bc - 1);

			z = -houseqr.hCoeffs()(bc) * (T * (Y.transpose() * v));

			Y.col(bc) = v;
			T.col(bc) = z;
			T(bc, bc) = -houseqr.hCoeffs()(bc);
		}
		// Save current Y and T. The block YTY contains a main diagonal and subdiagonal part separated by (numZeros) zero rows.
		m_blocksYT.insert(SparseBlockYTY::Element(this->denseBlockInfo.idxDiag, this->denseBlockInfo.idxDiag, BlockYTY<Scalar, StorageIndex>(Y, T, 0)));

		// Update the trailing columns of the matrix block
		this->updateMat(this->denseBlockInfo.idxDiag, mat.cols(), mat, this->m_blocksYT.size() - 1);
	}

	/*
	* General Householder product evaluation performing Q * A or Q.T * A.
	* The general version is assuming that A is sparse and that the output will be sparse as well.
	* Offers single-threaded and multi-threaded implementation.
	* The choice of implementation depends on a template parameter of the DenseBlockedThinQR class.
	* The single-threaded implementation cannot work in-place. It is implemented this way for performance related reasons.
	*/
	template <typename DenseBlockedThinQRType, typename Derived>
	struct DenseBlockedThinQR_QProduct : ReturnByValue<DenseBlockedThinQR_QProduct<DenseBlockedThinQRType, Derived> >
	{
		typedef typename SparseMatrix<Scalar, ColMajor, Index> MatrixType;
		typedef typename DenseBlockedThinQRType::MatrixType DenseMatrixType;
		typedef typename DenseBlockedThinQRType::Scalar Scalar;

		typedef typename internal::DenseBlockedThinQR_traits<MatrixType>::Vector SparseVector;

		// Get the references 
		DenseBlockedThinQR_QProduct(const DenseBlockedThinQRType& qr, const Derived& other, bool transpose) :
			m_qr(qr), m_other(other), m_transpose(transpose) {}
		inline Index rows() const { return m_transpose ? m_qr.rows() : m_qr.cols(); }
		inline Index cols() const { return m_other.cols(); }

		// Assign to a vector
		template<typename DesType>
		void evalTo(DesType& res) const
		{
			Index m = m_qr.rows();
			Index n = m_qr.cols();

			if (m_qr.m_useMultiThreading) {
				/********************************* MT *****************************/

				std::vector<std::vector<std::pair<typename MatrixType::Index, Scalar>>> resVals(m_other.cols());
				Index numNonZeros = 0;

				if (m_transpose)
				{
					// Compute res = Q' * other column by column using parallel for loop
					const size_t nloop = m_other.cols();
					const size_t nthreads = std::thread::hardware_concurrency();
					{
						std::vector<std::thread> threads(nthreads);
						std::mutex critical;
						for (int t = 0; t<nthreads; t++)
						{
							threads[t] = std::thread(std::bind(
								[&](const int bi, const int ei, const int t)
							{
								// loop over all items
								for (int j = bi; j<ei; j++)
								{
									// inner loop
									{
										VectorXd tmpResColJ;
										SparseVector resColJ;
										VectorXd resColJd;
										resColJd = m_other.col(j).toDense();
										for (Index k = 0; k < m_qr.m_blocksYT.size(); k++) {
											tmpResColJ = VectorXd(m_qr.m_blocksYT[k].value.rows()); 
											tmpResColJ.segment(0, m_qr.m_blocksYT[k].value.rows()) = resColJd.segment(m_qr.m_blocksYT[k].row, m_qr.m_blocksYT[k].value.rows());
											
											// We can afford noalias() in this case
											tmpResColJ.noalias() += m_qr.m_blocksYT[k].value.multTransposed(tmpResColJ);

											resColJd.segment(m_qr.m_blocksYT[k].row, m_qr.m_blocksYT[k].value.rows()) = tmpResColJ.segment(0, m_qr.m_blocksYT[k].value.rows());
										}

										std::lock_guard<std::mutex> lock(critical);
										// Write the result back to j-th column of res
										resColJ = resColJd.sparseView();
										numNonZeros += resColJ.nonZeros();
										resVals[j].reserve(resColJ.nonZeros());
										for (SparseVector::InnerIterator it(resColJ); it; ++it) {
											resVals[j].push_back(std::make_pair(it.row(), it.value()));
										}

									}
								}
							}, t*nloop / nthreads, (t + 1) == nthreads ? nloop : (t + 1)*nloop / nthreads, t));
						}
						std::for_each(threads.begin(), threads.end(), [](std::thread& x) {x.join(); });
					}
				}
				else {
					const size_t nloop = m_other.cols();
					const size_t nthreads = std::thread::hardware_concurrency();
					{
						std::vector<std::thread> threads(nthreads);
						std::mutex critical;
						for (int t = 0; t < nthreads; t++)
						{
							threads[t] = std::thread(std::bind(
								[&](const int bi, const int ei, const int t)
							{
								// loop over all items
								for (int j = bi; j < ei; j++)
								{
									// inner loop
									{
										VectorXd tmpResColJ;
										SparseVector resColJ;
										VectorXd resColJd;
										resColJd = m_other.col(j).toDense();
										for (Index k = m_qr.m_blocksYT.size() - 1; k >= 0; k--) {
											tmpResColJ = VectorXd(m_qr.m_blocksYT[k].value.rows());
											tmpResColJ.segment(0, m_qr.m_blocksYT[k].value.rows()) = resColJd.segment(m_qr.m_blocksYT[k].row, m_qr.m_blocksYT[k].value.rows());

											// We can afford noalias() in this case
											tmpResColJ.noalias() += m_qr.m_blocksYT[k].value * tmpResColJ;

											resColJd.segment(m_qr.m_blocksYT[k].row, m_qr.m_blocksYT[k].value.rows()) = tmpResColJ.segment(0, m_qr.m_blocksYT[k].value.rows());
										}

										std::lock_guard<std::mutex> lock(critical);
										// Write the result back to j-th column of res
										resColJ = resColJd.sparseView();
										numNonZeros += resColJ.nonZeros();
										resVals[j].reserve(resColJ.nonZeros());
										for (SparseVector::InnerIterator it(resColJ); it; ++it) {
											resVals[j].push_back(std::make_pair(it.row(), it.value()));
										}

									}
								}
							}, t*nloop / nthreads, (t + 1) == nthreads ? nloop : (t + 1)*nloop / nthreads, t));
						}
						std::for_each(threads.begin(), threads.end(), [](std::thread& x) {x.join(); });
					}

				}

				// Form the output
				res = Derived(m_other.rows(), m_other.cols());
				res.reserve(numNonZeros);
				for (int j = 0; j < resVals.size(); j++) {
					res.startVec(j);
					for (auto it = resVals[j].begin(); it != resVals[j].end(); ++it) {
						res.insertBack(it->first, j) = it->second;
					}
				}
				res.finalize();

			}
			else {
				/********************************* ST *****************************/
				res = Derived(m_other.rows(), m_other.cols());
				//res.reserve(m_other.rows() * m_other.cols() * 0.25);// FixMe: Better estimation of nonzeros?
				res.reserve(m_other.rows() * m_other.cols() * 0.9); // FixMe: Better estimation of nonzeros?

				if (m_transpose)
				{
					//Compute res = Q' * other column by column
					SparseVector resColJ;
					VectorXd resColJd;
					VectorXd tmpResColJ;
					for (Index j = 0; j < m_other.cols(); j++) {
						// Use temporary vector resColJ inside of the for loop - faster access
						resColJd = m_other.col(j).toDense();
						for (Index k = 0; k < m_qr.m_blocksYT.size(); k++) {
							tmpResColJ = VectorXd(m_qr.m_blocksYT[k].value.rows());
							tmpResColJ.segment(0, m_qr.m_blocksYT[k].value.rows()) = resColJd.segment(m_qr.m_blocksYT[k].row, m_qr.m_blocksYT[k].value.rows());

							// We can afford noalias() in this case
							tmpResColJ.noalias() += m_qr.m_blocksYT[k].value.multTransposed(tmpResColJ);

							resColJd.segment(m_qr.m_blocksYT[k].row, m_qr.m_blocksYT[k].value.rows()) = tmpResColJ.segment(0, m_qr.m_blocksYT[k].value.rows());
						}
						// Write the result back to j-th column of res
						resColJ = resColJd.sparseView();
						res.startVec(j);
						for (SparseVector::InnerIterator it(resColJ); it; ++it) {
							res.insertBack(it.row(), j) = it.value();
						}
					}
				}
				else
				{
					// Compute res = Q * other column by column
					SparseVector resColJ;
					VectorXd resColJd;
					VectorXd tmpResColJ;
					for (Index j = 0; j < m_other.cols(); j++) {
						resColJd = m_other.col(j).toDense();
						for (Index k = m_qr.m_blocksYT.size() - 1; k >= 0; k--) {
							tmpResColJ = VectorXd(m_qr.m_blocksYT[k].value.rows());
							tmpResColJ.segment(0, m_qr.m_blocksYT[k].value.rows()) = resColJd.segment(m_qr.m_blocksYT[k].row, m_qr.m_blocksYT[k].value.rows());

							// We can afford noalias() in this case
							tmpResColJ.noalias() += m_qr.m_blocksYT[k].value * tmpResColJ;

							resColJd.segment(m_qr.m_blocksYT[k].row, m_qr.m_blocksYT[k].value.rows()) = tmpResColJ.segment(0, m_qr.m_blocksYT[k].value.rows());
						}

						// Write the result back to j-th column of res
						resColJ = resColJd.sparseView();
						res.startVec(j);
						for (SparseVector::InnerIterator it(resColJ); it; ++it) {
							res.insertBack(it.row(), j) = it.value();
						}
					}
				}

				// Don't forget to call finalize
				res.finalize();
			}

		}

		const DenseBlockedThinQRType& m_qr;
		const Derived& m_other;
		bool m_transpose;
	};

	/*
	* Specialization of the Householder product evaluation performing Q * A or Q.T * A
	* for the case when A and the output are dense vectors.=
	* Offers only single-threaded implementation as the overhead of multithreading would not bring any speedup for a dense vector (A is single column).
	*/

	template <typename DenseBlockedThinQRType>
	struct DenseBlockedThinQR_QProduct<DenseBlockedThinQRType, VectorX> : ReturnByValue<DenseBlockedThinQR_QProduct<DenseBlockedThinQRType, VectorX> >
	{
		typedef typename DenseBlockedThinQRType::MatrixType MatrixType;
		typedef typename DenseBlockedThinQRType::Scalar Scalar;

		// Get the references 
		DenseBlockedThinQR_QProduct(const DenseBlockedThinQRType& qr, const VectorX& other, bool transpose) :
			m_qr(qr), m_other(other), m_transpose(transpose) {}
		inline Index rows() const { return m_transpose ? m_qr.rows() : m_qr.cols(); }
		inline Index cols() const { return m_other.cols(); }

		// Assign to a vector
		template<typename DesType>
		void evalTo(DesType& res) const
		{
			Index m = m_qr.rows();
			Index n = m_qr.cols();
			res = m_other;

			if (m_transpose)
			{
				//Compute res = Q' * other (other is vector - only one column => no iterations of j)
				VectorX partialRes;
				for (Index k = 0; k < m_qr.m_blocksYT.size(); k++) {
					partialRes = VectorXd(m_qr.m_blocksYT[k].value.rows());
					partialRes.segment(0, m_qr.m_blocksYT[k].value.rows()) = res.segment(m_qr.m_blocksYT[k].row, m_qr.m_blocksYT[k].value.rows());
					
					// We can afford noalias() in this case
					partialRes.noalias() += m_qr.m_blocksYT[k].value.multTransposed(partialRes);

					res.segment(m_qr.m_blocksYT[k].row, m_qr.m_blocksYT[k].value.rows()) = partialRes.segment(0, m_qr.m_blocksYT[k].value.rows());
				}
			}
			else
			{
				// Compute res = Q * other (other is vector - only one column => no iterations of j)
				VectorX partialRes;
				for (Index k = m_qr.m_blocksYT.size() - 1; k >= 0; k--) {
					partialRes = VectorXd(m_qr.m_blocksYT[k].value.rows());
					partialRes.segment(0, m_qr.m_blocksYT[k].value.rows()) = res.segment(m_qr.m_blocksYT[k].row, m_qr.m_blocksYT[k].value.rows());

					// We can afford noalias() in this case
					partialRes.noalias() += m_qr.m_blocksYT[k].value * partialRes;

					res.segment(m_qr.m_blocksYT[k].row, m_qr.m_blocksYT[k].value.rows()) = partialRes.segment(0, m_qr.m_blocksYT[k].value.rows());
				}
			}
		}

		const DenseBlockedThinQRType& m_qr;
		const VectorX& m_other;
		bool m_transpose;
	};

	template<typename DenseBlockedThinQRType>
	struct DenseBlockedThinQRMatrixQReturnType : public EigenBase<DenseBlockedThinQRMatrixQReturnType<DenseBlockedThinQRType> >
	{
		typedef typename DenseBlockedThinQRType::Scalar Scalar;
		typedef Matrix<Scalar, Dynamic, Dynamic> DenseMatrix;
		enum {
			RowsAtCompileTime = Dynamic,
			ColsAtCompileTime = Dynamic
		};
		explicit DenseBlockedThinQRMatrixQReturnType(const DenseBlockedThinQRType& qr) : m_qr(qr) {}
		/*DenseBlockedThinQR_QProduct<DenseBlockedThinQRType, Derived> operator*(const MatrixBase<Derived>& other)
		{
		return DenseBlockedThinQR_QProduct<DenseBlockedThinQRType,Derived>(m_qr,other.derived(),false);
		}*/
		template<typename Derived>
		DenseBlockedThinQR_QProduct<DenseBlockedThinQRType, Derived> operator*(const MatrixBase<Derived>& other)
		{
			return DenseBlockedThinQR_QProduct<DenseBlockedThinQRType, Derived>(m_qr, other.derived(), false);
		}
		template<typename _Scalar, int _Options, typename _Index>
		DenseBlockedThinQR_QProduct<DenseBlockedThinQRType, SparseMatrix<_Scalar, _Options, _Index>> operator*(const SparseMatrix<_Scalar, _Options, _Index>& other)
		{
			return DenseBlockedThinQR_QProduct<DenseBlockedThinQRType, SparseMatrix<_Scalar, _Options, _Index>>(m_qr, other, false);
		}
		DenseBlockedThinQRMatrixQTransposeReturnType<DenseBlockedThinQRType> adjoint() const
		{
			return DenseBlockedThinQRMatrixQTransposeReturnType<DenseBlockedThinQRType>(m_qr);
		}
		inline Index rows() const { return m_qr.rows(); }
		inline Index cols() const { return m_qr.rows(); }
		// To use for operations with the transpose of Q
		DenseBlockedThinQRMatrixQTransposeReturnType<DenseBlockedThinQRType> transpose() const
		{
			return DenseBlockedThinQRMatrixQTransposeReturnType<DenseBlockedThinQRType>(m_qr);
		}

		const DenseBlockedThinQRType& m_qr;
	};

	template<typename DenseBlockedThinQRType>
	struct DenseBlockedThinQRMatrixQTransposeReturnType
	{
		explicit DenseBlockedThinQRMatrixQTransposeReturnType(const DenseBlockedThinQRType& qr) : m_qr(qr) {}
		template<typename Derived>
		DenseBlockedThinQR_QProduct<DenseBlockedThinQRType, Derived> operator*(const MatrixBase<Derived>& other)
		{
			return DenseBlockedThinQR_QProduct<DenseBlockedThinQRType, Derived>(m_qr, other.derived(), true);
		}
		template<typename _Scalar, int _Options, typename _Index>
		DenseBlockedThinQR_QProduct<DenseBlockedThinQRType, SparseMatrix<_Scalar, _Options, _Index>> operator*(const SparseMatrix<_Scalar, _Options, _Index>& other)
		{
			return DenseBlockedThinQR_QProduct<DenseBlockedThinQRType, SparseMatrix<_Scalar, _Options, _Index>>(m_qr, other, true);
		}
		const DenseBlockedThinQRType& m_qr;
	};

	namespace internal {

		template<typename DenseBlockedThinQRType>
		struct evaluator_traits<DenseBlockedThinQRMatrixQReturnType<DenseBlockedThinQRType> >
		{
			typedef typename DenseBlockedThinQRType::MatrixType MatrixType;
			typedef typename storage_kind_to_evaluator_kind<typename MatrixType::StorageKind>::Kind Kind;
			typedef SparseShape Shape;
		};

		template< typename DstXprType, typename DenseBlockedThinQRType>
		struct Assignment<DstXprType, DenseBlockedThinQRMatrixQReturnType<DenseBlockedThinQRType>, internal::assign_op<typename DstXprType::Scalar, typename DstXprType::Scalar>, Sparse2Sparse>
		{
			typedef DenseBlockedThinQRMatrixQReturnType<DenseBlockedThinQRType> SrcXprType;
			typedef typename DstXprType::Scalar Scalar;
			typedef typename DstXprType::StorageIndex StorageIndex;
			static void run(DstXprType &dst, const SrcXprType &src, const internal::assign_op<Scalar, Scalar> &/*func*/)
			{
				typename DstXprType::PlainObject idMat(src.m_qr.rows(), src.m_qr.rows());
				idMat.setIdentity();
				// Sort the sparse householder reflectors if needed
				//const_cast<DenseBlockedThinQRType *>(&src.m_qr)->_sort_matrix_Q();
				dst = DenseBlockedThinQR_QProduct<DenseBlockedThinQRType, DstXprType>(src.m_qr, idMat, false);
			}
		};

		template< typename DstXprType, typename DenseBlockedThinQRType>
		struct Assignment<DstXprType, DenseBlockedThinQRMatrixQReturnType<DenseBlockedThinQRType>, internal::assign_op<typename DstXprType::Scalar, typename DstXprType::Scalar>, Sparse2Dense>
		{
			typedef DenseBlockedThinQRMatrixQReturnType<DenseBlockedThinQRType> SrcXprType;
			typedef typename DstXprType::Scalar Scalar;
			typedef typename DstXprType::StorageIndex StorageIndex;
			static void run(DstXprType &dst, const SrcXprType &src, const internal::assign_op<Scalar, Scalar> &/*func*/)
			{
				dst = src.m_qr.matrixQ() * DstXprType::Identity(src.m_qr.rows(), src.m_qr.rows());
			}
		};

	} // end namespace internal

} // end namespace Eigen

#endif
