// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2017 Jan Svoboda <jan.svoboda@usi.ch>
// Copyright (C) 2016 Andrew Fitzgibbon <awf@microsoft.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_DENSE_BLOCKED_THIN_SPARSE_QR_H
#define EIGEN_DENSE_BLOCKED_THIN_SPARSE_QR_H

#include <ctime>
#include <typeinfo>
#include <shared_mutex>
//#include "SparseBlockCOO_Ext2.h"
#include "unsupported/Eigen/src/SparseQRExtra/SparseBlockCOO.h"
#include "unsupported/Eigen/src/SparseQRExtra/eigen_extras.h"

namespace Eigen {

	template<typename MatrixType, typename OrderingType, int SuggestedBlockCols, bool MultiThreading> class DenseBlockedThinSparseQR;
	template<typename DenseBlockedThinSparseQRType> struct DenseBlockedThinSparseQRMatrixQReturnType;
	template<typename DenseBlockedThinSparseQRType> struct DenseBlockedThinSparseQRMatrixQTransposeReturnType;
	template<typename DenseBlockedThinSparseQRType, typename Derived> struct DenseBlockedThinSparseQR_QProduct;
	namespace internal {

		// traits<DenseBlockedThinSparseQRMatrixQ[Transpose]>
		template <typename DenseBlockedThinSparseQRType> struct traits<DenseBlockedThinSparseQRMatrixQReturnType<DenseBlockedThinSparseQRType> >
		{
			typedef typename DenseBlockedThinSparseQRType::MatrixType ReturnType;
			typedef typename ReturnType::StorageIndex StorageIndex;
			typedef typename ReturnType::StorageKind StorageKind;
			enum {
				RowsAtCompileTime = Dynamic,
				ColsAtCompileTime = Dynamic
			};
		};

		template <typename DenseBlockedThinSparseQRType> struct traits<DenseBlockedThinSparseQRMatrixQTransposeReturnType<DenseBlockedThinSparseQRType> >
		{
			typedef typename DenseBlockedThinSparseQRType::MatrixType ReturnType;
		};

		template <typename DenseBlockedThinSparseQRType, typename Derived> struct traits<DenseBlockedThinSparseQR_QProduct<DenseBlockedThinSparseQRType, Derived> >
		{
			typedef typename Derived::PlainObject ReturnType;
		};

		// DenseBlockedThinSparseQR_traits
		template <typename T> struct DenseBlockedThinSparseQR_traits {  };
		template <class T, int Rows, int Cols, int Options> struct DenseBlockedThinSparseQR_traits<Matrix<T, Rows, Cols, Options>> {
			typedef Matrix<T, Rows, 1, Options> Vector;
		};
		template <class Scalar, int Options, typename Index> struct DenseBlockedThinSparseQR_traits<SparseMatrix<Scalar, Options, Index>> {
			typedef SparseVector<Scalar, Options> Vector;
		};
	} // End namespace internal

	  /**
	  * \ingroup DenseBlockedThinSparseQR_Module
	  * \class DenseBlockedThinSparseQR
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
	class DenseBlockedThinSparseQR : public SparseSolverBase<DenseBlockedThinSparseQR<_MatrixType, _OrderingType, _SuggestedBlockCols, _MultiThreading> >
	{
	protected:
		typedef SparseSolverBase<DenseBlockedThinSparseQR<_MatrixType, _OrderingType, _SuggestedBlockCols, _MultiThreading> > Base;
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

		typedef DenseBlockedThinSparseQRMatrixQReturnType<DenseBlockedThinSparseQR> MatrixQType;
		typedef SparseMatrix<Scalar, ColMajor, StorageIndex> MatrixRType;
		typedef PermutationMatrix<Dynamic, Dynamic, StorageIndex> PermutationType;
		typedef Matrix<Scalar, Dynamic, Dynamic> DenseMatrixType;

		/*
		* Stores information about a dense block in a block sparse matrix.
		* Holds the position of the block (row index, column index) and its size (number of rows, number of columns).
		*/
		template <typename IndexType>
		struct BlockInfo {
			IndexType idxRow;
			IndexType idxCol;
			IndexType numRows;
			IndexType numCols;

			BlockInfo()
				: idxRow(0), idxCol(0), numRows(0), numCols(0) {
			}

			BlockInfo(const IndexType &rowIdx, const IndexType &colIdx, const IndexType &nr, const IndexType &nc)
				: idxRow(rowIdx), idxCol(colIdx), numRows(nr), numCols(nc) {
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
		DenseBlockedThinSparseQR() : m_analysisIsok(false), m_lastError(""), m_useDefaultThreshold(true), m_useMultiThreading(_MultiThreading), m_hasRowPermutation(false)
		{ }

		/** Construct a QR factorization of the matrix \a mat.
		*
		* \warning The matrix \a mat must be in compressed mode (see SparseMatrix::makeCompressed()).
		*
		* \sa compute()
		*/
		explicit DenseBlockedThinSparseQR(const MatrixType& mat) : m_analysisIsok(false), m_lastError(""), m_useDefaultThreshold(true), m_useMultiThreading(_MultiThreading), m_hasRowPermutation(false)
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
		void compute(const MatrixType& mat)
		{
			// Reset variables in case this method is called multiple times
			m_isInitialized = false;
			m_factorizationIsok = false;
			m_blocksYT.clear();
			this->m_nnzColPermIdxs.clear();
			this->m_zeroColPermIdxs.clear();

			clock_t beginFull = clock();

			// Analyze input matrix pattern and perform row and column permutations
			// Stores input matrix to m_pmat
			analyzePattern(mat);
		//	std::cout << "Analp: " << double(clock() - beginFull) / CLOCKS_PER_SEC << "s\n";

			// Create dense version of the already permuted input matrix
			// It is much faster to do the permutations on the sparse version
			beginFull = clock();
			this->m_pmatDense = this->m_pmat.toDense();

			// Initialize householder permutation matrix
			this->m_houseColPerm.setIdentity(this->m_pmatDense.cols());

			// Reset nonzero pivots count
			this->m_nonzeroPivots = 0;

			// Prepare m_R to be filled in
			m_R.resize(this->m_pmatDense.rows(), this->m_pmatDense.cols());
			m_R.setZero();
			// Reserve number of elements needed in case m_R is full upper triangular
			m_R.reserve(this->m_pmatDense.cols() * this->m_pmatDense.cols() / 2.0);

		//	std::cout << "todense: " << double(clock() - beginFull) / CLOCKS_PER_SEC << "s\n";
			// And start factorizing block-by-block
			Index solvedCols = 0;
			Index cntr = 0;
			// As long as there are some unsolved columns
			Index newPivots = 0;
			while (solvedCols < this->m_pmatDense.cols()) {
				clock_t begin = clock();

				// Get next block info
				this->updateBlockInfo(solvedCols, this->m_pmat, newPivots, _SuggestedBlockCols);

				// Factorize current block
				newPivots = this->m_nonzeroPivots;
				factorize(this->m_pmatDense);
				newPivots = this->m_nonzeroPivots - newPivots;
				solvedCols += this->denseBlockInfo.numCols;

		//		std::cout << "Fact_" << cntr++ << ": " << double(clock() - begin) / CLOCKS_PER_SEC << "s\n";
				//if (solvedCols > 1000)
				//	break;
			}
			//			std::cout << "Fact_total: " << double(clock() - beginFull) / CLOCKS_PER_SEC << "s\n";

			// Set computed Householder column permutation
			for (int c = 0; c < this->m_nnzColPermIdxs.size(); c++) {
				this->m_houseColPerm.indices()(c) = this->m_nnzColPermIdxs[c];
			}
			for (int c = 0; c < this->m_zeroColPermIdxs.size(); c++) {
				this->m_houseColPerm.indices()(this->m_nnzColPermIdxs.size() + c) = this->m_zeroColPermIdxs[c];
			}

			// Combine the two column permutation matrices together
			this->m_outputPerm_c = this->m_outputPerm_c * this->m_houseColPerm;

			// Don't forget to finalize m_R
			m_R.finalize();
		//	Logger::instance()->logMatrixCSV(m_R.toDense(), "2_R2.csv");
		//	Logger::instance()->logMatrixCSV(m_pmatDense * this->m_houseColPerm, "2_fact.csv");

			// m_pmatDense is now upper triangular (it is in fact the R)
			/*RowMajorMatrixType rmR(m_pmatDense.rows(), m_pmatDense.cols());
			rmR.setZero();
			rmR.topRows(rmR.cols()) = m_pmatDense.topRows(rmR.cols()).sparseView();
			this->m_R = MatrixType(rmR).template triangularView<Upper>();
			*/
			//this->m_nonzeropivots = this->m_R.cols();
            std::cout << "NNZ pivots: " << this->m_nonzeroPivots << std::endl;
            
			m_isInitialized = true;
		}
		void analyzePattern(const MatrixType& mat, bool rowPerm = true, bool colPerm = true);
		void updateBlockInfo(const Index solvedCols, const MatrixType& mat, const Index newPivots, const Index blockCols = -1);
		void factorize(const DenseMatrixType& mat);

		/** \returns the number of rows of the represented matrix.
		*/
		inline Index rows() const { return m_pmat.rows(); }

		/** \returns the number of columns of the represented matrix.
		*/
		inline Index cols() const { return m_pmat.cols(); }

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
			return m_nonzeroPivots;
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
		* Q = DenseBlockedThinSparseQR<SparseMatrix<double> >(A).matrixQ();
		* \endcode
		* Internally, this call simply performs a sparse product between the matrix Q
		* and a sparse identity matrix. However, due to the fact that the sparse
		* reflectors are stored unsorted, two transpositions are needed to sort
		* them before performing the product.
		*/
		DenseBlockedThinSparseQRMatrixQReturnType<DenseBlockedThinSparseQR> matrixQ() const
		{
			return DenseBlockedThinSparseQRMatrixQReturnType<DenseBlockedThinSparseQR>(*this);
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
			eigen_assert(this->rows() == B.rows() && "DenseBlockedThinSparseQR::solve() : invalid number of rows in the right hand side matrix");

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
		inline const Solve<DenseBlockedThinSparseQR, Rhs> solve(const MatrixBase<Rhs>& B) const
		{
			eigen_assert(m_isInitialized && "The factorization should be called first, use compute()");
			eigen_assert(this->rows() == B.rows() && "DenseBlockedThinSparseQR::solve() : invalid number of rows in the right hand side matrix");
			return Solve<DenseBlockedThinSparseQR, Rhs>(*this, B.derived());
		}
		template<typename Rhs>
		inline const Solve<DenseBlockedThinSparseQR, Rhs> solve(const SparseMatrixBase<Rhs>& B) const
		{
			eigen_assert(m_isInitialized && "The factorization should be called first, use compute()");
			eigen_assert(this->rows() == B.rows() && "DenseBlockedThinSparseQR::solve() : invalid number of rows in the right hand side matrix");
			return Solve<DenseBlockedThinSparseQR, Rhs>(*this, B.derived());
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

		MatrixType  m_pmat;				// Sparse version of input matrix - used for ordering and search purposes
		DenseMatrixType  m_pmatDense;	// Dense version of the input matrix - used for factorization (much faster than using sparse)

		MatrixRType m_R;                // The triangular factor matrix
		SparseBlockYTY m_blocksYT;		// Sparse block matrix storage holding the dense YTY blocks of the blocked representation of Householder reflectors.

		PermutationType m_outputPerm_c; // The final column permutation (for compatibility here, set to identity)
		PermutationType m_rowPerm;
		PermutationType m_houseColPerm;
		std::vector<Index> m_nnzColPermIdxs;
		std::vector<Index> m_zeroColPermIdxs;

		RealScalar m_threshold;         // Threshold to determine null Householder reflections
		bool m_useDefaultThreshold;     // Use default threshold
		Index m_nonzeroPivots;          // Number of non zero pivots found
		bool m_useMultiThreading;		// Use multithreaded implementation of Householder product evaluation
		bool m_hasRowPermutation;		// Row permutation performed during the factorization

										/*
										* Structures filled during sparse matrix pattern analysis.
										*/
		MatrixBlockInfo denseBlockInfo;

		template <typename, typename > friend struct DenseBlockedThinSparseQR_QProduct;

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
	template <typename IndexType>
	struct ColCount {
		IndexType origIdx;
		IndexType nnzs;

		ColCount() : origIdx(0), nnzs(0) {
		}

		ColCount(const IndexType &origIdx, const IndexType &nnzs)
			: origIdx(origIdx), nnzs(nnzs){
		}

		bool operator<(const ColCount& rhs) const { 
			return this->nnzs < rhs.nnzs;
		}
	};

	template <typename MatrixType, typename OrderingType, int SuggestedBlockCols, bool MultiThreading>
	void DenseBlockedThinSparseQR<MatrixType, OrderingType, SuggestedBlockCols, MultiThreading>::analyzePattern(const MatrixType& mat, bool rowPerm, bool colPerm)
	{
		typedef ColCount<typename MatrixType::StorageIndex> MatrixColCount;
		typedef RowRange<typename MatrixType::StorageIndex> MatrixRowRange;
		typedef std::map<typename MatrixType::StorageIndex, typename MatrixType::StorageIndex> BlockBandSize;

		Index n = mat.cols();
		Index m = mat.rows();
		Index diagSize = (std::min)(m, n);

		/******************************************************************/
		// Create column permutation (according to the number of nonzeros in columns
		if (colPerm) {
			std::vector<MatrixColCount> colNnzs;
			for (Index c = 0; c < mat.cols(); c++) {
				colNnzs.push_back(MatrixColCount(c, mat.col(c).nonZeros()));
			}
			std::stable_sort(colNnzs.begin(), colNnzs.end());
			Eigen::Matrix<typename MatrixType::StorageIndex, Dynamic, 1> colpermIndices(colNnzs.size());
			for (Index c = 0; c < colNnzs.size(); c++) {
				colpermIndices(colNnzs[c].origIdx) = c;
			}
			this->m_outputPerm_c = PermutationType(colpermIndices);
			/*
			this->m_outputPerm_c.resize(mat.cols());
			COLAMDOrdering<typename MatrixType::StorageIndex> ord;
			ord(mat, this->m_outputPerm_c);
			*/

			m_pmat = mat * this->m_outputPerm_c;
		}
		else {
			this->m_outputPerm_c.setIdentity(mat.cols());

			// Don't waste time calling matrix multiplication if the permutation is identity
			m_pmat = mat;
		}

		/******************************************************************/
		// 1) Compute and store band information for each row in the matrix
		if (rowPerm) {
			BlockBandSize bandWidths, bandHeights;
			RowMajorMatrixType rmMat(m_pmat);
			std::vector<MatrixRowRange> rowRanges;
			for (typename MatrixType::StorageIndex j = 0; j < rmMat.rows(); j++) {
				typename RowMajorMatrixType::InnerIterator rowIt(rmMat, j);
				typename MatrixType::StorageIndex startIdx = rowIt.index();
				typename MatrixType::StorageIndex endIdx = startIdx;
				while (++rowIt) { endIdx = rowIt.index(); }	// FixMe: Is there a better way?
				rowRanges.push_back(MatrixRowRange(j, startIdx, endIdx));
			}

			// 2) Sort the rows to form as-banded-as-possible matrix
			// Set an indicator whether row sorting is needed
			this->m_hasRowPermutation = !std::is_sorted(rowRanges.begin(), rowRanges.end(), [](const MatrixRowRange &lhs, const MatrixRowRange &rhs) {
				return (lhs.start < rhs.start);
			});
			// Perform the actual row sorting if needed
			if (this->m_hasRowPermutation) {
				std::stable_sort(rowRanges.begin(), rowRanges.end(), [](const MatrixRowRange &lhs, const MatrixRowRange &rhs) {
					return (lhs.start < rhs.start);
					/*if (lhs.start < rhs.start) {
					return true;
					}
					else if (lhs.start == rhs.start) {
					//else if (!(rhs.start < lhs.start)) {
					if (lhs.end < rhs.end) {
					return true;
					}
					else {
					return lhs.origIdx < rhs.origIdx;
					}
					}
					else {
					return false;
					}*/
				});
			}

			Eigen::Matrix<typename MatrixType::StorageIndex, Dynamic, 1> permIndices(rowRanges.size());
			typename MatrixType::StorageIndex rowIdx = 0;
			for (auto it = rowRanges.begin(); it != rowRanges.end(); ++it, rowIdx++) {
				permIndices(it->origIdx) = rowIdx;
			}
			// Create row permutation matrix that achieves the desired row reordering
			this->m_rowPerm = PermutationType(permIndices);

			m_pmat = this->m_rowPerm * m_pmat;
		}
		else {
			this->m_rowPerm.setIdentity(m_pmat.rows());

			// Don't waste time calling matrix multiplication if the permutation is identity
		}
		/******************************************************************/

		m_analysisIsok = true;
	}

	template <typename MatrixType, typename OrderingType, int SuggestedBlockCols, bool MultiThreading>
	void DenseBlockedThinSparseQR<MatrixType, OrderingType, SuggestedBlockCols, MultiThreading>::updateBlockInfo(const Index solvedCols, const MatrixType& mat, const Index newPivots, const Index blockCols) {
		Index newCols = (blockCols > 0) ? blockCols : SuggestedBlockCols;
		Index colIdx = solvedCols + newCols;
		Index numRows = 0;
		if (colIdx >= m_pmat.cols()) {
			colIdx = m_pmat.cols() - 1;
			newCols = m_pmat.cols() - solvedCols;
			numRows = m_pmat.rows() - this->m_nonzeroPivots;
		}
		else {
			typename MatrixType::StorageIndex biggestEndIdx = 0;
			for (int c = 0; c < newCols; c++) {
				//MatrixType::InnerIterator colIt(m_pmat, solvedCols + newCols - 1);
				typename MatrixType::InnerIterator colIt(m_pmat, solvedCols + c);
				typename MatrixType::StorageIndex endIdx = 0;// startIdx;
				if (colIt) {
					endIdx = colIt.index();
				}
				while (++colIt) { endIdx = colIt.index(); }	// FixMe: Is there a better way?

				if (endIdx > biggestEndIdx) {
					biggestEndIdx = endIdx;
				}
			}

			numRows = biggestEndIdx - this->m_nonzeroPivots + 1;
			if (numRows < (this->denseBlockInfo.numRows - newPivots)) {
				// In the next step we need to process at least all the rows we did in the last one
				// Even if the next block would be "shorter"
				numRows = this->denseBlockInfo.numRows - newPivots;
			}
		}

		//if (this->m_nonzeroPivots + numRows > m_pmat.rows()) {
		//	numRows = m_pmat.rows() - this->m_nonzeroPivots;
		//}

		this->denseBlockInfo = MatrixBlockInfo(this->m_nonzeroPivots, solvedCols, numRows, newCols);

		//		std::cout << "Solving Dense block: " << this->denseBlockInfo.idxDiag << ", " << this->denseBlockInfo.numRows << ", " << this->denseBlockInfo.numCols
		//			<< " of matrix size " << this->m_pmat.rows() - solvedCols << "x" << this->m_pmat.cols() - solvedCols << std::endl;
	}

	template <typename MatrixType, typename OrderingType, int SuggestedBlockCols, bool MultiThreading>
	void DenseBlockedThinSparseQR<MatrixType, OrderingType, SuggestedBlockCols, MultiThreading>::updateMat(const Index &fromIdx, const Index &toIdx, DenseMatrixType &mat, const Index &blockK) {
		// Now update the unsolved rest of m_pmat
		//typename SparseBlockYTY::Element blockYTY = this->m_blocksYT[this->m_blocksYT.size() - 1];

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
							mat.middleRows(this->denseBlockInfo.idxRow, blockRows).col(fromIdx + j).noalias()
								+= (this->m_blocksYT[blockK].value.Y() * (this->m_blocksYT[blockK].value.T().transpose() * (this->m_blocksYT[blockK].value.Y().transpose() * mat.middleRows(this->denseBlockInfo.idxRow, blockRows).col(fromIdx + j))));
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
	* The function DenseBlockedThinSparseQR::analyzePattern(const MatrixType&) must have been called beforehand with
	* a matrix having the same sparsity pattern than \a mat.
	*
	* \param mat The sparse column-major matrix
	*/
	template <typename MatrixType, typename OrderingType, int SuggestedBlockCols, bool MultiThreading>
	void DenseBlockedThinSparseQR<MatrixType, OrderingType, SuggestedBlockCols, MultiThreading>::factorize(const DenseMatrixType& mat)
	{
		// Triplet array for the matrix R
		//		Eigen::TripletArray<Scalar, typename MatrixType::Index> Rvals(2 * mat.nonZeros());

		// Dense QR solver used for each dense block 
		//Eigen::HouseholderQR<DenseMatrixType> houseqr;
		Eigen::ColPivHouseholderQR<DenseMatrixType> houseqr;

		// Prepare the first block
		DenseMatrixType Ji = m_pmatDense.block(this->denseBlockInfo.idxRow, this->denseBlockInfo.idxCol, this->denseBlockInfo.numRows, this->denseBlockInfo.numCols);

		/*********** Process the block ***********/
		// 1) Factorize the block
		houseqr.compute(Ji);

	//	std::cout << "Rank: " << houseqr.rank() << std::endl;
	//	std::cout << "----- Ji --------\n" << Ji.topRows(16) << "\n---------------------" << std::endl;

	//	std::cout << "----- MatrixQR --------\n" << houseqr.matrixQR().topRows(16) << "\n---------------------" << std::endl;

		// Update column permutation according to ColPivHouseholderQR
		for (Index c = 0; c < houseqr.nonzeroPivots(); c++) {
			this->m_nnzColPermIdxs.push_back(this->denseBlockInfo.idxCol + houseqr.colsPermutation().indices()(c));
		}
		for (Index c = houseqr.nonzeroPivots(); c < this->denseBlockInfo.numCols; c++) {
			this->m_zeroColPermIdxs.push_back(this->denseBlockInfo.idxCol + houseqr.colsPermutation().indices()(c));
		}
	
		// 2) Create matrices T and Y
		Index numRows = this->denseBlockInfo.numRows;
		//Index numCols = this->denseBlockInfo.numCols;
		Index numCols = houseqr.nonzeroPivots(); // !!! Only for the nonzero pivots !!!
		MatrixXd T = MatrixXd::Zero(numCols, numCols);
		MatrixXd Y = MatrixXd::Zero(numRows, numCols);
		VectorXd v = VectorXd::Zero(numRows);
		VectorXd z = VectorXd::Zero(numRows);
		v(0) = 1.0;
		v.segment(1, numRows - 1) = houseqr.householderQ().essentialVector(0).segment(0, numRows - 1);
		Y.col(0) = v;
		T(0, 0) = -houseqr.hCoeffs()(0);
		for (typename MatrixType::StorageIndex bc = 1; bc < numCols; bc++) {
			v.setZero();
			v(bc) = 1.0;
			v.segment(bc + 1, numRows - bc - 1) = houseqr.householderQ().essentialVector(bc).segment(0, numRows - bc - 1);

			z = -houseqr.hCoeffs()(bc) * (T * (Y.transpose() * v));

			Y.col(bc) = v;
			T.col(bc) = z;
			T(bc, bc) = -houseqr.hCoeffs()(bc);
		}
		// Colpermute Y and T
		/*PermutationType colp;
		colp.setIdentity(houseqr.nonzeroPivots());
		for (int i = 0; i < houseqr.nonzeroPivots(); i++) {
			colp.indices()(houseqr.colsPermutation().indices()(i)) = i;
		}
		Y = Y * colp;
		T = T * colp;*/
		// Save current Y and T. The block YTY contains a main diagonal and subdiagonal part separated by (numZeros) zero rows.
		//	m_blocksYT.insert(typename SparseBlockYTY::Element(this->denseBlockInfo.idxDiag, this->denseBlockInfo.idxDiag, BlockYTY<Scalar, StorageIndex>(Y, T, 0)));
		m_blocksYT.insert(typename SparseBlockYTY::Element(this->denseBlockInfo.idxRow, this->denseBlockInfo.idxCol, BlockYTY<Scalar, StorageIndex>(Y, T, 0)));

		// Update the trailing columns of the matrix block
		this->updateMat(this->denseBlockInfo.idxCol, m_pmatDense.cols(), m_pmatDense, this->m_blocksYT.size() - 1);

		// Add solved columns to R
        // m_nonzeroPivots is telling us where is the current diagonal position
        
        // Don't forget to add the upper overlap (anything above the current diagonal element is already processed, but is part of R
        for (typename MatrixType::StorageIndex bc = 0; bc < numCols; bc++) {
			m_R.startVec(this->m_nonzeroPivots + bc);
            for(typename MatrixType::StorageIndex br = 0; br < this->m_nonzeroPivots; br++) {
                m_R.insertBack(br, this->m_nonzeroPivots + bc) = this->m_pmatDense(br, this->denseBlockInfo.idxCol + houseqr.colsPermutation().indices()(bc));
            }
			for (typename MatrixType::StorageIndex br = 0; br <= bc; br++) {
				m_R.insertBack(this->m_nonzeroPivots + br, this->m_nonzeroPivots + bc) = houseqr.matrixQR()(br, bc);
			}
		}

		// Add nonzero pivots from this block
		this->m_nonzeroPivots += houseqr.nonzeroPivots();
	}

	/*
	* General Householder product evaluation performing Q * A or Q.T * A.
	* The general version is assuming that A is sparse and that the output will be sparse as well.
	* Offers single-threaded and multi-threaded implementation.
	* The choice of implementation depends on a template parameter of the DenseBlockedThinSparseQR class.
	* The single-threaded implementation cannot work in-place. It is implemented this way for performance related reasons.
	*/
	template <typename DenseBlockedThinSparseQRType, typename Derived>
	struct DenseBlockedThinSparseQR_QProduct : ReturnByValue<DenseBlockedThinSparseQR_QProduct<DenseBlockedThinSparseQRType, Derived> >
	{
		typedef typename DenseBlockedThinSparseQRType::MatrixType MatrixType;
		typedef typename DenseBlockedThinSparseQRType::Scalar Scalar;

		typedef typename internal::DenseBlockedThinSparseQR_traits<MatrixType>::Vector SparseVector;

		// Get the references 
		DenseBlockedThinSparseQR_QProduct(const DenseBlockedThinSparseQRType& qr, const Derived& other, bool transpose) :
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
										for (typename SparseVector::InnerIterator it(resColJ); it; ++it) {
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
										for (typename SparseVector::InnerIterator it(resColJ); it; ++it) {
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
						for (typename SparseVector::InnerIterator it(resColJ); it; ++it) {
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
						for (typename SparseVector::InnerIterator it(resColJ); it; ++it) {
							res.insertBack(it.row(), j) = it.value();
						}
					}
				}

				// Don't forget to call finalize
				res.finalize();
			}

		}

		const DenseBlockedThinSparseQRType& m_qr;
		const Derived& m_other;
		bool m_transpose;
	};

	/*
	* Specialization of the Householder product evaluation performing Q * A or Q.T * A
	* for the case when A and the output are dense vectors.=
	* Offers only single-threaded implementation as the overhead of multithreading would not bring any speedup for a dense vector (A is single column).
	*/

	template <typename DenseBlockedThinSparseQRType>
	struct DenseBlockedThinSparseQR_QProduct<DenseBlockedThinSparseQRType, VectorX> : ReturnByValue<DenseBlockedThinSparseQR_QProduct<DenseBlockedThinSparseQRType, VectorX> >
	{
		typedef typename DenseBlockedThinSparseQRType::MatrixType MatrixType;
		typedef typename DenseBlockedThinSparseQRType::Scalar Scalar;

		// Get the references 
		DenseBlockedThinSparseQR_QProduct(const DenseBlockedThinSparseQRType& qr, const VectorX& other, bool transpose) :
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

		const DenseBlockedThinSparseQRType& m_qr;
		const VectorX& m_other;
		bool m_transpose;
	};

	template<typename DenseBlockedThinSparseQRType>
	struct DenseBlockedThinSparseQRMatrixQReturnType : public EigenBase<DenseBlockedThinSparseQRMatrixQReturnType<DenseBlockedThinSparseQRType> >
	{
		typedef typename DenseBlockedThinSparseQRType::Scalar Scalar;
		typedef Matrix<Scalar, Dynamic, Dynamic> DenseMatrix;
		enum {
			RowsAtCompileTime = Dynamic,
			ColsAtCompileTime = Dynamic
		};
		explicit DenseBlockedThinSparseQRMatrixQReturnType(const DenseBlockedThinSparseQRType& qr) : m_qr(qr) {}
		/*DenseBlockedThinSparseQR_QProduct<DenseBlockedThinSparseQRType, Derived> operator*(const MatrixBase<Derived>& other)
		{
		return DenseBlockedThinSparseQR_QProduct<DenseBlockedThinSparseQRType,Derived>(m_qr,other.derived(),false);
		}*/
		template<typename Derived>
		DenseBlockedThinSparseQR_QProduct<DenseBlockedThinSparseQRType, Derived> operator*(const MatrixBase<Derived>& other)
		{
			return DenseBlockedThinSparseQR_QProduct<DenseBlockedThinSparseQRType, Derived>(m_qr, other.derived(), false);
		}
		template<typename _Scalar, int _Options, typename _Index>
		DenseBlockedThinSparseQR_QProduct<DenseBlockedThinSparseQRType, SparseMatrix<_Scalar, _Options, _Index>> operator*(const SparseMatrix<_Scalar, _Options, _Index>& other)
		{
			return DenseBlockedThinSparseQR_QProduct<DenseBlockedThinSparseQRType, SparseMatrix<_Scalar, _Options, _Index>>(m_qr, other, false);
		}
		DenseBlockedThinSparseQRMatrixQTransposeReturnType<DenseBlockedThinSparseQRType> adjoint() const
		{
			return DenseBlockedThinSparseQRMatrixQTransposeReturnType<DenseBlockedThinSparseQRType>(m_qr);
		}
		inline Index rows() const { return m_qr.rows(); }
		inline Index cols() const { return m_qr.rows(); }
		// To use for operations with the transpose of Q
		DenseBlockedThinSparseQRMatrixQTransposeReturnType<DenseBlockedThinSparseQRType> transpose() const
		{
			return DenseBlockedThinSparseQRMatrixQTransposeReturnType<DenseBlockedThinSparseQRType>(m_qr);
		}

		const DenseBlockedThinSparseQRType& m_qr;
	};

	template<typename DenseBlockedThinSparseQRType>
	struct DenseBlockedThinSparseQRMatrixQTransposeReturnType
	{
		explicit DenseBlockedThinSparseQRMatrixQTransposeReturnType(const DenseBlockedThinSparseQRType& qr) : m_qr(qr) {}
		template<typename Derived>
		DenseBlockedThinSparseQR_QProduct<DenseBlockedThinSparseQRType, Derived> operator*(const MatrixBase<Derived>& other)
		{
			return DenseBlockedThinSparseQR_QProduct<DenseBlockedThinSparseQRType, Derived>(m_qr, other.derived(), true);
		}
		template<typename _Scalar, int _Options, typename _Index>
		DenseBlockedThinSparseQR_QProduct<DenseBlockedThinSparseQRType, SparseMatrix<_Scalar, _Options, _Index>> operator*(const SparseMatrix<_Scalar, _Options, _Index>& other)
		{
			return DenseBlockedThinSparseQR_QProduct<DenseBlockedThinSparseQRType, SparseMatrix<_Scalar, _Options, _Index>>(m_qr, other, true);
		}
		const DenseBlockedThinSparseQRType& m_qr;
	};

	namespace internal {

		template<typename DenseBlockedThinSparseQRType>
		struct evaluator_traits<DenseBlockedThinSparseQRMatrixQReturnType<DenseBlockedThinSparseQRType> >
		{
			typedef typename DenseBlockedThinSparseQRType::MatrixType MatrixType;
			typedef typename storage_kind_to_evaluator_kind<typename MatrixType::StorageKind>::Kind Kind;
			typedef SparseShape Shape;
		};

		template< typename DstXprType, typename DenseBlockedThinSparseQRType>
		struct Assignment<DstXprType, DenseBlockedThinSparseQRMatrixQReturnType<DenseBlockedThinSparseQRType>, internal::assign_op<typename DstXprType::Scalar, typename DstXprType::Scalar>, Sparse2Sparse>
		{
			typedef DenseBlockedThinSparseQRMatrixQReturnType<DenseBlockedThinSparseQRType> SrcXprType;
			typedef typename DstXprType::Scalar Scalar;
			typedef typename DstXprType::StorageIndex StorageIndex;
			static void run(DstXprType &dst, const SrcXprType &src, const internal::assign_op<Scalar, Scalar> &/*func*/)
			{
				typename DstXprType::PlainObject idMat(src.m_qr.rows(), src.m_qr.rows());
				idMat.setIdentity();
				// Sort the sparse householder reflectors if needed
				//const_cast<DenseBlockedThinSparseQRType *>(&src.m_qr)->_sort_matrix_Q();
				dst = DenseBlockedThinSparseQR_QProduct<DenseBlockedThinSparseQRType, DstXprType>(src.m_qr, idMat, false);
			}
		};

		template< typename DstXprType, typename DenseBlockedThinSparseQRType>
		struct Assignment<DstXprType, DenseBlockedThinSparseQRMatrixQReturnType<DenseBlockedThinSparseQRType>, internal::assign_op<typename DstXprType::Scalar, typename DstXprType::Scalar>, Sparse2Dense>
		{
			typedef DenseBlockedThinSparseQRMatrixQReturnType<DenseBlockedThinSparseQRType> SrcXprType;
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
