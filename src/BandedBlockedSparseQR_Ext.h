// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2017 Jan Svoboda <jan.svoboda@usi.ch>
// Copyright (C) 2016 Andrew Fitzgibbon <awf@microsoft.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_SPARSE_BANDED_BLOCKED_QR_EXT_H
#define EIGEN_SPARSE_BANDED_BLOCKED_QR_EXT_H

#include <ctime>
#include <typeinfo>
#include <shared_mutex>
#include "SparseBlockCOO_Ext.h"
#include "unsupported/Eigen/src/SparseQRExtra/eigen_extras.h"

namespace Eigen {

	template<typename MatrixType, typename OrderingType, int SuggestedBlockCols, bool MultiThreading> class BandedBlockedSparseQR_Ext;
	template<typename BandedBlockedSparseQR_ExtType> struct BandedBlockedSparseQR_ExtMatrixQReturnType;
	template<typename BandedBlockedSparseQR_ExtType> struct BandedBlockedSparseQR_ExtMatrixQTransposeReturnType;
	template<typename BandedBlockedSparseQR_ExtType, typename Derived> struct BandedBlockedSparseQR_Ext_QProduct;
	namespace internal {

		// traits<BandedBlockedSparseQR_ExtMatrixQ[Transpose]>
		template <typename BandedBlockedSparseQR_ExtType> struct traits<BandedBlockedSparseQR_ExtMatrixQReturnType<BandedBlockedSparseQR_ExtType> >
		{
			typedef typename BandedBlockedSparseQR_ExtType::MatrixType ReturnType;
			typedef typename ReturnType::StorageIndex StorageIndex;
			typedef typename ReturnType::StorageKind StorageKind;
			enum {
				RowsAtCompileTime = Dynamic,
				ColsAtCompileTime = Dynamic
			};
		};

		template <typename BandedBlockedSparseQR_ExtType> struct traits<BandedBlockedSparseQR_ExtMatrixQTransposeReturnType<BandedBlockedSparseQR_ExtType> >
		{
			typedef typename BandedBlockedSparseQR_ExtType::MatrixType ReturnType;
		};

		template <typename BandedBlockedSparseQR_ExtType, typename Derived> struct traits<BandedBlockedSparseQR_Ext_QProduct<BandedBlockedSparseQR_ExtType, Derived> >
		{
			typedef typename Derived::PlainObject ReturnType;
		};

		// BandedBlockedSparseQR_Ext_traits
		template <typename T> struct BandedBlockedSparseQR_Ext_traits {  };
		template <class T, int Rows, int Cols, int Options> struct BandedBlockedSparseQR_Ext_traits<Matrix<T, Rows, Cols, Options>> {
			typedef Matrix<T, Rows, 1, Options> Vector;
		};
		template <class Scalar, int Options, typename Index> struct BandedBlockedSparseQR_Ext_traits<SparseMatrix<Scalar, Options, Index>> {
			typedef SparseVector<Scalar, Options> Vector;
		};
	} // End namespace internal

	  /**
	  * \ingroup BandedBlockedSparseQR_Ext_Module
	  * \class BandedBlockedSparseQR_Ext
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
	class BandedBlockedSparseQR_Ext : public SparseSolverBase<BandedBlockedSparseQR_Ext<_MatrixType, _OrderingType, _SuggestedBlockCols, _MultiThreading> >
	{
	protected:
		typedef SparseSolverBase<BandedBlockedSparseQR_Ext<_MatrixType, _OrderingType, _SuggestedBlockCols, _MultiThreading> > Base;
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

		typedef BandedBlockedSparseQR_ExtMatrixQReturnType<BandedBlockedSparseQR_Ext> MatrixQType;
		typedef SparseMatrix<Scalar, ColMajor, StorageIndex> MatrixRType;
		typedef PermutationMatrix<Dynamic, Dynamic, StorageIndex> PermutationType;

		/*
		* Stores information about a dense block in a block sparse matrix.
		* Holds the position of the block (row index, column index) and its size (number of rows, number of columns).
		*/
		template <typename IndexType>
		struct BlockInfo {
			IndexType rowIdx;
			IndexType colIdx;
			IndexType numRows;
			IndexType numCols;

			BlockInfo()
				: rowIdx(0), colIdx(0), numRows(0), numCols(0) {
			}

			BlockInfo(const IndexType &ri, const IndexType &ci, const IndexType &nr, const IndexType &nc)
				: rowIdx(ri), colIdx(ci), numRows(nr), numCols(nc) {
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
		BandedBlockedSparseQR_Ext() : m_analysisIsok(false), m_lastError(""), m_useDefaultThreshold(true), m_useMultiThreading(_MultiThreading), m_hasRowPermutation(false)
		{ }

		/** Construct a QR factorization of the matrix \a mat.
		*
		* \warning The matrix \a mat must be in compressed mode (see SparseMatrix::makeCompressed()).
		*
		* \sa compute()
		*/
		explicit BandedBlockedSparseQR_Ext(const MatrixType& mat) : m_analysisIsok(false), m_lastError(""), m_useDefaultThreshold(true), m_useMultiThreading(_MultiThreading), m_hasRowPermutation(false)
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
		void compute(const MatrixType& mat, bool forcePatternAlaysis = false)
		{
			// If successful analysis was performed before
			if (!m_analysisIsok || forcePatternAlaysis) {
				analyzePattern(mat);
			}

			// !!! Reset variables before the factorization !!!
			m_isInitialized = false;
			m_factorizationIsok = false;
			m_blocksYT.clear();
			factorize(mat);
		}
		void setPattern(const StorageIndex matRows, const StorageIndex matCols, const StorageIndex blockRows, const StorageIndex blockCols, const StorageIndex blockOverlap);
		void analyzePattern(const MatrixType& mat);
		void factorize(const MatrixType& mat);

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
		* Q = BandedBlockedSparseQR_Ext<SparseMatrix<double> >(A).matrixQ();
		* \endcode
		* Internally, this call simply performs a sparse product between the matrix Q
		* and a sparse identity matrix. However, due to the fact that the sparse
		* reflectors are stored unsorted, two transpositions are needed to sort
		* them before performing the product.
		*/
		BandedBlockedSparseQR_ExtMatrixQReturnType<BandedBlockedSparseQR_Ext> matrixQ() const
		{
			return BandedBlockedSparseQR_ExtMatrixQReturnType<BandedBlockedSparseQR_Ext>(*this);
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
			eigen_assert(this->rows() == B.rows() && "BandedBlockedSparseQR_Ext::solve() : invalid number of rows in the right hand side matrix");

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

		/** \returns the solution X of \f$ A X = B \f$ using the current decomposition of A.
		*
		* \sa compute()
		*/
		template<typename Rhs>
		inline const Solve<BandedBlockedSparseQR_Ext, Rhs> solve(const MatrixBase<Rhs>& B) const
		{
			eigen_assert(m_isInitialized && "The factorization should be called first, use compute()");
			eigen_assert(this->rows() == B.rows() && "BandedBlockedSparseQR_Ext::solve() : invalid number of rows in the right hand side matrix");
			return Solve<BandedBlockedSparseQR_Ext, Rhs>(*this, B.derived());
		}
		template<typename Rhs>
		inline const Solve<BandedBlockedSparseQR_Ext, Rhs> solve(const SparseMatrixBase<Rhs>& B) const
		{
			eigen_assert(m_isInitialized && "The factorization should be called first, use compute()");
			eigen_assert(this->rows() == B.rows() && "BandedBlockedSparseQR_Ext::solve() : invalid number of rows in the right hand side matrix");
			return Solve<BandedBlockedSparseQR_Ext, Rhs>(*this, B.derived());
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
		typedef SparseBlockCOO_Ext<BlockYTY_Ext<Scalar, StorageIndex>, StorageIndex> SparseBlockYTY;

		bool m_analysisIsok;
		bool m_factorizationIsok;
		mutable ComputationInfo m_info;
		std::string m_lastError;
		MatrixQStorageType m_pmat;            // Temporary matrix
		MatrixRType m_R;                // The triangular factor matrix
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
		BlockInfoMap m_blockMap;		// Sparse matrix block information
		BlockInfoMapOrder m_blockOrder; // Sparse matrix block order

		Index m_denseStartRow;			// Dense subblock start row
		Index m_denseNumRows;			// Dense subblock no. rows

		template <typename, typename > friend struct BandedBlockedSparseQR_Ext_QProduct;

		void mergeBlocks(BlockInfoMapOrder &blockOrder, BlockInfoMap &blockMap, const StorageIndex maxColStep);
		void updateMat(const Index &fromIdx, const Index &toIdx, MatrixType &mat, const Index &blockK = -1);
	};

	/*
	* Helper structure holding band information for a single row.
	* Stores original row index (before any row reordering was performed),
	* index of the first nonzero (start) and last nonzero(end) in the band and the band length (length).
	*/
	
	template <typename IndexType>
	struct RowRange {
		IndexType origIdx;
		IndexType start;
		IndexType end;
		IndexType length;

		RowRange() : start(0), end(0), length(0) {
		}

		RowRange(const IndexType &origIdx, const IndexType &start, const IndexType &end)
			: origIdx(origIdx), start(start), end(end) {
			this->length = this->end - this->start + 1;
		}
	};
	
	/*
	* Helper function going through a block map and looking for a possibility to merge several blocks together
	* in order to obtain better dense block sizes for the YTY representation.
	*/
	template <typename MatrixType, typename OrderingType, int SuggestedBlockCols, bool MultiThreading>
	void BandedBlockedSparseQR_Ext<MatrixType, OrderingType, SuggestedBlockCols, MultiThreading>::mergeBlocks(BlockInfoMapOrder &blockOrder, BlockInfoMap &blockMap, const StorageIndex maxColStep) {
		BlockInfoMap newBlockMap;
		BlockInfoMapOrder newBlockOrder;
		MatrixBlockInfo firstBlock;
		typename MatrixType::StorageIndex prevBlockEndCol = 0;
		typename MatrixType::StorageIndex sumRows = 0;
		typename MatrixType::StorageIndex numCols = 0;
		typename MatrixType::StorageIndex colStep = 0;
		auto it = blockOrder.begin();
		for (; it != blockOrder.end(); ++it) {
			if (sumRows == 0) {
				firstBlock = blockMap.at(*it);
			}

			sumRows += blockMap.at(*it).numRows;
			numCols = (blockMap.at(*it).colIdx + blockMap.at(*it).numCols) - firstBlock.colIdx;
			colStep = blockMap.at(*it).colIdx - firstBlock.colIdx;

			if ((newBlockOrder.empty() || colStep >= maxColStep / 2 - 1) && sumRows > numCols && numCols >= SuggestedBlockCols) {
				newBlockOrder.push_back(firstBlock.colIdx);
				newBlockMap.insert(std::make_pair(firstBlock.colIdx, MatrixBlockInfo(firstBlock.rowIdx, firstBlock.colIdx, sumRows, numCols)));

				sumRows = 0;
				prevBlockEndCol = firstBlock.colIdx + numCols;
			}
		}
		// Process also last collection
		--it;
		if (sumRows > 0) {
			colStep = blockMap.at(*it).colIdx - firstBlock.colIdx;

			if (colStep >= maxColStep / 2 && sumRows > numCols && numCols >= SuggestedBlockCols) {
				newBlockOrder.push_back(firstBlock.colIdx);
				int numCols = (blockMap.at(*it).colIdx + blockMap.at(*it).numCols) - firstBlock.colIdx;
				newBlockMap.insert(std::make_pair(firstBlock.colIdx, MatrixBlockInfo(firstBlock.rowIdx, firstBlock.colIdx, sumRows, numCols)));
			}
			else {
				firstBlock = newBlockMap[newBlockOrder.back()];
				int numCols = (blockMap.at(*it).colIdx + blockMap.at(*it).numCols) - firstBlock.colIdx;
				newBlockMap[newBlockOrder.back()] = MatrixBlockInfo(firstBlock.rowIdx, firstBlock.colIdx, firstBlock.numRows + sumRows, numCols);
			}
		}

		// If the last block has numRows < numCols, start merging back
		while (newBlockMap[newBlockOrder.back()].numRows < newBlockMap[newBlockOrder.back()].numCols) {
			// Create new block info
			int rowIdx = newBlockMap[newBlockOrder.at(newBlockOrder.size() - 2)].rowIdx;
			int colIdx = newBlockMap[newBlockOrder.at(newBlockOrder.size() - 2)].colIdx;
			int numRows = newBlockMap[newBlockOrder.back()].rowIdx - newBlockMap[newBlockOrder.at(newBlockOrder.size() - 2)].rowIdx + newBlockMap[newBlockOrder.back()].numRows;
			int numCols = newBlockMap[newBlockOrder.back()].colIdx - newBlockMap[newBlockOrder.at(newBlockOrder.size() - 2)].colIdx + newBlockMap[newBlockOrder.back()].numCols;
			newBlockMap.erase(newBlockOrder.back());
			newBlockOrder.pop_back();

			newBlockOrder.push_back(colIdx);
			newBlockMap.insert(std::make_pair(colIdx, MatrixBlockInfo(rowIdx, colIdx, numRows, numCols)));
		}

		// Save the final banded block structure that will be used during the factorization process.
		blockOrder = newBlockOrder;
		blockMap = newBlockMap;
	}

	template <typename MatrixType, typename OrderingType, int SuggestedBlockCols, bool MultiThreading>
	void BandedBlockedSparseQR_Ext<MatrixType, OrderingType, SuggestedBlockCols, MultiThreading>::setPattern(const StorageIndex matRows, const StorageIndex matCols,
		const StorageIndex blockRows, const StorageIndex blockCols, const StorageIndex blockOverlap) {
		typedef RowRange<typename MatrixType::StorageIndex> MatrixRowRange;
		typedef std::map<typename MatrixType::StorageIndex, typename MatrixType::StorageIndex> BlockBandSize;
		typedef SparseMatrix<Scalar, RowMajor, typename MatrixType::StorageIndex> RowMajorMatrixType;

		Index n = matCols;
		Index m = matRows;
		Index diagSize = (std::min)(m, n);

		// In case we know the pattern, rows are already sorted, no permutation needed
		this->m_hasRowPermutation = false;
		this->m_rowPerm.resize(m);
		this->m_rowPerm.setIdentity();

		/******************************************************************/
		// 1) Set the block map based on block paramters passed ot this method	
		typename MatrixType::StorageIndex maxColStep = blockCols - blockOverlap;
		typename MatrixType::StorageIndex numBlocks = n / maxColStep;
		this->m_blockMap.clear();
		this->m_blockOrder.clear();
		typename MatrixType::StorageIndex rowIdx = 0;
		typename MatrixType::StorageIndex colIdx = 0;
		for (int i = 0; i < numBlocks; i++) {
			rowIdx = i * blockRows;
			colIdx = i * maxColStep;
			this->m_blockOrder.push_back(colIdx);
			if (i < numBlocks - 1) {
				this->m_blockMap.insert(std::make_pair(colIdx, MatrixBlockInfo(rowIdx, colIdx, blockRows, blockCols)));
			}
			else {
				// Last block need to be treated separately (only block overlap - we're at matrix bound)
				this->m_blockMap.insert(std::make_pair(colIdx, MatrixBlockInfo(rowIdx, colIdx, blockRows, blockCols - blockOverlap)));
			}
		}

		/******************************************************************/
		// 2) Go through the estimated block structure
		// And merge several blocks together if needed/possible in order to form reasonably big banded blocks
		this->mergeBlocks(this->m_blockOrder, this->m_blockMap, maxColStep);

		/******************************************************************/
		// 3) Finalize
		m_R.resize(matRows, matCols);

		m_analysisIsok = true;
	}

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
	template <typename MatrixType, typename OrderingType, int SuggestedBlockCols, bool MultiThreading>
	void BandedBlockedSparseQR_Ext<MatrixType, OrderingType, SuggestedBlockCols, MultiThreading>::analyzePattern(const MatrixType& mat)
	{
		typedef RowRange<typename MatrixType::StorageIndex> MatrixRowRange;
		typedef std::map<typename MatrixType::StorageIndex, typename MatrixType::StorageIndex> BlockBandSize;
		typedef SparseMatrix<Scalar, RowMajor, typename MatrixType::StorageIndex> RowMajorMatrixType;

		Index n = mat.cols();
		Index m = mat.rows();
		Index diagSize = (std::min)(m, n);

		// Looking for as-banded-as-possible structure in the matrix
		/******************************************************************/
		// 1) Compute and store band information for each row in the matrix
		//BlockBandSize bandWidths, bandHeights;
		RowMajorMatrixType rmMat(mat);
		std::vector<MatrixRowRange> rowRanges;
		for (typename MatrixType::StorageIndex j = 0; j < rmMat.rows(); j++) {
			typename RowMajorMatrixType::InnerIterator rowIt(rmMat, j);
			typename MatrixType::StorageIndex startIdx = rowIt.index();
			typename MatrixType::StorageIndex endIdx = startIdx;
			while (++rowIt) { endIdx = rowIt.index(); }	// FixMe: Is there a better way?
			rowRanges.push_back(MatrixRowRange(j, startIdx, endIdx));
			/*
			typename MatrixType::StorageIndex bw = endIdx - startIdx + 1;
			if (bandWidths.find(startIdx) == bandWidths.end()) {
				bandWidths.insert(std::make_pair(startIdx, bw));
			}
			else {
				if (bandWidths.at(startIdx) < bw) {
					bandWidths.at(startIdx) = bw;
				}
			}

			if (bandHeights.find(startIdx) == bandHeights.end()) {
				bandHeights.insert(std::make_pair(startIdx, 1));
			}
			else {
				bandHeights.at(startIdx) += 1;
			}
			*/
		}

		/******************************************************************/
		// 2) Sort the rows to form as-banded-as-possible matrix
		// Set an indicator whether row sorting is needed
		this->m_hasRowPermutation = !std::is_sorted(rowRanges.begin(), rowRanges.end(), [](const MatrixRowRange &lhs, const MatrixRowRange &rhs) {
			return (lhs.start < rhs.start);
		});
		// Perform the actual row sorting if needed
		if (this->m_hasRowPermutation) {
			std::stable_sort(rowRanges.begin(), rowRanges.end(), [](const MatrixRowRange &lhs, const MatrixRowRange &rhs) {
				return (lhs.length < rhs.length);//(lhs.start < rhs.start);
			});
			std::stable_sort(rowRanges.begin(), rowRanges.end(), [](const MatrixRowRange &lhs, const MatrixRowRange &rhs) {
				if (lhs.length <= 2 * rhs.length) {
					return lhs.start < rhs.start;//(lhs.start < rhs.start);
				}
				else {
					return false;
				}
			});
		}

		/******************************************************************/
		// 3) Search for column blocks (blocks of columns starting at the same row)		
		// And record the estimated block structure
		this->m_blockMap.clear();
		this->m_blockOrder.clear();
		Eigen::Matrix<typename MatrixType::StorageIndex, Dynamic, 1> permIndices(rowRanges.size());
		typename MatrixType::StorageIndex rowIdx = 0;
		typename MatrixType::StorageIndex lastStart = rowRanges.at(0).start;
		bool skipDense = false;
		for (auto it = rowRanges.begin(); it != rowRanges.end(); ++it, rowIdx++) {
			permIndices(it->origIdx) = rowIdx;

			if(!skipDense) {
				if (lastStart > it->start) {
					std::cout << "Breaking at row " << rowIdx << " ..." << std::endl;
					// Now we know where the dense block will start
					this->m_denseStartRow = rowIdx;
					this->m_denseNumRows = mat.rows() - this->m_denseStartRow;
					std::cout << this->m_denseStartRow << ", " << this->m_denseNumRows << std::endl;
					skipDense = true;
				}
				lastStart = it->start;

				// std::find is terribly slow for large arrays
				// assuming m_blockOrder is ordered, we can use binary_search
				// is m_blockOrder always ordered? can we always use binary_search???
				//if (!std::binary_search(this->m_blockOrder.begin(), this->m_blockOrder.end(), it->start)) {
					//if (std::find(this->m_blockOrder.begin(), this->m_blockOrder.end(), it->start) == this->m_blockOrder.end()) {
				//	this->m_blockOrder.push_back(it->start);
				//	this->m_blockMap.insert(std::make_pair(it->start, MatrixBlockInfo(rowIdx, it->start, bandHeights.at(it->start), it->length)));
				//}
			}
		}
		// Create row permutation matrix that achieves the desired row reordering
		this->m_rowPerm = PermutationType(permIndices);

		// Now apply the row permutation
		m_pmat = this->m_rowPerm * mat;

		// Now go through the sparse part of the block by columns and finally detect block sizes...
		// 1) Compute and store band information for each row in the matrix
		BlockBandSize bandWidths, bandHeights;
		MatrixType spBlock = m_pmat.block(0, 0, this->m_denseStartRow, m_pmat.cols());
		std::vector<MatrixRowRange> colRanges;
		for (typename MatrixType::StorageIndex j = 0; j < spBlock.cols(); j++) {
			typename MatrixType::InnerIterator colIt(spBlock, j);
			typename MatrixType::StorageIndex startIdx = colIt.index();
			typename MatrixType::StorageIndex endIdx = startIdx;
			while (++colIt) { endIdx = colIt.index(); }	// FixMe: Is there a better way?
			colRanges.push_back(MatrixRowRange(j, startIdx, endIdx));

			typename MatrixType::StorageIndex bw = endIdx - startIdx + 1;
			if (bandWidths.find(startIdx) == bandWidths.end()) {
				bandWidths.insert(std::make_pair(startIdx, bw));
			}
			else {
				if (bandWidths.at(startIdx) < bw) {
					bandWidths.at(startIdx) = bw;
				}
			}

			if (bandHeights.find(startIdx) == bandHeights.end()) {
				bandHeights.insert(std::make_pair(startIdx, 1));
			}
			else {
				bandHeights.at(startIdx) += 1;
			}
		}

		// Compose block map
		this->m_blockMap.clear();
		this->m_blockOrder.clear();
		typename MatrixType::StorageIndex colIdx = 0;
		for (auto it = colRanges.begin(); it != colRanges.end(); ++it, colIdx++) {
			// std::find is terribly slow for large arrays
			// assuming m_blockOrder is ordered, we can use binary_search
			// is m_blockOrder always ordered? can we always use binary_search???
			if (!std::binary_search(this->m_blockOrder.begin(), this->m_blockOrder.end(), it->start)) {
				this->m_blockOrder.push_back(it->start);
				this->m_blockMap.insert(std::make_pair(it->start, MatrixBlockInfo(it->start, colIdx, it->length, bandHeights.at(it->start)))); // Switch numRows and numCols (analysis was done by columns)
			}
		}

		/******************************************************************/
		// 4) Go through the estimated block structure
		// And merge several blocks together if needed/possible in order to form reasonably big banded blocks
		//this->mergeBlocks(this->m_blockOrder, this->m_blockMap, maxColStep);

		/******************************************************************/
		// 5) Finalize
		m_R.resize(mat.rows(), mat.cols());

		/*
		std::cout << "No of blocks: " << this->m_blockOrder.size() << std::endl;
		if (this->m_blockOrder.size() < 3000) {
			for (int i = 0; i < this->m_blockOrder.size(); i++) {
				std::cout << this->m_blockMap.at(this->m_blockOrder.at(i)).rowIdx << ", " << this->m_blockMap.at(this->m_blockOrder.at(i)).colIdx << ": " << this->m_blockMap.at(this->m_blockOrder.at(i)).numRows << ", " << this->m_blockMap.at(this->m_blockOrder.at(i)).numCols << " || ";
			}
			std::cout << std::endl;
		}*/

		m_analysisIsok = true;
	}
	
	template <typename MatrixType, typename OrderingType, int SuggestedBlockCols, bool MultiThreading>
	void BandedBlockedSparseQR_Ext<MatrixType, OrderingType, SuggestedBlockCols, MultiThreading>::updateMat(const Index &fromIdx, const Index &toIdx, MatrixType &mat, const Index &blockK) {
		// Now update the unsolved rest of m_pmat
		typename SparseBlockYTY::Element blockYTY = this->m_blocksYT[this->m_blocksYT.size() - 1];
		VectorXd resColJd;
		VectorXd tmpResColJ;

		if (blockK < 0) {
			for (int ci = fromIdx; ci < toIdx; ci++) {
				// Use temporary vector resColJ inside of the for loop - faster access
				resColJd = mat.col(ci).toDense();
				for (Index k = 0; k < this->m_blocksYT.size(); k++) {
					typename MatrixType::StorageIndex subdiagElems = this->m_blocksYT[k].value.rows() - this->m_blocksYT[k].value.cols();
					FULL_TO_BLOCK_VEC_EXT(resColJd, tmpResColJ, this->m_blocksYT[k].value.rows(), this->m_blocksYT[k].value.cols(), this->m_blocksYT[k].row, this->m_blocksYT[k].value.numZeros(), subdiagElems,
						this->m_denseStartRow, this->m_denseNumRows)

						// We can afford noalias() in this case
						tmpResColJ += this->m_blocksYT[k].value.multTransposed(tmpResColJ);

					BLOCK_VEC_TO_FULL_EXT(resColJd, tmpResColJ, this->m_blocksYT[k].value.rows(), this->m_blocksYT[k].value.cols(), this->m_blocksYT[k].row, this->m_blocksYT[k].value.numZeros(), subdiagElems,
						this->m_denseStartRow, this->m_denseNumRows)
				}
				// Write the result back to ci-th column of res
				mat.col(ci) = resColJd.sparseView();
			}
		}
		else {
			for (int ci = fromIdx; ci < toIdx; ci++) {
				// Use temporary vector resColJ inside of the for loop - faster access
				resColJd = mat.col(ci).toDense();
				Index k = blockK;
				typename MatrixType::StorageIndex subdiagElems = this->m_blocksYT[k].value.rows() - this->m_blocksYT[k].value.cols();
				FULL_TO_BLOCK_VEC_EXT(resColJd, tmpResColJ, this->m_blocksYT[k].value.rows(), this->m_blocksYT[k].value.cols(), this->m_blocksYT[k].row, this->m_blocksYT[k].value.numZeros(), subdiagElems,
					this->m_denseStartRow, this->m_denseNumRows)

					// We can afford noalias() in this case
					tmpResColJ += this->m_blocksYT[k].value.multTransposed(tmpResColJ);

				BLOCK_VEC_TO_FULL_EXT(resColJd, tmpResColJ, this->m_blocksYT[k].value.rows(), this->m_blocksYT[k].value.cols(), this->m_blocksYT[k].row, this->m_blocksYT[k].value.numZeros(), subdiagElems,
					this->m_denseStartRow, this->m_denseNumRows)
				// Write the result back to ci-th column of res
				mat.col(ci) = resColJd.sparseView();
			}
		}

		/*
		//for (int k = 0; k < blockYTY.cols(); k++) {
		for (int ci = fromIdx; ci < toIdx; ci++) {
			// Use temporary vector resColJ inside of the for loop - faster access
			resColJd = mat.col(ci).toDense();
			typename MatrixType::StorageIndex subdiagElems = blockYTY.value.rows() - blockYTY.value.cols();
			FULL_TO_BLOCK_VEC_EXT(resColJd, tmpResColJ, blockYTY.value.rows(), blockYTY.value.cols(), blockYTY.row, blockYTY.value.numZeros(), subdiagElems,
				this->m_denseStartRow, this->m_denseNumRows)

			// We can afford noalias() in this case
			tmpResColJ += blockYTY.value.multTransposed(tmpResColJ);
			//tmpResColJ += blockYTY.value * tmpResColJ;

			BLOCK_VEC_TO_FULL_EXT(resColJd, tmpResColJ, blockYTY.value.rows(), blockYTY.value.cols(), blockYTY.row, blockYTY.value.numZeros(), subdiagElems,
				this->m_denseStartRow, this->m_denseNumRows)

			// Write the result back to ci-th column of res
			mat.col(ci) = resColJd.sparseView();
		}
		//}
		*/
	}

	/** \brief Performs the numerical QR factorization of the input matrix
	*
	* The function BandedBlockedSparseQR_Ext::analyzePattern(const MatrixType&) must have been called beforehand with
	* a matrix having the same sparsity pattern than \a mat.
	*
	* \param mat The sparse column-major matrix
	*/
	template <typename MatrixType, typename OrderingType, int SuggestedBlockCols, bool MultiThreading>
	void BandedBlockedSparseQR_Ext<MatrixType, OrderingType, SuggestedBlockCols, MultiThreading>::factorize(const MatrixType& mat)
	{
		// Not rank-revealing, column permutation is identity
		m_outputPerm_c.setIdentity(mat.cols());

		// Permute the input matrix using the precomputed row permutation
		m_pmat = (this->m_rowPerm * mat);

		// Triplet array for the matrix R
//		Eigen::TripletArray<Scalar, typename MatrixType::Index> Rvals(2 * mat.nonZeros());

		// Dense QR solver used for each dense block 
		typedef Matrix<Scalar, Dynamic, Dynamic> DenseMatrixType;
		Eigen::HouseholderQR<DenseMatrixType> houseqr;
		Index numBlocks = this->m_blockOrder.size();

		// Prepare the first block
		MatrixBlockInfo bi = this->m_blockMap.at(this->m_blockOrder.at(0));
		DenseMatrixType Ji = DenseMatrixType(bi.numRows + this->m_denseNumRows, bi.numCols);
		Ji.topRows(bi.numRows) = m_pmat.block(bi.rowIdx, bi.colIdx, bi.numRows, bi.numCols);
		Ji.bottomRows(this->m_denseNumRows) = m_pmat.block(this->m_denseStartRow, bi.colIdx, this->m_denseNumRows, bi.numCols).toDense();
		Index activeRows = bi.numRows;
		Index numZeros = 0;

		// Auxiliary variables
		MatrixBlockInfo biNext;
		Index colIncrement, blockOverlap;

		// Process all blocks
		for (Index i = 0; i < numBlocks; i++) {
			clock_t begin = clock();
			// Current block info
			bi = this->m_blockMap.at(this->m_blockOrder.at(i));

			// 1) Solve the current dense block using dense Householder QR
			houseqr.compute(Ji);

			// 2) Create matrices T and Y
			MatrixXd T = MatrixXd::Zero(bi.numCols, bi.numCols);
			MatrixXd Y = MatrixXd::Zero(activeRows + this->m_denseNumRows, bi.numCols);
			VectorXd v = VectorXd::Zero(activeRows + this->m_denseNumRows);
			VectorXd z = VectorXd::Zero(activeRows + this->m_denseNumRows);
			v(0) = 1.0;
			v.segment(1, houseqr.householderQ().essentialVector(0).rows()) = houseqr.householderQ().essentialVector(0);
			Y.col(0) = v;
			T(0, 0) = -houseqr.hCoeffs()(0);
			for (typename MatrixType::StorageIndex bc = 1; bc < bi.numCols; bc++) {
				v.setZero();
				v(bc) = 1.0;
				v.segment(bc + 1, houseqr.householderQ().essentialVector(bc).rows()) = houseqr.householderQ().essentialVector(bc);
				
				z = -houseqr.hCoeffs()(bc) * (T * (Y.transpose() * v));

				Y.col(bc) = v;
				T.col(bc) = z;
				T(bc, bc) = -houseqr.hCoeffs()(bc);
			}
			// Save current Y and T. The block YTY contains a main diagonal and subdiagonal part separated by (numZeros) zero rows.
			Index diagIdx = bi.colIdx;
			m_blocksYT.insert(SparseBlockYTY::Element(diagIdx, diagIdx, BlockYTY_Ext<Scalar, StorageIndex>(Y, T, numZeros, activeRows)));

			// 3) Get the R part of the dense QR decomposition 
	/*		MatrixXd V = houseqr.matrixQR().template triangularView<Upper>();
			// Update sparse R with the rows solved in this step
			int solvedRows = (i == numBlocks - 1) ? bi.numRows : this->m_blockMap.at(this->m_blockOrder.at(i + 1)).colIdx - bi.colIdx;
			for (typename MatrixType::StorageIndex br = 0; br < solvedRows; br++) {
				for (typename MatrixType::StorageIndex bc = 0; bc < bi.numCols; bc++) {
					Rvals.add_if_nonzero(diagIdx + br, bi.colIdx + bc, V(br, bc));
				}
			}
	*/
			this->updateMat(bi.colIdx, bi.colIdx + bi.numCols, m_pmat, this->m_blocksYT.size() - 1);

			// 4) If this is not the last block, proceed to the next block
			if (i < numBlocks - 1) {
				biNext = this->m_blockMap.at(this->m_blockOrder.at(i + 1));

				//if (i > 0) {
				// Now update the unsolved rest of m_pmat
				this->updateMat(biNext.colIdx, biNext.colIdx + biNext.numCols, m_pmat);
				//}

				blockOverlap = (bi.colIdx + bi.numCols) - biNext.colIdx;
				colIncrement = bi.numCols - blockOverlap;
				activeRows = biNext.rowIdx + biNext.numRows - biNext.colIdx;
				//activeRows = bi.numRows + biNext.numRows - colIncrement;	// subtracting ((bi.rowIdx + bi.numRows) - biNext.rowIdx) is handling the row-overlap case
				//activeRows = biNext.numRows;
				//numZeros = (biNext.rowIdx + biNext.numRows) - activeRows - colIncrement;// -biNext.colIdx;
				numZeros = 0;
				//numZeros = biNext.rowIdx - biNext.colIdx;
				numZeros = (numZeros < 0) ? 0 : numZeros;

				typename MatrixType::StorageIndex numCols = (biNext.numCols >= blockOverlap) ? biNext.numCols : blockOverlap;
				Ji = DenseMatrixType(activeRows + this->m_denseNumRows, numCols);
				Ji.topRows(activeRows) = m_pmat.block(biNext.colIdx, biNext.colIdx, activeRows, numCols).toDense();
				//Ji.topRows(activeRows) = m_pmat.block(bi.rowIdx + colIncrement, biNext.colIdx, activeRows, numCols).toDense();
				//Ji.topRows(activeRows) = m_pmat.block(biNext.rowIdx, biNext.colIdx, activeRows, numCols).toDense();
				Ji.bottomRows(this->m_denseNumRows) = m_pmat.block(this->m_denseStartRow, biNext.colIdx, this->m_denseNumRows, numCols).toDense();

//				if (blockOverlap > 0) {
//					Ji.block(0, 0, activeRows - biNext.numRows, blockOverlap) = V.block(colIncrement, colIncrement, activeRows - biNext.numRows, blockOverlap);
//				}
			}
			std::cout << "iter   " << double(clock() - begin) / CLOCKS_PER_SEC << "s\n";
		}

		// 5) Finalize the R matrix and set factorization-related flags
//		m_R.setFromTriplets(Rvals.begin(), Rvals.end());
//		m_R.makeCompressed();

		m_R = m_pmat;
		m_R.makeCompressed();

		m_nonzeropivots = m_R.cols();	// Assuming all cols are nonzero

		m_isInitialized = true;
		m_factorizationIsok = true;
		m_info = Success;
	}

	/*
	* General Householder product evaluation performing Q * A or Q.T * A.
	* The general version is assuming that A is sparse and that the output will be sparse as well.
	* Offers single-threaded and multi-threaded implementation.
	* The choice of implementation depends on a template parameter of the BandedBlockedSparseQR_Ext class.
	* The single-threaded implementation cannot work in-place. It is implemented this way for performance related reasons.
	*/
	template <typename BandedBlockedSparseQR_ExtType, typename Derived>
	struct BandedBlockedSparseQR_Ext_QProduct : ReturnByValue<BandedBlockedSparseQR_Ext_QProduct<BandedBlockedSparseQR_ExtType, Derived> >
	{
		typedef typename BandedBlockedSparseQR_ExtType::MatrixType MatrixType;
		typedef typename BandedBlockedSparseQR_ExtType::Scalar Scalar;

		typedef typename internal::BandedBlockedSparseQR_Ext_traits<MatrixType>::Vector SparseVector;

		// Get the references 
		BandedBlockedSparseQR_Ext_QProduct(const BandedBlockedSparseQR_ExtType& qr, const Derived& other, bool transpose) :
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
											typename MatrixType::StorageIndex subdiagElems = m_qr.m_blocksYT[k].value.rows() - m_qr.m_blocksYT[k].value.cols();
											FULL_TO_BLOCK_VEC_EXT(resColJd, tmpResColJ, m_qr.m_blocksYT[k].value.rows(), m_qr.m_blocksYT[k].value.cols(), m_qr.m_blocksYT[k].row, m_qr.m_blocksYT[k].value.numZeros(), subdiagElems,
												m_qr.m_denseStartRow, m_qr.m_denseNumRows)

												// We can afford noalias() in this case
												tmpResColJ.noalias() += m_qr.m_blocksYT[k].value.multTransposed(tmpResColJ);

											BLOCK_VEC_TO_FULL_EXT(resColJd, tmpResColJ, m_qr.m_blocksYT[k].value.rows(), m_qr.m_blocksYT[k].value.cols(), m_qr.m_blocksYT[k].row, m_qr.m_blocksYT[k].value.numZeros(), subdiagElems,
												m_qr.m_denseStartRow, m_qr.m_denseNumRows)
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
											typename MatrixType::StorageIndex subdiagElems = m_qr.m_blocksYT[k].value.rows() - m_qr.m_blocksYT[k].value.cols();
											FULL_TO_BLOCK_VEC_EXT(resColJd, tmpResColJ, m_qr.m_blocksYT[k].value.rows(), m_qr.m_blocksYT[k].value.cols(), m_qr.m_blocksYT[k].row, m_qr.m_blocksYT[k].value.numZeros(), subdiagElems,
												m_qr.m_denseStartRow, m_qr.m_denseNumRows)

												// We can afford noalias() in this case
												tmpResColJ.noalias() += m_qr.m_blocksYT[k].value * tmpResColJ;

											BLOCK_VEC_TO_FULL_EXT(resColJd, tmpResColJ, m_qr.m_blocksYT[k].value.rows(), m_qr.m_blocksYT[k].value.cols(), m_qr.m_blocksYT[k].row, m_qr.m_blocksYT[k].value.numZeros(), subdiagElems,
												m_qr.m_denseStartRow, m_qr.m_denseNumRows)
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
				res.reserve(m_other.rows() * m_other.cols() * 0.25);// FixMe: Better estimation of nonzeros?

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
							typename MatrixType::StorageIndex subdiagElems = m_qr.m_blocksYT[k].value.rows() - m_qr.m_blocksYT[k].value.cols();
							FULL_TO_BLOCK_VEC_EXT(resColJd, tmpResColJ, m_qr.m_blocksYT[k].value.rows(), m_qr.m_blocksYT[k].value.cols(), m_qr.m_blocksYT[k].row, m_qr.m_blocksYT[k].value.numZeros(), subdiagElems,
								m_qr.m_denseStartRow, m_qr.m_denseNumRows)

								// We can afford noalias() in this case
								tmpResColJ.noalias() += m_qr.m_blocksYT[k].value.multTransposed(tmpResColJ);

							BLOCK_VEC_TO_FULL_EXT(resColJd, tmpResColJ, m_qr.m_blocksYT[k].value.rows(), m_qr.m_blocksYT[k].value.cols(), m_qr.m_blocksYT[k].row, m_qr.m_blocksYT[k].value.numZeros(), subdiagElems,
								m_qr.m_denseStartRow, m_qr.m_denseNumRows)
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
							typename MatrixType::StorageIndex subdiagElems = m_qr.m_blocksYT[k].value.rows() - m_qr.m_blocksYT[k].value.cols();
							FULL_TO_BLOCK_VEC_EXT(resColJd, tmpResColJ, m_qr.m_blocksYT[k].value.rows(), m_qr.m_blocksYT[k].value.cols(), m_qr.m_blocksYT[k].row, m_qr.m_blocksYT[k].value.numZeros(), subdiagElems,
								m_qr.m_denseStartRow, m_qr.m_denseNumRows)

								// We can afford noalias() in this case
								tmpResColJ.noalias() += m_qr.m_blocksYT[k].value * tmpResColJ;

							BLOCK_VEC_TO_FULL_EXT(resColJd, tmpResColJ, m_qr.m_blocksYT[k].value.rows(), m_qr.m_blocksYT[k].value.cols(), m_qr.m_blocksYT[k].row, m_qr.m_blocksYT[k].value.numZeros(), subdiagElems,
								m_qr.m_denseStartRow, m_qr.m_denseNumRows)
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

		const BandedBlockedSparseQR_ExtType& m_qr;
		const Derived& m_other;
		bool m_transpose;
	};

	/*
	* Specialization of the Householder product evaluation performing Q * A or Q.T * A
	* for the case when A and the output are dense vectors.=
	* Offers only single-threaded implementation as the overhead of multithreading would not bring any speedup for a dense vector (A is single column).
	*/
	template <typename BandedBlockedSparseQR_ExtType>
	struct BandedBlockedSparseQR_Ext_QProduct<BandedBlockedSparseQR_ExtType, VectorX> : ReturnByValue<BandedBlockedSparseQR_Ext_QProduct<BandedBlockedSparseQR_ExtType, VectorX> >
	{
		typedef typename BandedBlockedSparseQR_ExtType::MatrixType MatrixType;
		typedef typename BandedBlockedSparseQR_ExtType::Scalar Scalar;

		// Get the references 
		BandedBlockedSparseQR_Ext_QProduct(const BandedBlockedSparseQR_ExtType& qr, const VectorX& other, bool transpose) :
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
					typename MatrixType::StorageIndex subdiagElems = m_qr.m_blocksYT[k].value.rows() - m_qr.m_blocksYT[k].value.cols();
					FULL_TO_BLOCK_VEC_EXT(res, partialRes, m_qr.m_blocksYT[k].value.rows(), m_qr.m_blocksYT[k].value.cols(), m_qr.m_blocksYT[k].row, m_qr.m_blocksYT[k].value.numZeros(), subdiagElems,
						m_qr.m_denseStartRow, m_qr.m_denseNumRows)

						// We can afford noalias() in this case
						partialRes.noalias() += m_qr.m_blocksYT[k].value.multTransposed(partialRes);

					BLOCK_VEC_TO_FULL_EXT(res, partialRes, m_qr.m_blocksYT[k].value.rows(), m_qr.m_blocksYT[k].value.cols(), m_qr.m_blocksYT[k].row, m_qr.m_blocksYT[k].value.numZeros(), subdiagElems, 
						m_qr.m_denseStartRow, m_qr.m_denseNumRows)
				}
			}
			else
			{
				// Compute res = Q * other (other is vector - only one column => no iterations of j)
				VectorX partialRes;
				for (Index k = m_qr.m_blocksYT.size() - 1; k >= 0; k--) {
					typename MatrixType::StorageIndex subdiagElems = m_qr.m_blocksYT[k].value.rows() - m_qr.m_blocksYT[k].value.cols();
					FULL_TO_BLOCK_VEC_EXT(res, partialRes, m_qr.m_blocksYT[k].value.rows(), m_qr.m_blocksYT[k].value.cols(), m_qr.m_blocksYT[k].row, m_qr.m_blocksYT[k].value.numZeros(), subdiagElems, 
						m_qr.m_denseStartRow, m_qr.m_denseNumRows)

						// We can afford noalias() in this case
						partialRes.noalias() += m_qr.m_blocksYT[k].value * partialRes;

					BLOCK_VEC_TO_FULL_EXT(res, partialRes, m_qr.m_blocksYT[k].value.rows(), m_qr.m_blocksYT[k].value.cols(), m_qr.m_blocksYT[k].row, m_qr.m_blocksYT[k].value.numZeros(), subdiagElems, 
						m_qr.m_denseStartRow, m_qr.m_denseNumRows)
				}
			}
		}

		const BandedBlockedSparseQR_ExtType& m_qr;
		const VectorX& m_other;
		bool m_transpose;
	};

	template<typename BandedBlockedSparseQR_ExtType>
	struct BandedBlockedSparseQR_ExtMatrixQReturnType : public EigenBase<BandedBlockedSparseQR_ExtMatrixQReturnType<BandedBlockedSparseQR_ExtType> >
	{
		typedef typename BandedBlockedSparseQR_ExtType::Scalar Scalar;
		typedef Matrix<Scalar, Dynamic, Dynamic> DenseMatrix;
		enum {
			RowsAtCompileTime = Dynamic,
			ColsAtCompileTime = Dynamic
		};
		explicit BandedBlockedSparseQR_ExtMatrixQReturnType(const BandedBlockedSparseQR_ExtType& qr) : m_qr(qr) {}
		/*BandedBlockedSparseQR_Ext_QProduct<BandedBlockedSparseQR_ExtType, Derived> operator*(const MatrixBase<Derived>& other)
		{
		return BandedBlockedSparseQR_Ext_QProduct<BandedBlockedSparseQR_ExtType,Derived>(m_qr,other.derived(),false);
		}*/
		template<typename Derived>
		BandedBlockedSparseQR_Ext_QProduct<BandedBlockedSparseQR_ExtType, Derived> operator*(const MatrixBase<Derived>& other)
		{
			return BandedBlockedSparseQR_Ext_QProduct<BandedBlockedSparseQR_ExtType, Derived>(m_qr, other.derived(), false);
		}
		template<typename _Scalar, int _Options, typename _Index>
		BandedBlockedSparseQR_Ext_QProduct<BandedBlockedSparseQR_ExtType, SparseMatrix<_Scalar, _Options, _Index>> operator*(const SparseMatrix<_Scalar, _Options, _Index>& other)
		{
			return BandedBlockedSparseQR_Ext_QProduct<BandedBlockedSparseQR_ExtType, SparseMatrix<_Scalar, _Options, _Index>>(m_qr, other, false);
		}
		BandedBlockedSparseQR_ExtMatrixQTransposeReturnType<BandedBlockedSparseQR_ExtType> adjoint() const
		{
			return BandedBlockedSparseQR_ExtMatrixQTransposeReturnType<BandedBlockedSparseQR_ExtType>(m_qr);
		}
		inline Index rows() const { return m_qr.rows(); }
		inline Index cols() const { return m_qr.rows(); }
		// To use for operations with the transpose of Q
		BandedBlockedSparseQR_ExtMatrixQTransposeReturnType<BandedBlockedSparseQR_ExtType> transpose() const
		{
			return BandedBlockedSparseQR_ExtMatrixQTransposeReturnType<BandedBlockedSparseQR_ExtType>(m_qr);
		}

		const BandedBlockedSparseQR_ExtType& m_qr;
	};

	template<typename BandedBlockedSparseQR_ExtType>
	struct BandedBlockedSparseQR_ExtMatrixQTransposeReturnType
	{
		explicit BandedBlockedSparseQR_ExtMatrixQTransposeReturnType(const BandedBlockedSparseQR_ExtType& qr) : m_qr(qr) {}
		template<typename Derived>
		BandedBlockedSparseQR_Ext_QProduct<BandedBlockedSparseQR_ExtType, Derived> operator*(const MatrixBase<Derived>& other)
		{
			return BandedBlockedSparseQR_Ext_QProduct<BandedBlockedSparseQR_ExtType, Derived>(m_qr, other.derived(), true);
		}
		template<typename _Scalar, int _Options, typename _Index>
		BandedBlockedSparseQR_Ext_QProduct<BandedBlockedSparseQR_ExtType, SparseMatrix<_Scalar, _Options, _Index>> operator*(const SparseMatrix<_Scalar, _Options, _Index>& other)
		{
			return BandedBlockedSparseQR_Ext_QProduct<BandedBlockedSparseQR_ExtType, SparseMatrix<_Scalar, _Options, _Index>>(m_qr, other, true);
		}
		const BandedBlockedSparseQR_ExtType& m_qr;
	};

	namespace internal {

		template<typename BandedBlockedSparseQR_ExtType>
		struct evaluator_traits<BandedBlockedSparseQR_ExtMatrixQReturnType<BandedBlockedSparseQR_ExtType> >
		{
			typedef typename BandedBlockedSparseQR_ExtType::MatrixType MatrixType;
			typedef typename storage_kind_to_evaluator_kind<typename MatrixType::StorageKind>::Kind Kind;
			typedef SparseShape Shape;
		};

		template< typename DstXprType, typename BandedBlockedSparseQR_ExtType>
		struct Assignment<DstXprType, BandedBlockedSparseQR_ExtMatrixQReturnType<BandedBlockedSparseQR_ExtType>, internal::assign_op<typename DstXprType::Scalar, typename DstXprType::Scalar>, Sparse2Sparse>
		{
			typedef BandedBlockedSparseQR_ExtMatrixQReturnType<BandedBlockedSparseQR_ExtType> SrcXprType;
			typedef typename DstXprType::Scalar Scalar;
			typedef typename DstXprType::StorageIndex StorageIndex;
			static void run(DstXprType &dst, const SrcXprType &src, const internal::assign_op<Scalar, Scalar> &/*func*/)
			{
				typename DstXprType::PlainObject idMat(src.m_qr.rows(), src.m_qr.rows());
				idMat.setIdentity();
				// Sort the sparse householder reflectors if needed
				//const_cast<BandedBlockedSparseQR_ExtType *>(&src.m_qr)->_sort_matrix_Q();
				dst = BandedBlockedSparseQR_Ext_QProduct<BandedBlockedSparseQR_ExtType, DstXprType>(src.m_qr, idMat, false);
			}
		};

		template< typename DstXprType, typename BandedBlockedSparseQR_ExtType>
		struct Assignment<DstXprType, BandedBlockedSparseQR_ExtMatrixQReturnType<BandedBlockedSparseQR_ExtType>, internal::assign_op<typename DstXprType::Scalar, typename DstXprType::Scalar>, Sparse2Dense>
		{
			typedef BandedBlockedSparseQR_ExtMatrixQReturnType<BandedBlockedSparseQR_ExtType> SrcXprType;
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
