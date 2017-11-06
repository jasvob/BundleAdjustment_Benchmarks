// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2016 Andrew Fitzgibbon <awf@microsoft.com>
// Copyright (C) 2012-2013 Desire Nuentsa <desire.nuentsa_wakam@inria.fr>
// Copyright (C) 2012-2014 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_BLOCK_DIAGONAL_SPARSE_QR_EXT_H
#define EIGEN_BLOCK_DIAGONAL_SPARSE_QR_EXT_H

namespace Eigen {
	/**
	* \ingroup SparseQR_Module
	* \class BlockDiagonalSparseQR_Ext
	* \brief QR factorization of block-diagonal matrix
	*
	* \implsparsesolverconcept
	*
	*/
	template<typename _MatrixType, typename _BlockQRSolver>
	class BlockDiagonalSparseQR_Ext : public SparseSolverBase<BlockDiagonalSparseQR_Ext<_MatrixType, _BlockQRSolver> >
	{
	protected:
		typedef SparseSolverBase<BlockDiagonalSparseQR_Ext<_MatrixType, _BlockQRSolver> > Base;
		using Base::m_isInitialized;
	public:
		using Base::_solve_impl;
		typedef _MatrixType MatrixType;
		typedef _BlockQRSolver BlockQRSolver;
		typedef typename BlockQRSolver::MatrixType BlockMatrixType;
		typedef typename MatrixType::Scalar Scalar;
		typedef typename MatrixType::RealScalar RealScalar;
		typedef typename MatrixType::StorageIndex StorageIndex;
		typedef typename MatrixType::Index Index;
		typedef Matrix<StorageIndex, Dynamic, 1> IndexVector;
		typedef Matrix<Scalar, Dynamic, 1> ScalarVector;

		typedef SparseMatrix<Scalar, RowMajor, StorageIndex> MatrixQType;
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
		BlockDiagonalSparseQR_Ext() : m_analysisIsok(false), m_factorizationIsok(false), m_hasRowPermutation(false)
		{ }

		/** Construct a QR factorization of the matrix \a mat.
		*
		* \warning The matrix \a mat must be in compressed mode (see SparseMatrix::makeCompressed()).
		*
		* \sa compute()
		*/
		explicit BlockDiagonalSparseQR_Ext(const MatrixType& mat) : m_analysisIsok(false), m_factorizationIsok(false), m_hasRowPermutation(false)
		{
			compute(mat);
		}

		/** Computes the QR factorization of the sparse matrix \a mat.
		*
		* \warning The matrix \a mat must be in compressed mode (see SparseMatrix::makeCompressed()).
		*
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
			factorize(mat);
		}
		void setPattern(const StorageIndex matRows, const StorageIndex matCols, const StorageIndex blockRows, const StorageIndex blockCols);
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
		* Q = SparseQR<SparseMatrix<double> >(A).matrixQ();
		* \endcode
		* Internally, this call simply performs a sparse product between the matrix Q
		* and a sparse identity matrix. However, due to the fact that the sparse
		* reflectors are stored unsorted, two transpositions are needed to sort
		* them before performing the product.
		*/
		const MatrixQType& matrixQ() const
		{
			return m_Q;
		}

		/**
		* \returns a const reference to the row permutation P that was applied to A such that P * A = Q * R
		* Added for compatibility with other solvers.
		* This solver does not perform any row permutations and so P will always be identity.
		*/
		const PermutationType& rowsPermutation() const {
			//eigen_assert(m_isInitialized && "Decomposition is not initialized.");
			return m_rowPerm;
		}

		/**
		* \returns a flag indicating whether the factorization introduced some row permutations
		* It is determined during the input pattern analysis step.
		*/
		bool hasRowPermutation() const {
			return this->m_hasRowPermutation;
		}

		/** \returns a const reference to the column permutation P that was applied to A such that A*P = Q*R
		* It is the combination of the fill-in reducing permutation and numerical column pivoting.
		*/
		const PermutationType& colsPermutation() const
		{
			eigen_assert(m_isInitialized && "Decomposition is not initialized.");
			return m_outputPerm_c;
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
			if (colsPermutation().size())  dest = colsPermutation() * y.topRows(cols());
			else                  dest = y.topRows(cols());

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
		inline const Solve<BlockDiagonalSparseQR_Ext, Rhs> solve(const MatrixBase<Rhs>& B) const
		{
			eigen_assert(m_isInitialized && "The factorization should be called first, use compute()");
			eigen_assert(this->rows() == B.rows() && "SparseQR::solve() : invalid number of rows in the right hand side matrix");
			return Solve<BlockDiagonalSparseQR_Ext, Rhs>(*this, B.derived());
		}
		template<typename Rhs>
		inline const Solve<BlockDiagonalSparseQR_Ext, Rhs> solve(const SparseMatrixBase<Rhs>& B) const
		{
			eigen_assert(m_isInitialized && "The factorization should be called first, use compute()");
			eigen_assert(this->rows() == B.rows() && "SparseQR::solve() : invalid number of rows in the right hand side matrix");
			return Solve<BlockDiagonalSparseQR_Ext, Rhs>(*this, B.derived());
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
		mutable ComputationInfo m_info;

		MatrixType m_pmat;            // Temporary matrix
		MatrixRType m_R;               // The triangular factor matrix
		MatrixQType m_Q;               // The orthogonal reflectors
		ScalarVector m_hcoeffs;         // The Householder coefficients

		PermutationType m_outputPerm_c; // The final column permutation
		PermutationType m_rowPerm;		// Row permutation matrix, always identity as solver does not perform row permutations

		Index m_nonzeropivots;          // Number of non zero pivots found
		IndexVector m_etree;            // Column elimination tree
		IndexVector m_firstRowElt;      // First element in each row

		bool m_analysisIsok;
		bool m_factorizationIsok;
		bool m_hasRowPermutation;		// Row permutation performed during the factorization

		Index m_numNonZeroQ;

										/*
										* Structures filled during sparse matrix pattern analysis.
										*/
		BlockInfoMap m_blockMap;		// Sparse matrix block information
		BlockInfoMapOrder m_blockOrder; // Sparse matrix block order

        std::string m_lastError;
        bool m_useDefaultThreshold;
        Scalar m_threshold;
        
		void mergeBlocks(BlockInfoMapOrder &blockOrder, BlockInfoMap &blockMap, const StorageIndex maxColStep);
	};

	/*
	* Helper structure holding band information for a single row.
	* Stores original row index (before any row reordering was performed),
	* index of the first nonzero (start) and last nonzero(end) in the band and the band length (length).
	*/
	/*
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
	*/

	/*
	* Helper function going through a block map and looking for a possibility to merge several blocks together
	* in order to obtain better dense block sizes for the YTY representation.
	*/
	const int SuggestedBlockCols = 3;

	template <typename MatrixType, typename BlockQRSolver>
	void BlockDiagonalSparseQR_Ext<MatrixType, BlockQRSolver>::mergeBlocks(BlockInfoMapOrder &blockOrder, BlockInfoMap &blockMap, const StorageIndex maxColStep) {
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

	template <typename MatrixType, typename BlockQRSolver>
	void BlockDiagonalSparseQR_Ext<MatrixType, BlockQRSolver>::setPattern(const StorageIndex matRows, const StorageIndex matCols,
		const StorageIndex blockRows, const StorageIndex blockCols) {
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
		typename MatrixType::StorageIndex numBlocks = n / blockCols;
		this->m_blockMap.clear();
		this->m_blockOrder.clear();
		this->m_numNonZeroQ = 0;
		typename MatrixType::StorageIndex rowIdx = 0;
		typename MatrixType::StorageIndex colIdx = 0;
		for (int i = 0; i < numBlocks; i++) {
			rowIdx = i * blockRows;
			colIdx = i * blockCols;
			this->m_blockOrder.push_back(colIdx);
			this->m_blockMap.insert(std::make_pair(colIdx, MatrixBlockInfo(rowIdx, colIdx, blockRows, blockCols)));
			this->m_numNonZeroQ += blockRows * blockRows;
		}

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
	template <typename MatrixType, typename BlockQRSolver>
	void BlockDiagonalSparseQR_Ext<MatrixType, BlockQRSolver>::analyzePattern(const MatrixType& mat)
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
		BlockBandSize bandWidths, bandHeights;
		RowMajorMatrixType rmMat(mat);
		std::vector<MatrixRowRange> rowRanges;
		for (typename MatrixType::StorageIndex j = 0; j < rmMat.rows(); j++) {
			typename RowMajorMatrixType::InnerIterator rowIt(rmMat, j);
			typename MatrixType::StorageIndex startIdx = rowIt.index();
			typename MatrixType::StorageIndex endIdx = startIdx;
			while (++rowIt) { endIdx = rowIt.index(); }	// FixMe: Is there a better way?
			rowRanges.push_back(MatrixRowRange(j, startIdx, endIdx));

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

		/******************************************************************/
		// 2) Sort the rows to form as-banded-as-possible matrix
		// Set an indicator whether row sorting is needed
		this->m_hasRowPermutation = !std::is_sorted(rowRanges.begin(), rowRanges.end(), [](const MatrixRowRange &lhs, const MatrixRowRange &rhs) {
			return (lhs.start < rhs.start);
		});
		// Perform the actual row sorting if needed
		if (this->m_hasRowPermutation) {
			std::stable_sort(rowRanges.begin(), rowRanges.end(), [](const MatrixRowRange &lhs, const MatrixRowRange &rhs) {
				if (lhs.start < rhs.start) {
					return true;
				}
				else if (lhs.start == rhs.start) {
					return lhs.origIdx < rhs.origIdx;
				}
				else {
					return false;
				}
				//return (lhs.start < rhs.start);
			});
		}

		/******************************************************************/
		// 3) Search for banded blocks (blocks of row sharing same/similar band)		
		typename MatrixType::StorageIndex maxColStep = 0;
		for (typename MatrixType::StorageIndex j = 0; j < rowRanges.size() - 1; j++) {
			if ((rowRanges.at(j + 1).start - rowRanges.at(j).start) > maxColStep) {
				maxColStep = (rowRanges.at(j + 1).start - rowRanges.at(j).start);
			}
		}

		// And record the estimated block structure
		this->m_numNonZeroQ = 0;
		this->m_blockMap.clear();
		this->m_blockOrder.clear();
		Eigen::Matrix<typename MatrixType::StorageIndex, Dynamic, 1> permIndices(rowRanges.size());
		typename MatrixType::StorageIndex rowIdx = 0;
		for (auto it = rowRanges.begin(); it != rowRanges.end(); ++it, rowIdx++) {
			permIndices(it->origIdx) = rowIdx;

			// std::find is terribly slow for large arrays
			// assuming m_blockOrder is ordered, we can use binary_search
			// is m_blockOrder always ordered? can we always use binary_search???
			if (!std::binary_search(this->m_blockOrder.begin(), this->m_blockOrder.end(), it->start)) {
				//if (std::find(this->m_blockOrder.begin(), this->m_blockOrder.end(), it->start) == this->m_blockOrder.end()) {
				this->m_blockOrder.push_back(it->start);
				this->m_blockMap.insert(std::make_pair(it->start, MatrixBlockInfo(rowIdx, it->start, bandHeights.at(it->start), bandWidths.at(it->start))));
				this->m_numNonZeroQ += this->m_blockMap.at(it->start).numRows * this->m_blockMap.at(it->start).numRows;
			}
		}
		// Create row permutation matrix that achieves the desired row reordering
		this->m_rowPerm = PermutationType(permIndices);

		/******************************************************************/
		// 4) Go through the estimated block structure
		// And merge several blocks together if needed/possible in order to form reasonably big banded blocks
		this->mergeBlocks(this->m_blockOrder, this->m_blockMap, maxColStep);

		/******************************************************************/
		// 5) Finalize
		m_R.resize(mat.rows(), mat.cols());
		
		m_analysisIsok = true;
	}

	/** \brief Preprocessing step of a QR factorization
	*
	* \warning The matrix \a mat must be in compressed mode (see SparseMatrix::makeCompressed()).
	*
	* In this step, the fill-reducing permutation is computed and applied to the columns of A
	* and the column elimination tree is computed as well. Only the sparsity pattern of \a mat is exploited.
	*
	* \note In this step it is assumed that there is no empty row in the matrix \a mat.
	*/
	/*
	template <typename MatrixType, typename BlockQRSolver>
	void BlockDiagonalSparseQR_Ext<MatrixType, BlockQRSolver>::analyzePattern(const MatrixType& mat)
	{
		eigen_assert(mat.isCompressed() && "SparseQR requires a sparse matrix in compressed mode. Call .makeCompressed() before passing it to SparseQR");

		/// Check block structure is valid
		eigen_assert(mat.rows() % m_blocksRows == mat.cols() % m_blocksCols && mat.rows() / m_blocksRows == mat.cols() / m_blocksCols && mat.cols() % m_blocksCols == 0);

		StorageIndex n = mat.cols();

		m_outputPerm_c.resize(n);
		m_outputPerm_c.indices().setLinSpaced(n, 0, n - 1);

		StorageIndex m = mat.rows();

		m_rowPerm.resize(m);
		m_rowPerm.indices().setLinSpaced(m, 0, m - 1);

		assert(_CrtCheckMemory());
	}
	*/

	/** \brief Performs the numerical QR factorization of the input matrix
	*
	* The function SparseQR::analyzePattern(const MatrixType&) must have been called beforehand with
	* a matrix having the same sparsity pattern than \a mat.
	*
	* \param mat The sparse column-major matrix
	*/
	template <typename MatrixType, typename BlockQRSolver>
	void BlockDiagonalSparseQR_Ext<MatrixType, BlockQRSolver>::factorize(const MatrixType& mat)
	{
		// Not rank-revealing, column permutation is identity
		m_outputPerm_c.setIdentity(mat.cols());

		// Permute the input matrix using the precomputed row permutation
		m_pmat = (this->m_rowPerm * mat);

		typedef typename MatrixType::Index Index;

		Index numBlocks = this->m_blockOrder.size();
		Index rank = 0;
		MatrixBlockInfo bi = this->m_blockMap.at(this->m_blockOrder.at(0));

		// Q is rows x rows, R is rows x cols
		Eigen::TripletArray<Scalar> tripletsR(numBlocks * bi.numRows * bi.numCols);
		m_Q.resize(m_pmat.rows(), m_pmat.rows());
		m_Q.reserve(this->m_numNonZeroQ);

		Index m1 = 0;
		Index N_start = m_pmat.cols();
		for (int i = 0; i < numBlocks; i++) {
			bi = this->m_blockMap.at(this->m_blockOrder.at(i));

			// Copy current block
			BlockMatrixType block_i = m_pmat.block(bi.rowIdx, bi.colIdx, bi.numRows, bi.numCols);
		
			// Perform QR
			BlockQRSolver blockSolver;
			blockSolver.compute(block_i);
			rank += blockSolver.rank();

			BlockMatrixType Qi = blockSolver.matrixQ();
			typename BlockQRSolver::MatrixRType Ri = blockSolver.matrixR();

			auto base_row = bi.rowIdx;
			auto base_col = bi.colIdx;

			// Assemble into final Q
			if (bi.numRows >= bi.numCols) {
				// each rectangular Qi is partitioned into [U N] where U is rxc and N is rx(r-c)
				// All the Us are gathered in the leftmost nc columns of Q, all Ns to the right
				
				// Q
				///*
				auto curr_m1 = (bi.numRows - bi.numCols);
				for (Index j = 0; j < bi.numRows; j++) {
					assert(base_row + j < m_Q.rows());
					m_Q.startVec(base_row + j);
					// Us
					for (Index k = 0; k < bi.numCols; k++) {
						assert(base_col + k < m_Q.cols());
						m_Q.insertBack(base_row + j, base_col + k) = Qi.coeff(j, k);
					}
					// Ns
					for (Index k = 0; k < curr_m1; k++) {
						assert(N_start + m1 + k < m_Q.cols());
						m_Q.insertBack(base_row + j, N_start + m1 + k) = Qi.coeff(j, bi.numCols + k);
					}
					//
				}
				m1 += curr_m1;

				// R
				// Only the top cxc of R is nonzero, so c rows at a time
				for (Index j = 0; j < bi.numCols; j++) {
					for (Index k = j; k < bi.numCols; k++) {
						tripletsR.add(base_col + j, base_col + k, Ri.coeff(j, k));
					}
				}
				//*/
				// Q diag
				/*
				auto curr_m1 = (bi.numRows - bi.numCols);
				for (Index j = 0; j < bi.numRows; j++) {
					assert(base_row + j < m_Q.rows());
					m_Q.startVec(base_row + j);
					// Us
					//for (Index k = 0; k < bi.numCols; k++) {
					for (Index k = 0; k < bi.numRows; k++) {
						assert(base_col + k < m_Q.cols());
						m_Q.insertBack(base_row + j, base_row + k) = Qi.coeff(j, k);
						//	m_Q.insertBack(base_row + j, base_col + k) = Qi.coeff(j, k);
					}
					//
				}
				m1 += curr_m1;
				
				// R
				// Only the top cxc of R is nonzero, so c rows at a time
				for (Index j = 0; j < bi.numCols; j++) {
					for (Index k = j; k < bi.numCols; k++) {
						//tripletsR.add(base_col + j, base_col + k, Ri.coeff(j, k));
						tripletsR.add(base_row + j, base_col + k, Ri.coeff(j, k));
					}
				}
				//*/
				
			} else {
				// Just concatenate everything -- it's upper triangular anyway (although not rank-revealing... xxfixme with colperm?)
				// xx and indeed for landscape, don't even need to compute QR after we've done the leftmost #rows columns

				assert(false);
			}

			// fill cols permutation
			for (Index j = 0; j < bi.numCols; j++) {
				m_outputPerm_c.indices()(base_col + j, 0) = base_col + blockSolver.colsPermutation().indices()(j, 0);
			}
		}

		// Now build Q and R from Qs and Rs of each block
		m_Q.finalize();

		m_R.resize(m_pmat.rows(), m_pmat.cols());
		m_R.setZero();
		m_R.setFromTriplets(tripletsR.begin(), tripletsR.end());
		m_R.makeCompressed();

		m_nonzeropivots = rank;
		m_isInitialized = true;
		m_info = Success;
		m_factorizationIsok = true;
	}

} // end namespace Eigen

#endif
