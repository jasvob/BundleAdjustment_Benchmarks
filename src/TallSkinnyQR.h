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

#ifndef TALL_SKINNY_QR_H
#define TALL_SKINNY_QR_H

namespace Eigen {


	/**
	* \ingroup SparseQR_Module
	* \class TallSkinnyQR
	* \brief QR factorization of block-diagonal matrix
	*
	* \implsparsesolverconcept
	*
	*/
	template<typename _MatrixType, typename _BlockQRSolver>
	class TallSkinnyQR : public SparseSolverBase<TallSkinnyQR<_MatrixType, _BlockQRSolver> >
	{
	protected:
		typedef SparseSolverBase<TallSkinnyQR<_MatrixType, _BlockQRSolver> > Base;
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
		TallSkinnyQR() : m_analysisIsok(false), m_factorizationIsok(false), m_hasRowPermutation(false)
		{ }

		/** Construct a QR factorization of the matrix \a mat.
		*
		* \warning The matrix \a mat must be in compressed mode (see SparseMatrix::makeCompressed()).
		*
		* \sa compute()
		*/
		explicit TallSkinnyQR(const MatrixType& mat) : m_analysisIsok(false), m_factorizationIsok(false), m_hasRowPermutation(false)
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
			// Recursive computation
			// If m > 6 * n, call this method recursively on upper and lower part
			Index n = mat.cols();
			Index m = mat.rows();

			Index blockHeight = 2 * n;
			Index numBlocks = std::ceil(m / double(blockHeight));
			if (numBlocks > 2) {

				const size_t nloop = numBlocks;
				const size_t nthreads = 6;// std::thread::hardware_concurrency();
				{
					std::vector<std::thread> threads(nthreads);
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
									Index startIdx = j * blockHeight;
									this->compute(mat.block(startIdx, 0, blockHeight, n), forcePatternAlaysis);
								}
							}
						}, t*nloop / nthreads, (t + 1) == nthreads ? nloop : (t + 1)*nloop / nthreads, t));
					}
					std::for_each(threads.begin(), threads.end(), [](std::thread& x) {x.join(); });
				}

				/*
				for(int i = 0; i < numBlocks; i++) {
					Index startIdx = i * blockHeight;
					this->compute(mat.block(startIdx, 0, blockHeight, n), forcePatternAlaysis);
				}*/
			} else {
				// We have 2 blocks with desired size -> analyze & factorize
				analyzePattern(mat);

				// !!! Reset variables before the factorization !!!
				m_isInitialized = false;
				m_factorizationIsok = false;
				factorize(mat);
			}
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
		MatrixQType matrixQ() const
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
		inline const Solve<TallSkinnyQR, Rhs> solve(const MatrixBase<Rhs>& B) const
		{
			eigen_assert(m_isInitialized && "The factorization should be called first, use compute()");
			eigen_assert(this->rows() == B.rows() && "SparseQR::solve() : invalid number of rows in the right hand side matrix");
			return Solve<TallSkinnyQR, Rhs>(*this, B.derived());
		}
		template<typename Rhs>
		inline const Solve<TallSkinnyQR, Rhs> solve(const SparseMatrixBase<Rhs>& B) const
		{
			eigen_assert(m_isInitialized && "The factorization should be called first, use compute()");
			eigen_assert(this->rows() == B.rows() && "SparseQR::solve() : invalid number of rows in the right hand side matrix");
			return Solve<TallSkinnyQR, Rhs>(*this, B.derived());
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
		
		Matrix<Scalar, Dynamic, Dynamic> m_R;
		//MatrixRType m_R;               // The triangular factor matrix
		//MatrixQType m_Q;               // The orthogonal reflectors
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
		MatrixBlockInfo blockUpper;
		MatrixBlockInfo blockLower;
		PermutationType m_blockReorder;

		template <typename, typename > friend struct SparseQR_QProduct;
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
	void TallSkinnyQR<MatrixType, BlockQRSolver>::analyzePattern(const MatrixType& mat)
	{
		typedef RowRange<MatrixType::StorageIndex> MatrixRowRange;
		typedef std::map<MatrixType::StorageIndex, MatrixType::StorageIndex> BlockBandSize;
		typedef SparseMatrix<Scalar, RowMajor, MatrixType::StorageIndex> RowMajorMatrixType;

		Index n = mat.cols();
		Index m = mat.rows();
		Index diagSize = (std::min)(m, n);

		// 1) Divide the tall skinny matrix into subblocks of 2n x n and prepare merging hierarchy
		// Separate input into U and L block
		Index bandHeight = m / 2;
		// Upper block
		this->blockUpper = MatrixBlockInfo(0, 0, bandHeight, n);
		// Lower block
		this->blockLower = MatrixBlockInfo(bandHeight, 0, m - bandHeight, n);

		// Compose reordering row permutation for merging the results of the two blocks
		Eigen::Matrix<MatrixType::StorageIndex, Dynamic, 1> reorderIndices(m);
		Index commonBlockRows = (this->blockUpper.numRows > this->blockLower.numRows) ? this->blockLower.numRows : this->blockUpper.numRows;
		for (int r = 0; r < commonBlockRows; r++) {
			reorderIndices(r * 2) = r;
			reorderIndices(r * 2 + 1) = r + commonBlockRows;
		}
		for (int r = commonBlockRows; r < m; r++) {
			reorderIndices(r) = r;
		}
		this->m_blockReorder = PermutationType(reorderIndices);

		// Create row permutation matrix that achieves the desired row reordering
		this->m_rowPerm = this->m_blockReorder;
		//this->m_rowPerm = PermutationType(permIndices);

		/******************************************************************/
		// 4) Go through the estimated block structure
		// And merge several blocks together if needed/possible in order to form reasonably big banded blocks
		//this->mergeBlocks(this->m_blockOrder, this->m_blockMap, maxColStep);

		/******************************************************************/
		// 5) Finalize
		//m_R.resize(mat.rows(), mat.cols());
/*	
		std::cout << "Matrix size: " << m << " x " << n << std::endl;
		std::cout << "No of blocks: " << double(m) / (2 * n) << std::endl;
		std::cout << "Upper block: (" << this->blockUpper.rowIdx << ", " << this->blockUpper.colIdx << "): " << this->blockUpper.numRows << " x " << this->blockUpper.numCols << std::endl;
		std::cout << "Lower block: (" << this->blockLower.rowIdx << ", " << this->blockLower.colIdx << "): " << this->blockLower.numRows << " x " << this->blockLower.numCols << std::endl;
	*/	

		m_analysisIsok = true;
	}

	/** \brief Performs the numerical QR factorization of the input matrix
	*
	* The function SparseQR::analyzePattern(const MatrixType&) must have been called beforehand with
	* a matrix having the same sparsity pattern than \a mat.
	*
	* \param mat The sparse column-major matrix
	*/
	template <typename MatrixType, typename BlockQRSolver>
	void TallSkinnyQR<MatrixType, BlockQRSolver>::factorize(const MatrixType& mat)
	{
		// Not rank-revealing, column permutation is identity
		m_outputPerm_c.setIdentity(mat.cols());

		m_pmat = mat;

//		clock_t begin = clock();
		// Prepare the solver
		BlockQRSolver blockSolverU, blockSolverL;

		// Factorize upper block
		BlockMatrixType blockU = m_pmat.block(this->blockUpper.rowIdx, this->blockUpper.colIdx, this->blockUpper.numRows, this->blockUpper.numCols);
		blockSolverU.compute(blockU);

		// Factorize lower block
		BlockMatrixType blockL = m_pmat.block(this->blockLower.rowIdx, this->blockLower.colIdx, this->blockLower.numRows, this->blockLower.numCols);
		blockSolverL.compute(blockL);

		// Compose output Q and R
		m_R.resize(mat.rows(), mat.cols());
		m_R.topRows(this->blockUpper.numRows) = blockSolverU.matrixQR().template triangularView<Upper>();
		m_R.bottomRows(this->blockLower.numRows) = blockSolverL.matrixQR().template triangularView<Upper>();
//		std::cout << "TSQR compute 1 block:   " << double(clock() - begin) / CLOCKS_PER_SEC << "s\n";

		// Make one Householder vector out of the two

		

		/*
		// Permute the input matrix using the precomputed row permutation
		m_pmat = (this->m_rowPerm * mat);

		typedef MatrixType::Index Index;

		Index numBlocks = this->m_blockOrder.size();
		Index rank = 0;
		MatrixBlockInfo bi = this->m_blockMap.at(this->m_blockOrder.at(0));

		// Q is rows x rows, R is rows x cols
		Eigen::TripletArray<Scalar> tripletsR(numBlocks * bi.numRows * bi.numCols);
		m_Q.resize(m_pmat.rows(), m_pmat.rows());
		m_Q.reserve(this->m_numNonZeroQ);
		std::cout << this->m_numNonZeroQ << std::endl;

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
				}
				m1 += curr_m1;

				// R
				// Only the top cxc of R is nonzero, so c rows at a time
				for (Index j = 0; j < bi.numCols; j++) {
					for (Index k = j; k < bi.numCols; k++) {
						tripletsR.add(base_col + j, base_col + k, Ri.coeff(j, k));
					}
				}
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
		*/
	}


} // end namespace Eigen

#endif
