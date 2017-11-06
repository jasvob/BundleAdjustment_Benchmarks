// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2017 Jan Svoboda <jan.svoboda@usi.ch>
// Copyright (C) 2016 Andrew Fitzgibbon <awf@microsoft.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_DENSE_BLOCKED_SPARSE_QR_H
#define EIGEN_DENSE_BLOCKED_SPARSE_QR_H

#include <ctime>
#include <typeinfo>
#include <shared_mutex>
#include "SparseBlockCOO_Ext2.h"
//#include "unsupported/Eigen/src/SparseQRExtra/SparseBlockCOO.h"
#include "unsupported/Eigen/src/SparseQRExtra/eigen_extras.h"

namespace Eigen {

	template<typename MatrixType, typename OrderingType, int SuggestedBlockCols, bool MultiThreading> class DenseBlockedSparseQR;
	template<typename DenseBlockedSparseQRType> struct DenseBlockedSparseQRMatrixQReturnType;
	template<typename DenseBlockedSparseQRType> struct DenseBlockedSparseQRMatrixQTransposeReturnType;
	template<typename DenseBlockedSparseQRType, typename Derived> struct DenseBlockedSparseQR_QProduct;
	namespace internal {

		// traits<DenseBlockedSparseQRMatrixQ[Transpose]>
		template <typename DenseBlockedSparseQRType> struct traits<DenseBlockedSparseQRMatrixQReturnType<DenseBlockedSparseQRType> >
		{
			typedef typename DenseBlockedSparseQRType::MatrixType ReturnType;
			typedef typename ReturnType::StorageIndex StorageIndex;
			typedef typename ReturnType::StorageKind StorageKind;
			enum {
				RowsAtCompileTime = Dynamic,
				ColsAtCompileTime = Dynamic
			};
		};

		template <typename DenseBlockedSparseQRType> struct traits<DenseBlockedSparseQRMatrixQTransposeReturnType<DenseBlockedSparseQRType> >
		{
			typedef typename DenseBlockedSparseQRType::MatrixType ReturnType;
		};

		template <typename DenseBlockedSparseQRType, typename Derived> struct traits<DenseBlockedSparseQR_QProduct<DenseBlockedSparseQRType, Derived> >
		{
			typedef typename Derived::PlainObject ReturnType;
		};

		// DenseBlockedSparseQR_traits
		template <typename T> struct DenseBlockedSparseQR_traits {  };
		template <class T, int Rows, int Cols, int Options> struct DenseBlockedSparseQR_traits<Matrix<T, Rows, Cols, Options>> {
			typedef Matrix<T, Rows, 1, Options> Vector;
		};
		template <class Scalar, int Options, typename Index> struct DenseBlockedSparseQR_traits<SparseMatrix<Scalar, Options, Index>> {
			typedef SparseVector<Scalar, Options> Vector;
		};
	} // End namespace internal

	  /**
	  * \ingroup DenseBlockedSparseQR_Module
	  * \class DenseBlockedSparseQR
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
	class DenseBlockedSparseQR : public SparseSolverBase<DenseBlockedSparseQR<_MatrixType, _OrderingType, _SuggestedBlockCols, _MultiThreading> >
	{
	protected:
		typedef SparseSolverBase<DenseBlockedSparseQR<_MatrixType, _OrderingType, _SuggestedBlockCols, _MultiThreading> > Base;
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

		typedef DenseBlockedSparseQRMatrixQReturnType<DenseBlockedSparseQR> MatrixQType;
		typedef SparseMatrix<Scalar, ColMajor, StorageIndex> MatrixRType;
		typedef PermutationMatrix<Dynamic, Dynamic, StorageIndex> PermutationType;

		/*
		* Stores information about a dense block in a block sparse matrix.
		* Holds the position of the block (row index, column index) and its size (number of rows, number of columns).
		*/
		template <typename IndexType>
		struct BlockInfo {
			IndexType idxDiagRow;
			IndexType idxDenseRow;
			IndexType numDiagRows;
			IndexType numDenseRows;
			IndexType numCols;

			BlockInfo()
				: idxDiagRow(0), idxDenseRow(0), numDiagRows(0), numDenseRows(0), numCols(0) {
			}

			BlockInfo(const IndexType &diagRi, const IndexType &denseRi, const IndexType &diagNr, const IndexType &denseNr, const IndexType &nc)
				: idxDiagRow(diagRi), idxDenseRow(denseRi), numDiagRows(diagNr), numDenseRows(denseNr), numCols(nc) {
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
		DenseBlockedSparseQR() : m_analysisIsok(false), m_lastError(""), m_useDefaultThreshold(true), m_useMultiThreading(_MultiThreading), m_hasRowPermutation(false)
		{ }

		/** Construct a QR factorization of the matrix \a mat.
		*
		* \warning The matrix \a mat must be in compressed mode (see SparseMatrix::makeCompressed()).
		*
		* \sa compute()
		*/
		explicit DenseBlockedSparseQR(const MatrixType& mat) : m_analysisIsok(false), m_lastError(""), m_useDefaultThreshold(true), m_useMultiThreading(_MultiThreading), m_hasRowPermutation(false)
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
		* Q = DenseBlockedSparseQR<SparseMatrix<double> >(A).matrixQ();
		* \endcode
		* Internally, this call simply performs a sparse product between the matrix Q
		* and a sparse identity matrix. However, due to the fact that the sparse
		* reflectors are stored unsorted, two transpositions are needed to sort
		* them before performing the product.
		*/
		DenseBlockedSparseQRMatrixQReturnType<DenseBlockedSparseQR> matrixQ() const
		{
			return DenseBlockedSparseQRMatrixQReturnType<DenseBlockedSparseQR>(*this);
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
			eigen_assert(this->rows() == B.rows() && "DenseBlockedSparseQR::solve() : invalid number of rows in the right hand side matrix");

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
		inline const Solve<DenseBlockedSparseQR, Rhs> solve(const MatrixBase<Rhs>& B) const
		{
			eigen_assert(m_isInitialized && "The factorization should be called first, use compute()");
			eigen_assert(this->rows() == B.rows() && "DenseBlockedSparseQR::solve() : invalid number of rows in the right hand side matrix");
			return Solve<DenseBlockedSparseQR, Rhs>(*this, B.derived());
		}
		template<typename Rhs>
		inline const Solve<DenseBlockedSparseQR, Rhs> solve(const SparseMatrixBase<Rhs>& B) const
		{
			eigen_assert(m_isInitialized && "The factorization should be called first, use compute()");
			eigen_assert(this->rows() == B.rows() && "DenseBlockedSparseQR::solve() : invalid number of rows in the right hand side matrix");
			return Solve<DenseBlockedSparseQR, Rhs>(*this, B.derived());
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
		typedef SparseBlockCOO_Ext2<BlockYTY_Ext2<Scalar, StorageIndex>, StorageIndex> SparseBlockYTY;
		typedef Matrix<Scalar, Dynamic, Dynamic> DenseMatrixType;

		bool m_analysisIsok;
		bool m_factorizationIsok;
		mutable ComputationInfo m_info;
		std::string m_lastError;
		DenseMatrixType  m_pmat;
		//MatrixQStorageType m_pmat;            // Temporary matrix
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

		template <typename, typename > friend struct DenseBlockedSparseQR_QProduct;

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
    
//    template <typename MatrixType, typename OrderingType, int SuggestedBlockCols, bool MultiThreading>
//    void DenseBlockedSparseQR<MatrixType, OrderingType, SuggestedBlockCols, MultiThreading>::analyzePattern(const MatrixType& mat)
//    {
//        typedef RowRange<typename MatrixType::StorageIndex> MatrixRowRange;
//        typedef std::map<typename MatrixType::StorageIndex, typename MatrixType::StorageIndex> BlockBandSize;
//        typedef SparseMatrix<Scalar, RowMajor, typename MatrixType::StorageIndex> RowMajorMatrixType;
//        
//        Index n = mat.cols();
//        Index m = mat.rows();
//        Index diagSize = (std::min)(m, n);
//        
//        // Looking for as-banded-as-possible structure in the matrix
//        /******************************************************************/
//        // 1) Compute and store band information for each row in the matrix
//        //BlockBandSize bandWidths, bandHeights;
//        RowMajorMatrixType rmMat(mat);
//        std::vector<MatrixRowRange> rowRanges;
//        for (typename MatrixType::StorageIndex j = 0; j < rmMat.rows(); j++) {
//            typename RowMajorMatrixType::InnerIterator rowIt(rmMat, j);
//            typename MatrixType::StorageIndex startIdx = rowIt.index();
//            typename MatrixType::StorageIndex endIdx = startIdx;
//            while (++rowIt) { endIdx = rowIt.index(); }	// FixMe: Is there a better way?
//            rowRanges.push_back(MatrixRowRange(j, startIdx, endIdx));
//        }
//        
//        /******************************************************************/
//        // 2) Sort the rows to form as-banded-as-possible matrix
//        // Set an indicator whether row sorting is needed
//        this->m_hasRowPermutation = !std::is_sorted(rowRanges.begin(), rowRanges.end(), [](const MatrixRowRange &lhs, const MatrixRowRange &rhs) {
//            return (lhs.start < rhs.start);
//        });
//        // Perform the actual row sorting if needed
//        if (this->m_hasRowPermutation) {
//            std::stable_sort(rowRanges.begin(), rowRanges.end(), [](const MatrixRowRange &lhs, const MatrixRowRange &rhs) {
//                return (lhs.length < rhs.length);//(lhs.start < rhs.start);
//            });
//            std::stable_sort(rowRanges.begin(), rowRanges.end(), [](const MatrixRowRange &lhs, const MatrixRowRange &rhs) {
//                if (lhs.length <= 2 * rhs.length) {
//                    return lhs.start < rhs.start;//(lhs.start < rhs.start);
//                }
//                else {
//                    return false;
//                }
//            });
//        }
//        
//        /******************************************************************/
//        // 3) Search for column blocks (blocks of columns starting at the same row)
//        // And record the estimated block structure
//        this->m_blockMap.clear();
//        this->m_blockOrder.clear();
//        Eigen::Matrix<typename MatrixType::StorageIndex, Dynamic, 1> permIndices(rowRanges.size());
//        typename MatrixType::StorageIndex rowIdx = 0;
//        typename MatrixType::StorageIndex lastStart = rowRanges.at(0).start;
//        bool skipDense = false;
//        for (auto it = rowRanges.begin(); it != rowRanges.end(); ++it, rowIdx++) {
//            permIndices(it->origIdx) = rowIdx;
//            
//            if(!skipDense) {
//                if (lastStart > it->start) {
//                    std::cout << "Breaking at row " << rowIdx << " ..." << std::endl;
//                    // Now we know where the dense block will start
//                    this->m_denseStartRow = rowIdx;
//                    this->m_denseNumRows = mat.rows() - this->m_denseStartRow;
//                    std::cout << this->m_denseStartRow << ", " << this->m_denseNumRows << std::endl;
//                    skipDense = true;
//                }
//                lastStart = it->start;
//                
//                // std::find is terribly slow for large arrays
//                // assuming m_blockOrder is ordered, we can use binary_search
//                // is m_blockOrder always ordered? can we always use binary_search???
//                //if (!std::binary_search(this->m_blockOrder.begin(), this->m_blockOrder.end(), it->start)) {
//                //if (std::find(this->m_blockOrder.begin(), this->m_blockOrder.end(), it->start) == this->m_blockOrder.end()) {
//                //	this->m_blockOrder.push_back(it->start);
//                //	this->m_blockMap.insert(std::make_pair(it->start, MatrixBlockInfo(rowIdx, it->start, bandHeights.at(it->start), it->length)));
//                //}
//            }
//        }
//        // Create row permutation matrix that achieves the desired row reordering
//        this->m_rowPerm = PermutationType(permIndices);
//        
//        // Now apply the row permutation
//        m_pmat = this->m_rowPerm * mat;
//        
//        // Now go through the sparse part of the block by columns and finally detect block sizes...
//        // 1) Compute and store band information for each row in the matrix
//        BlockBandSize bandWidths, bandHeights;
//        MatrixType spBlock = m_pmat.block(0, 0, this->m_denseStartRow, m_pmat.cols());
//        std::vector<MatrixRowRange> colRanges;
//        for (typename MatrixType::StorageIndex j = 0; j < spBlock.cols(); j++) {
//            typename MatrixType::InnerIterator colIt(spBlock, j);
//            typename MatrixType::StorageIndex startIdx = colIt.index();
//            typename MatrixType::StorageIndex endIdx = startIdx;
//            while (++colIt) { endIdx = colIt.index(); }	// FixMe: Is there a better way?
//            colRanges.push_back(MatrixRowRange(j, startIdx, endIdx));
//            
//            typename MatrixType::StorageIndex bw = endIdx - startIdx + 1;
//            if (bandWidths.find(startIdx) == bandWidths.end()) {
//                bandWidths.insert(std::make_pair(startIdx, bw));
//            }
//            else {
//                if (bandWidths.at(startIdx) < bw) {
//                    bandWidths.at(startIdx) = bw;
//                }
//            }
//            
//            if (bandHeights.find(startIdx) == bandHeights.end()) {
//                bandHeights.insert(std::make_pair(startIdx, 1));
//            }
//            else {
//                bandHeights.at(startIdx) += 1;
//            }
//        }
//        
//        // Compose block map
//        this->m_blockMap.clear();
//        this->m_blockOrder.clear();
//        typename MatrixType::StorageIndex colIdx = 0;
//        for (auto it = colRanges.begin(); it != colRanges.end(); ++it, colIdx++) {
//            // std::find is terribly slow for large arrays
//            // assuming m_blockOrder is ordered, we can use binary_search
//            // is m_blockOrder always ordered? can we always use binary_search???
//            if (!std::binary_search(this->m_blockOrder.begin(), this->m_blockOrder.end(), it->start)) {
//                this->m_blockOrder.push_back(it->start);
//                this->m_blockMap.insert(std::make_pair(it->start, MatrixBlockInfo(it->start, colIdx, it->length, bandHeights.at(it->start)))); // Switch numRows and numCols (analysis was done by columns)
//            }
//        }
//        
//        /******************************************************************/
//        // 4) Go through the estimated block structure
//        // And merge several blocks together if needed/possible in order to form reasonably big banded blocks
//        //this->mergeBlocks(this->m_blockOrder, this->m_blockMap, maxColStep);
//        
//        /******************************************************************/
//        // 5) Finalize
//        m_R.resize(mat.rows(), mat.cols());
//        
//        
//        std::cout << "No of blocks: " << this->m_blockOrder.size() << std::endl;
//        if (this->m_blockOrder.size() < 3000) {
//            for (int i = 0; i < this->m_blockOrder.size(); i++) {
//                std::cout << this->m_blockMap.at(this->m_blockOrder.at(i)).rowIdx << ", " << this->m_blockMap.at(this->m_blockOrder.at(i)).colIdx << ": " << this->m_blockMap.at(this->m_blockOrder.at(i)).numRows << ", " << this->m_blockMap.at(this->m_blockOrder.at(i)).numCols << " || ";
//            }
//            std::cout << std::endl;
//        }
//        
//        m_analysisIsok = true;
//    }
    
	template <typename MatrixType, typename OrderingType, int SuggestedBlockCols, bool MultiThreading>
	void DenseBlockedSparseQR<MatrixType, OrderingType, SuggestedBlockCols, MultiThreading>::analyzePattern(const MatrixType& mat)
	{
		typedef RowRange<typename MatrixType::StorageIndex> MatrixRowRange;
		typedef std::map<typename MatrixType::StorageIndex, typename MatrixType::StorageIndex> BlockBandSize;
		typedef SparseMatrix<Scalar, RowMajor, typename MatrixType::StorageIndex> RowMajorMatrixType;

		Index n = mat.cols();
		Index m = mat.rows();
		Index diagSize = (std::min)(m, n);

		Index numBlocks = mat.cols() / SuggestedBlockCols;

		Index denseNumRows = mat.rows() - this->m_denseStartRow;

		/******************************************************************/
		// 1) Recording matrix block structure we will want to use
		this->m_blockMap.clear();
		this->m_blockOrder.clear();
		for (int i = 0; i < numBlocks; i++) {
			Index diagIdx = i * SuggestedBlockCols;
			this->m_blockOrder.push_back(diagIdx);
			this->m_blockMap.insert(std::make_pair(diagIdx, MatrixBlockInfo(diagIdx, this->m_denseStartRow, SuggestedBlockCols, denseNumRows, SuggestedBlockCols)));
		}

		/******************************************************************/
		// 5) Finalize
		m_R.resize(mat.rows(), mat.cols());


		/*std::cout << "No of blocks: " << this->m_blockOrder.size() << std::endl;
		if (this->m_blockOrder.size() < 3000) {
			for (int i = 0; i < this->m_blockOrder.size(); i++) {
				std::cout << this->m_blockMap.at(this->m_blockOrder.at(i)).idxDiagRow << ", " << this->m_blockMap.at(this->m_blockOrder.at(i)).idxDiagRow << ": " << this->m_blockMap.at(this->m_blockOrder.at(i)).numDiagRows << ", " << this->m_blockMap.at(this->m_blockOrder.at(i)).numCols << " || ";
			}
			std::cout << std::endl;
		}*/

		m_analysisIsok = true;
	}

	template <typename MatrixType, typename OrderingType, int SuggestedBlockCols, bool MultiThreading>
	void DenseBlockedSparseQR<MatrixType, OrderingType, SuggestedBlockCols, MultiThreading>::updateMat(const Index &fromIdx, const Index &toIdx, DenseMatrixType &mat, const Index &blockK) {
		// Now update the unsolved rest of m_pmat
		typename SparseBlockYTY::Element blockYTY = this->m_blocksYT[this->m_blocksYT.size() - 1];

		if (blockK < 0) {
			/*
			const size_t nloop = toIdx - fromIdx;
			const size_t nthreads = 4;// std::thread::hardware_concurrency();
			{
			std::vector<std::thread> threads(nthreads);
			std::mutex critical;
			for (int t = 0; t < nthreads; t++)
			{
			threads[t] = std::thread(std::bind(
			[&](const int bi, const int ei, const int t)
			{
			VectorXd resColJd;
			VectorXd tmpResColJ;

			// loop over all items
			for (int j = bi; j < ei; j++)
			{
			// inner loop
			{
			// Use temporary vector resColJ inside of the for loop - faster access
			resColJd = mat.col(fromIdx + j);// .toDense();
			for (Index k = 0; k < this->m_blocksYT.size(); k++) {
			typename MatrixType::StorageIndex subdiagElems = this->m_blocksYT[k].value.rows() - this->m_blocksYT[k].value.cols();
			FULL_TO_BLOCK_VEC_EXT(resColJd, tmpResColJ, this->m_blocksYT[k].value.rows() - this->m_blocksYT[k].value.numDenseRows(), this->m_blocksYT[k].value.cols(), this->m_blocksYT[k].row, this->m_blocksYT[k].value.numZeros(), subdiagElems,
			mat.rows() - this->m_blocksYT[k].value.numDenseRows(), this->m_blocksYT[k].value.numDenseRows())

			// We can afford noalias() in this case
			tmpResColJ += this->m_blocksYT[k].value.multTransposed(tmpResColJ);

			BLOCK_VEC_TO_FULL_EXT(resColJd, tmpResColJ, this->m_blocksYT[k].value.rows() - this->m_blocksYT[k].value.numDenseRows(), this->m_blocksYT[k].value.cols(), this->m_blocksYT[k].row, this->m_blocksYT[k].value.numZeros(), subdiagElems,
			mat.rows() - this->m_blocksYT[k].value.numDenseRows(), this->m_blocksYT[k].value.numDenseRows())
			}

			//std::lock_guard<std::mutex> lock(critical);
			// Write the result back to ci-th column of res
			mat.col(fromIdx + j) = resColJd;// .sparseView();

			}
			}
			}, t*nloop / nthreads, (t + 1) == nthreads ? nloop : (t + 1)*nloop / nthreads, t));
			}
			std::for_each(threads.begin(), threads.end(), [](std::thread& x) {x.join(); });
			}
			//*/
			///*
			VectorXd resColJd;
			VectorXd tmpResColJ;
			for (int ci = fromIdx; ci < toIdx; ci++) {
				// Use temporary vector resColJ inside of the for loop - faster access
				resColJd = mat.col(ci);// .toDense();
				for (Index k = 0; k < this->m_blocksYT.size(); k++) {
					typename MatrixType::StorageIndex subdiagElems = this->m_blocksYT[k].value.rows() - this->m_blocksYT[k].value.cols();
					FULL_TO_BLOCK_VEC_EXT(resColJd, tmpResColJ, this->m_blocksYT[k].value.rows() - this->m_blocksYT[k].value.numDenseRows(), this->m_blocksYT[k].value.cols(), this->m_blocksYT[k].row, this->m_blocksYT[k].value.numZeros(), subdiagElems,
						mat.rows() - this->m_blocksYT[k].value.numDenseRows(), this->m_blocksYT[k].value.numDenseRows())

						// We can afford noalias() in this case
						tmpResColJ += this->m_blocksYT[k].value.multTransposed(tmpResColJ);

					BLOCK_VEC_TO_FULL_EXT(resColJd, tmpResColJ, this->m_blocksYT[k].value.rows() - this->m_blocksYT[k].value.numDenseRows(), this->m_blocksYT[k].value.cols(), this->m_blocksYT[k].row, this->m_blocksYT[k].value.numZeros(), subdiagElems,
						mat.rows() - this->m_blocksYT[k].value.numDenseRows(), this->m_blocksYT[k].value.numDenseRows())
				}
				// Write the result back to ci-th column of res
				mat.col(ci) = resColJd;// .sparseView();
			}
			//*/
		}
		else {
			Index blockRows = this->m_blocksYT[blockK].value.rows();
			/*
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
						//VectorXd resColJd;
						//VectorXd tmpResColJ;
						//double duration = 0;
						//clock_t start;

						// loop over all items
						for (int j = bi; j < ei; j++)
						{
							// inner loop
							{
								// Use temporary vector resColJ inside of the for loop - faster access

								//resColJd = mat.col(fromIdx + j);// .toDense();
								//Index k = blockK;
								//typename MatrixType::StorageIndex subdiagElems = this->m_blocksYT[k].value.rows() - this->m_blocksYT[k].value.cols();
								//FULL_TO_BLOCK_VEC_EXT(resColJd, tmpResColJ, this->m_blocksYT[k].value.rows() - this->m_blocksYT[k].value.numDenseRows(), this->m_blocksYT[k].value.cols(), this->m_blocksYT[k].row, this->m_blocksYT[k].value.numZeros(), subdiagElems,
								//mat.rows() - this->m_blocksYT[k].value.numDenseRows(), this->m_blocksYT[k].value.numDenseRows())

								//// We can afford noalias() in this case
								//start = clock();
								//tmpResColJ.noalias() += this->m_blocksYT[k].value.multTransposed(tmpResColJ);
								//duration += double(clock() - start);

								//BLOCK_VEC_TO_FULL_EXT(resColJd, tmpResColJ, this->m_blocksYT[k].value.rows() - this->m_blocksYT[k].value.numDenseRows(), this->m_blocksYT[k].value.cols(), this->m_blocksYT[k].row, this->m_blocksYT[k].value.numZeros(), subdiagElems,
								//mat.rows() - this->m_blocksYT[k].value.numDenseRows(), this->m_blocksYT[k].value.numDenseRows())

								////std::lock_guard<std::mutex> lock(critical);
								//// Write the result back to ci-th column of res
								//mat.col(fromIdx + j) = resColJd;// .sparseView();
								//start = clock();
								mat.bottomRows(blockRows).col(fromIdx + j).noalias()
									+= (this->m_blocksYT[blockK].value.Y() * (this->m_blocksYT[blockK].value.T().transpose() * (this->m_blocksYT[blockK].value.Y().transpose() * mat.bottomRows(blockRows).col(fromIdx + j))));
								//mat.bottomRows(blockRows).col(fromIdx + j).noalias() += this->m_blocksYT[blockK].value.multTransposed(mat.bottomRows(blockRows).col(fromIdx + j));
								//duration += double(clock() - start);
							}
						}
						//std::cout << "Thread in mult:   " << duration / CLOCKS_PER_SEC << "s\n";

					}, t*nloop / nthreads, (t + 1) == nthreads ? nloop : (t + 1)*nloop / nthreads, t));
				}
				std::for_each(threads.begin(), threads.end(), [](std::thread& x) {x.join(); });
			}
			//*/
			/*
			VectorXd resColJd;
			VectorXd tmpResColJ;
			for (int ci = fromIdx; ci < toIdx; ci++) {
			// Use temporary vector resColJ inside of the for loop - faster access
			resColJd = mat.col(ci);// .toDense();
			Index k = blockK;
			typename MatrixType::StorageIndex subdiagElems = this->m_blocksYT[k].value.rows() - this->m_blocksYT[k].value.cols();
			FULL_TO_BLOCK_VEC_EXT(resColJd, tmpResColJ, this->m_blocksYT[k].value.rows() - this->m_blocksYT[k].value.numDenseRows(), this->m_blocksYT[k].value.cols(), this->m_blocksYT[k].row, this->m_blocksYT[k].value.numZeros(), subdiagElems,
			mat.rows() - this->m_blocksYT[k].value.numDenseRows(), this->m_blocksYT[k].value.numDenseRows())

			// We can afford noalias() in this case
			tmpResColJ += this->m_blocksYT[k].value.multTransposed(tmpResColJ);

			BLOCK_VEC_TO_FULL_EXT(resColJd, tmpResColJ, this->m_blocksYT[k].value.rows() - this->m_blocksYT[k].value.numDenseRows(), this->m_blocksYT[k].value.cols(), this->m_blocksYT[k].row, this->m_blocksYT[k].value.numZeros(), subdiagElems,
			mat.rows() - this->m_blocksYT[k].value.numDenseRows(), this->m_blocksYT[k].value.numDenseRows())
			// Write the result back to ci-th column of res
			mat.col(ci) = resColJd;// .sparseView();
			}
			//*/
			///*
			//double duration = 0;
			//clock_t start;
			//start = clock();
			//if (toIdx - fromIdx == 1){
			//	SparseBlockYTY::Element blockYTY = this->m_blocksYT[blockK];
				//mat.bottomRows(blockRows).middleCols(fromIdx, SuggestedBlockCols) += (blockYTY.value.Y() * (blockYTY.value.T() * (blockYTY.value.Y().transpose() * mat.bottomRows(blockRows).middleCols(fromIdx, SuggestedBlockCols))));
				//mat.bottomRows(blockRows).middleCols(fromIdx, SuggestedBlockCols) += this->m_blocksYT[blockK].value.multTransposed(mat.bottomRows(blockRows).middleCols(fromIdx, SuggestedBlockCols));
			//} else {
				for (int ci = fromIdx; ci < toIdx; ci++) {
					// We can afford noalias() in this case
					// mat.col(ci).segment(mat.rows() - this->m_blocksYT[blockK].value.rows(), this->m_blocksYT[blockK].value.rows()).noalias() += this->m_blocksYT[blockK].value.multTransposed(mat.col(ci).segment(mat.rows() - this->m_blocksYT[blockK].value.rows(), this->m_blocksYT[blockK].value.rows()));
					//mat.bottomRows(blockRows).col(ci) += this->m_blocksYT[blockK].value.multTransposed(mat.bottomRows(blockRows).col(ci));

					mat.bottomRows(blockRows).col(ci).noalias()
						+= (this->m_blocksYT[blockK].value.Y() * (this->m_blocksYT[blockK].value.T().transpose() * (this->m_blocksYT[blockK].value.Y().transpose() * mat.bottomRows(blockRows).col(ci))));
					//mat.bottomRows(blockRows).col(ci).noalias() += this->m_blocksYT[blockK].value * mat.bottomRows(blockRows).col(ci);
				}
			//}
			//duration += double(clock() - start);
			//std::cout << "Thread in mult:   " << duration / CLOCKS_PER_SEC << "s\n";
			//*/
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
	* The function DenseBlockedSparseQR::analyzePattern(const MatrixType&) must have been called beforehand with
	* a matrix having the same sparsity pattern than \a mat.
	*
	* \param mat The sparse column-major matrix
	*/
	template <typename MatrixType, typename OrderingType, int SuggestedBlockCols, bool MultiThreading>
	void DenseBlockedSparseQR<MatrixType, OrderingType, SuggestedBlockCols, MultiThreading>::factorize(const MatrixType& mat)
	{
		// Not rank-revealing, column permutation is identity
		m_outputPerm_c.setIdentity(mat.cols());

		// Permute the input matrix using the precomputed row permutation
		Index numDenseRows = (mat.rows() - this->m_denseStartRow);
		m_pmat = DenseMatrixType(mat.cols() + numDenseRows, mat.cols());
		m_pmat.topRows(mat.cols()) = mat.block(0, 0, mat.cols(), mat.cols()).toDense();
		m_pmat.bottomRows(numDenseRows) = mat.block(this->m_denseStartRow, 0, numDenseRows, mat.cols()).toDense();
		Index idxDenseRow = mat.cols();
		
		//m_pmat = mat.toDense();// (this->m_rowPerm * mat);

		// Triplet array for the matrix R
		//		Eigen::TripletArray<Scalar, typename MatrixType::Index> Rvals(2 * mat.nonZeros());

		// Dense QR solver used for each dense block 
		Eigen::HouseholderQR<DenseMatrixType> houseqr;
		Index numBlocks = this->m_blockOrder.size();

		// Prepare the first block
		MatrixBlockInfo bi = this->m_blockMap.at(this->m_blockOrder.at(0));
		DenseMatrixType Ji = DenseMatrixType(bi.numDiagRows + bi.numDenseRows, bi.numCols);
		Ji.topRows(bi.numDiagRows) = m_pmat.block(bi.idxDiagRow, bi.idxDiagRow, bi.numDiagRows, bi.numCols);
		Ji.bottomRows(bi.numDenseRows) = m_pmat.block(idxDenseRow, bi.idxDiagRow, bi.numDenseRows, bi.numCols);//.toDense();
		Index activeRows = bi.numDiagRows + bi.numDenseRows;
		Index numZeros = idxDenseRow - (bi.idxDiagRow + bi.numDiagRows);

		// Auxiliary variables
		MatrixBlockInfo biNext;

		// Process all blocks
//		clock_t beginLoop = clock();
		for (Index i = 0; i < numBlocks; i++) {
//			clock_t begin = clock();
			// Current block info
			bi = this->m_blockMap.at(this->m_blockOrder.at(i));

			// 1) Solve the current dense block using dense Householder QR
		//	clock_t dd = clock();
			houseqr.compute(Ji);
		//	std::cout << "factorize block   " << double(clock() - dd) / CLOCKS_PER_SEC << "s\n";
		

			// 2) Create matrices T and Y
			MatrixXd T = MatrixXd::Zero(bi.numCols, bi.numCols);
			MatrixXd Y = MatrixXd::Zero(m_pmat.rows() - bi.idxDiagRow, bi.numCols);
			VectorXd v = VectorXd::Zero(m_pmat.rows() - bi.idxDiagRow);
			VectorXd z = VectorXd::Zero(m_pmat.rows() - bi.idxDiagRow);
			v(0) = 1.0;
			v.segment(1, bi.numDiagRows - 1) = houseqr.householderQ().essentialVector(0).segment(0, bi.numDiagRows - 1);
			v.segment(idxDenseRow - bi.idxDiagRow, numDenseRows) = houseqr.householderQ().essentialVector(0).segment(bi.numDiagRows - 1, numDenseRows);
			Y.col(0) = v;
			T(0, 0) = -houseqr.hCoeffs()(0);
			//std::cout << "--matcol0--\n" << Ji.col(0) << std::endl;
			//std::cout << "--matcol0--\n" << m_pmat.col(0) << std::endl;
			//std::cout << "--h0--\n" << houseqr.householderQ().essentialVector(0) << std::endl;
			//std::cout << "--v0--\n" << v << std::endl;
			for (typename MatrixType::StorageIndex bc = 1; bc < bi.numCols; bc++) {
				v.setZero();
				v(bc) = 1.0;
				v.segment(bc + 1, bi.numDiagRows - bc - 1) = houseqr.householderQ().essentialVector(bc).segment(0, bi.numDiagRows - bc - 1);
				v.segment(idxDenseRow - bi.idxDiagRow, numDenseRows) = houseqr.householderQ().essentialVector(bc).segment(bi.numDiagRows - bc - 1, numDenseRows);

				z = -houseqr.hCoeffs()(bc) * (T * (Y.transpose() * v));

				Y.col(bc) = v;
				T.col(bc) = z;
				T(bc, bc) = -houseqr.hCoeffs()(bc);
			}
			//std::cout << "--Y--\n" << Y << std::endl;
			// Save current Y and T. The block YTY contains a main diagonal and subdiagonal part separated by (numZeros) zero rows.
			m_blocksYT.insert(SparseBlockYTY::Element(bi.idxDiagRow, bi.idxDiagRow, BlockYTY_Ext2<Scalar, StorageIndex>(Y, T, numZeros, bi.numDenseRows)));

			// 3) Get the R part of the dense QR decomposition 
			/*		MatrixXd V = houseqr.matrixQR().template triangularView<Upper>();
			// Update sparse R with the rows solved in this step
			int solvedRows = (i == numBlocks - 1) ? bi.numDiagRows : this->m_blockMap.at(this->m_blockOrder.at(i + 1)).idxCol - bi.idxCol;
			for (typename MatrixType::StorageIndex br = 0; br < solvedRows; br++) {
			for (typename MatrixType::StorageIndex bc = 0; bc < bi.numCols; bc++) {
			Rvals.add_if_nonzero(diagIdx + br, bi.idxCol + bc, V(br, bc));
			}
			}
			*/
			//this->updateMat(bi.idxDiagRow, bi.idxDiagRow + 3, m_pmat, this->m_blocksYT.size() - 1);
			//clock_t updatetime = clock();
			this->updateMat(bi.idxDiagRow, m_pmat.cols(), m_pmat, this->m_blocksYT.size() - 1);
			//std::cout << "Upd   " << double(clock() - updatetime) / CLOCKS_PER_SEC << "s\n";

			//break;
			//this->updateMat(bi.idxDiagRow, bi.idxDiagRow + bi.numCols, m_pmat, this->m_blocksYT.size() - 1);

			// 4) If this is not the last block, proceed to the next block
			if (i < numBlocks - 1) {
				biNext = this->m_blockMap.at(this->m_blockOrder.at(i + 1));

				// Now update the unsolved rest of m_pmat
				//this->updateMat(biNext.idxDiagRow, biNext.idxDiagRow + biNext.numCols, m_pmat);

				activeRows = biNext.numDiagRows + biNext.numDenseRows;
				numZeros = biNext.idxDenseRow - (biNext.idxDiagRow + biNext.numDiagRows);

				Ji = DenseMatrixType(activeRows, biNext.numCols);
				Ji.topRows(biNext.numDiagRows) = m_pmat.block(biNext.idxDiagRow, biNext.idxDiagRow, biNext.numDiagRows, biNext.numCols);
				Ji.bottomRows(biNext.numDenseRows) = m_pmat.block(idxDenseRow, biNext.idxDiagRow, biNext.numDenseRows, biNext.numCols);//.toDense();
			}
//			std::cout << "iter   " << double(clock() - begin) / CLOCKS_PER_SEC << "s\n";
		}
//		std::cout << "Loop   " << double(clock() - beginLoop) / CLOCKS_PER_SEC << "s\n";

		// 5) Finalize the R matrix and set factorization-related flags
		//		m_R.setFromTriplets(Rvals.begin(), Rvals.end());
		//		m_R.makeCompressed();

		m_R = m_pmat.sparseView();
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
	* The choice of implementation depends on a template parameter of the DenseBlockedSparseQR class.
	* The single-threaded implementation cannot work in-place. It is implemented this way for performance related reasons.
	*/
	template <typename DenseBlockedSparseQRType, typename Derived>
	struct DenseBlockedSparseQR_QProduct : ReturnByValue<DenseBlockedSparseQR_QProduct<DenseBlockedSparseQRType, Derived> >
	{
		typedef typename DenseBlockedSparseQRType::MatrixType MatrixType;
		typedef typename DenseBlockedSparseQRType::Scalar Scalar;

		typedef typename internal::DenseBlockedSparseQR_traits<MatrixType>::Vector SparseVector;

		// Get the references 
		DenseBlockedSparseQR_QProduct(const DenseBlockedSparseQRType& qr, const Derived& other, bool transpose) :
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

		const DenseBlockedSparseQRType& m_qr;
		const Derived& m_other;
		bool m_transpose;
	};

	/*
	* Specialization of the Householder product evaluation performing Q * A or Q.T * A
	* for the case when A and the output are dense vectors.=
	* Offers only single-threaded implementation as the overhead of multithreading would not bring any speedup for a dense vector (A is single column).
	*/
	template <typename DenseBlockedSparseQRType>
	struct DenseBlockedSparseQR_QProduct<DenseBlockedSparseQRType, VectorX> : ReturnByValue<DenseBlockedSparseQR_QProduct<DenseBlockedSparseQRType, VectorX> >
	{
		typedef typename DenseBlockedSparseQRType::MatrixType MatrixType;
		typedef typename DenseBlockedSparseQRType::Scalar Scalar;

		// Get the references 
		DenseBlockedSparseQR_QProduct(const DenseBlockedSparseQRType& qr, const VectorX& other, bool transpose) :
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

		const DenseBlockedSparseQRType& m_qr;
		const VectorX& m_other;
		bool m_transpose;
	};

	template<typename DenseBlockedSparseQRType>
	struct DenseBlockedSparseQRMatrixQReturnType : public EigenBase<DenseBlockedSparseQRMatrixQReturnType<DenseBlockedSparseQRType> >
	{
		typedef typename DenseBlockedSparseQRType::Scalar Scalar;
		typedef Matrix<Scalar, Dynamic, Dynamic> DenseMatrix;
		enum {
			RowsAtCompileTime = Dynamic,
			ColsAtCompileTime = Dynamic
		};
		explicit DenseBlockedSparseQRMatrixQReturnType(const DenseBlockedSparseQRType& qr) : m_qr(qr) {}
		/*DenseBlockedSparseQR_QProduct<DenseBlockedSparseQRType, Derived> operator*(const MatrixBase<Derived>& other)
		{
		return DenseBlockedSparseQR_QProduct<DenseBlockedSparseQRType,Derived>(m_qr,other.derived(),false);
		}*/
		template<typename Derived>
		DenseBlockedSparseQR_QProduct<DenseBlockedSparseQRType, Derived> operator*(const MatrixBase<Derived>& other)
		{
			return DenseBlockedSparseQR_QProduct<DenseBlockedSparseQRType, Derived>(m_qr, other.derived(), false);
		}
		template<typename _Scalar, int _Options, typename _Index>
		DenseBlockedSparseQR_QProduct<DenseBlockedSparseQRType, SparseMatrix<_Scalar, _Options, _Index>> operator*(const SparseMatrix<_Scalar, _Options, _Index>& other)
		{
			return DenseBlockedSparseQR_QProduct<DenseBlockedSparseQRType, SparseMatrix<_Scalar, _Options, _Index>>(m_qr, other, false);
		}
		DenseBlockedSparseQRMatrixQTransposeReturnType<DenseBlockedSparseQRType> adjoint() const
		{
			return DenseBlockedSparseQRMatrixQTransposeReturnType<DenseBlockedSparseQRType>(m_qr);
		}
		inline Index rows() const { return m_qr.rows(); }
		inline Index cols() const { return m_qr.rows(); }
		// To use for operations with the transpose of Q
		DenseBlockedSparseQRMatrixQTransposeReturnType<DenseBlockedSparseQRType> transpose() const
		{
			return DenseBlockedSparseQRMatrixQTransposeReturnType<DenseBlockedSparseQRType>(m_qr);
		}

		const DenseBlockedSparseQRType& m_qr;
	};

	template<typename DenseBlockedSparseQRType>
	struct DenseBlockedSparseQRMatrixQTransposeReturnType
	{
		explicit DenseBlockedSparseQRMatrixQTransposeReturnType(const DenseBlockedSparseQRType& qr) : m_qr(qr) {}
		template<typename Derived>
		DenseBlockedSparseQR_QProduct<DenseBlockedSparseQRType, Derived> operator*(const MatrixBase<Derived>& other)
		{
			return DenseBlockedSparseQR_QProduct<DenseBlockedSparseQRType, Derived>(m_qr, other.derived(), true);
		}
		template<typename _Scalar, int _Options, typename _Index>
		DenseBlockedSparseQR_QProduct<DenseBlockedSparseQRType, SparseMatrix<_Scalar, _Options, _Index>> operator*(const SparseMatrix<_Scalar, _Options, _Index>& other)
		{
			return DenseBlockedSparseQR_QProduct<DenseBlockedSparseQRType, SparseMatrix<_Scalar, _Options, _Index>>(m_qr, other, true);
		}
		const DenseBlockedSparseQRType& m_qr;
	};

	namespace internal {

		template<typename DenseBlockedSparseQRType>
		struct evaluator_traits<DenseBlockedSparseQRMatrixQReturnType<DenseBlockedSparseQRType> >
		{
			typedef typename DenseBlockedSparseQRType::MatrixType MatrixType;
			typedef typename storage_kind_to_evaluator_kind<typename MatrixType::StorageKind>::Kind Kind;
			typedef SparseShape Shape;
		};

		template< typename DstXprType, typename DenseBlockedSparseQRType>
		struct Assignment<DstXprType, DenseBlockedSparseQRMatrixQReturnType<DenseBlockedSparseQRType>, internal::assign_op<typename DstXprType::Scalar, typename DstXprType::Scalar>, Sparse2Sparse>
		{
			typedef DenseBlockedSparseQRMatrixQReturnType<DenseBlockedSparseQRType> SrcXprType;
			typedef typename DstXprType::Scalar Scalar;
			typedef typename DstXprType::StorageIndex StorageIndex;
			static void run(DstXprType &dst, const SrcXprType &src, const internal::assign_op<Scalar, Scalar> &/*func*/)
			{
				typename DstXprType::PlainObject idMat(src.m_qr.rows(), src.m_qr.rows());
				idMat.setIdentity();
				// Sort the sparse householder reflectors if needed
				//const_cast<DenseBlockedSparseQRType *>(&src.m_qr)->_sort_matrix_Q();
				dst = DenseBlockedSparseQR_QProduct<DenseBlockedSparseQRType, DstXprType>(src.m_qr, idMat, false);
			}
		};

		template< typename DstXprType, typename DenseBlockedSparseQRType>
		struct Assignment<DstXprType, DenseBlockedSparseQRMatrixQReturnType<DenseBlockedSparseQRType>, internal::assign_op<typename DstXprType::Scalar, typename DstXprType::Scalar>, Sparse2Dense>
		{
			typedef DenseBlockedSparseQRMatrixQReturnType<DenseBlockedSparseQRType> SrcXprType;
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
