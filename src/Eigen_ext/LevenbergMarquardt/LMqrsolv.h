// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Thomas Capricelli <orzel@freehackers.org>
// Copyright (C) 2012 Desire Nuentsa <desire.nuentsa_wakam@inria.fr>
//
// This code initially comes from MINPACK whose original authors are:
// Copyright Jorge More - Argonne National Laboratory
// Copyright Burt Garbow - Argonne National Laboratory
// Copyright Ken Hillstrom - Argonne National Laboratory
//
// This Source Code Form is subject to the terms of the Minpack license
// (a BSD-like license) described in the campaigned CopyrightMINPACK.txt file.

#ifndef EIGEN_LMQRSOLV_H
#define EIGEN_LMQRSOLV_H

#include "../../BlockAngularSparseQR_Ext.h"
#include "../../BlockDiagonalSparseQR_Ext.h"
#include "../../DenseBlockedThinQR.h"

namespace Eigen {

	namespace internal {

		template <typename Scalar, int Rows, int Cols, typename PermIndex>
		void lmqrsolv(
			Matrix<Scalar, Rows, Cols> &s,
			const PermutationMatrix<Cols, Cols, PermIndex> &iPerm,
			const Matrix<Scalar, Cols, 1> &diag,
			const Matrix<Scalar, Cols, 1> &qtb,
			Matrix<Scalar, Cols, 1> &x,
			Matrix<Scalar, Cols, 1> &sdiag)
		{
			/* Local variables */
			Index i, j, k;
			Scalar temp;
			Index n = s.cols();
			Matrix<Scalar, Dynamic, 1>  wa(n);
			JacobiRotation<Scalar> givens;

			/* Function Body */
			// the following will only change the lower triangular part of s, including
			// the diagonal, though the diagonal is restored afterward

			/*     copy r and (q transpose)*b to preserve input and initialize s. */
			/*     in particular, save the diagonal elements of r in x. */
			x = s.diagonal();
			wa = qtb;

			s.topLeftCorner(n, n).template triangularView<StrictlyLower>() = s.topLeftCorner(n, n).transpose();
			/*     eliminate the diagonal matrix d using a givens rotation. */
			for (j = 0; j < n; ++j) {

				/*        prepare the row of d to be eliminated, locating the */
				/*        diagonal element using p from the qr factorization. */
				const PermIndex l = iPerm.indices()(j);
				if (diag[l] == 0.) {
					break;
				}
				sdiag.tail(n - j).setZero();
				sdiag[j] = diag[l];

				/*        the transformations to eliminate the row of d */
				/*        modify only a single element of (q transpose)*b */
				/*        beyond the first n, which is initially zero. */
				Scalar qtbpj = 0.;
				for (k = j; k < n; ++k) {
					/*           determine a givens rotation which eliminates the */
					/*           appropriate element in the current row of d. */
					givens.makeGivens(-s(k, k), sdiag[k]);

					/*           compute the modified diagonal element of r and */
					/*           the modified element of ((q transpose)*b,0). */
					s(k, k) = givens.c() * s(k, k) + givens.s() * sdiag[k];
					temp = givens.c() * wa[k] + givens.s() * qtbpj;
					qtbpj = -givens.s() * wa[k] + givens.c() * qtbpj;
					wa[k] = temp;

					/*           accumulate the tranformation in the row of s. */
					for (i = k + 1; i < n; ++i) {
						temp = givens.c() * s(i, k) + givens.s() * sdiag[i];
						sdiag[i] = -givens.s() * s(i, k) + givens.c() * sdiag[i];
						s(i, k) = temp;
					}
				}
			}

			/*     solve the triangular system for z. if the system is */
			/*     singular, then obtain a least squares solution. */
			Index nsing;
			for (nsing = 0; nsing < n && sdiag[nsing] != 0; nsing++) {}

			wa.tail(n - nsing).setZero();
			s.topLeftCorner(nsing, nsing).transpose().template triangularView<Upper>().solveInPlace(wa.head(nsing));

			// restore
			sdiag = s.diagonal();
			s.diagonal() = x;

			/* permute the components of z back to components of x. */
			x = iPerm * wa;
		}

		template <typename Scalar, int _Options, typename _Index, typename PermIndex>
		void lmqrsolv(
			SparseMatrix<Scalar, _Options, _Index> &s,
			const PermutationMatrix<Dynamic, Dynamic, PermIndex> &iPerm,
			const Matrix<Scalar, Dynamic, 1> &diag,
			const Matrix<Scalar, Dynamic, 1> &qtb,
			Matrix<Scalar, Dynamic, 1> &x,
			Matrix<Scalar, Dynamic, 1> &sdiag)
		{
			typedef typename SparseMatrix<Scalar, _Options, _Index>::Index Index;
			typedef SparseMatrix<Scalar, RowMajor, Index> FactorType;

			/* Local variables */
			Index i, j, k, l;
			Scalar temp;
			Index n = s.cols();
			Matrix<Scalar, Dynamic, 1>  wa(n);
			JacobiRotation<Scalar> givens;

			/* Function Body */
			// the following will only change the lower triangular part of s, including
			// the diagonal, though the diagonal is restored afterward

			/*     copy r and (q transpose)*b to preserve input and initialize R. */
			wa = qtb;
			FactorType R(s);
			// Eliminate the diagonal matrix d using a givens rotation
			for (j = 0; j < n; ++j) {
				// Prepare the row of d to be eliminated, locating the 
				// diagonal element using p from the qr factorization
				l = iPerm.indices()(j);
				if (diag(l) == Scalar(0)) {
					break;
				}
				sdiag.tail(n - j).setZero();
				sdiag[j] = diag[l];
				// the transformations to eliminate the row of d
				// modify only a single element of (q transpose)*b
				// beyond the first n, which is initially zero. 

				Scalar qtbpj = 0;
				// Browse the nonzero elements of row j of the upper triangular s
				for (k = j; k < n; ++k) {
					typename FactorType::InnerIterator itk(R, k);
					for (; itk; ++itk) {
						if (itk.index() < k) {
							continue;
						}
						else {
							break;
						}
					}
					//At this point, we have the diagonal element R(k,k)
					// Determine a givens rotation which eliminates 
					// the appropriate element in the current row of d
					givens.makeGivens(-itk.value(), sdiag(k));

					// Compute the modified diagonal element of r and 
					// the modified element of ((q transpose)*b,0).
					itk.valueRef() = givens.c() * itk.value() + givens.s() * sdiag(k);
					temp = givens.c() * wa(k) + givens.s() * qtbpj;
					qtbpj = -givens.s() * wa(k) + givens.c() * qtbpj;
					wa(k) = temp;

					// Accumulate the transformation in the remaining k row/column of R
					for (++itk; itk; ++itk) {
						i = itk.index();
						temp = givens.c() *  itk.value() + givens.s() * sdiag(i);
						sdiag(i) = -givens.s() * itk.value() + givens.c() * sdiag(i);
						itk.valueRef() = temp;
					}
				}
			}

			// awf moved here from 2nd last line below -- sdiag was not signalling rank deficiency in R (last 3 rows all zero)
			sdiag = R.diagonal();


			// Solve the triangular system for z. If the system is 
			// singular, then obtain a least squares solution
			Index nsing;
			for (nsing = 0; nsing < n && sdiag(nsing) != 0; nsing++) {}

			//write(R, "c:/tmp/R.txt");
			//write(sdiag, "c:/tmp/sdiag.txt");
			//write(diag, "c:/tmp/diag.txt");


			wa.tail(n - nsing).setZero();
			//     x = wa; 
			wa.head(nsing) = R.topLeftCorner(nsing, nsing).template triangularView<Upper>().solve/*InPlace*/(wa.head(nsing));

			// Permute the components of z back to components of x
			x = iPerm * wa;
		}


		/************************* New more efficient QR solver **************************/
		template <typename MatrixType, typename PermIndex>
		void interleaveSparse(const MatrixType &R, const Matrix<Scalar, Dynamic, 1> &diag, const PermutationMatrix<Dynamic, Dynamic, PermIndex> &iPerm, MatrixType &res) {
			res.resize(R.cols() * 2, R.cols());
			res.reserve(R.nonZeros() + R.cols()); // New matrix will contain R + diagonal

			// By default interleave 1 - 1 (can be changed by changing blocksz)
			Index blocksz = 1;
			Index stride = blocksz * 2;
			for (int i = 0; i < R.cols() / blocksz; i += 1) {
				for (int j = 0; j < blocksz; j++) {
					res.startVec(i * stride + j);
					// Insert i-th row of the upper triangular part (stored in R)
					for (typename MatrixType::InnerIterator rowIt(R, i * blocksz + j); rowIt; ++rowIt) {
						res.insertBack(i * stride + j, rowIt.index()) = rowIt.value();
					}
				}
				// Insert i-th row of the diagonal part (stored in diag)
				for (int j = blocksz; j < stride; j++) {
					res.startVec(i * stride + j);
					res.insertBack(i * stride + j, i + j - blocksz) = diag(iPerm.indices()(i + j - blocksz));
					//for (JacobianTypeRowMajor::InnerIterator rowIt(spJ, i * blocksz + R.cols() + j - blocksz); rowIt; ++rowIt) {
					//	res.insertBack(i * stride + j, rowIt.index()) = rowIt.value();
					//}
				}
			}
			res.finalize();
		}

		template <typename QRSolver, typename QRSolverInner, typename Scalar, int _Options, typename _Index, typename PermIndex>
		void lmqrsolv2(
			const QRSolver &qr,
			QRSolverInner &slvr,
			SparseMatrix<Scalar, _Options, _Index> &s,
			const PermutationMatrix<Dynamic, Dynamic, PermIndex> &iPerm,
			const Matrix<Scalar, Dynamic, 1> &diag,
			const Matrix<Scalar, Dynamic, 1> &qtb,
			Matrix<Scalar, Dynamic, 1> &x,
			Matrix<Scalar, Dynamic, 1> &sdiag)
		{
			typedef typename SparseMatrix<Scalar, _Options, _Index>::Index Index;
			typedef SparseMatrix<Scalar, RowMajor, Index> FactorType;
			typedef Matrix<Scalar, Dynamic, Dynamic> DenseMatrix;
			/*
			typedef ColPivHouseholderQR<Matrix<Scalar, Dynamic, Dynamic> > DenseBlockSolver;
			typedef BlockDiagonalSparseQR_Ext<FactorType, DenseBlockSolver> LeftBlkSolver;
			typedef DenseBlockedThinQR<DenseMatrix, NaturalOrdering<Index>, 10, true> RightBlkSolver;
			typedef BlockAngularSparseQR_Ext<FactorType, LeftBlkSolver, RightBlkSolver> SchurlikeQRSolver;
			*/
			/* Local variables */
			Index i, j, k, l;
			Scalar temp;
			Index n = s.cols();
			// Interleave qtb with vector of 0 (interleaved right side corresponding to interleaving R and lambdas on the left)
			Matrix<Scalar, Dynamic, 1>  wa = Matrix<Scalar, Dynamic, 1>::Zero(2 * n);
			Map<Matrix<Scalar, Dynamic, 1>, 0, InnerStride<2>>(wa.data(), n) = qtb;
			//wa.head(n) = qtb;

			// Compose the final matrix to be factorized by interleaving R and the diagonal
			FactorType RD;
			interleaveSparse<FactorType, PermIndex>(s, diag, iPerm, RD);
		
			//// Factorize the interleaved matrix to get "something" upper triangular again
			//static SchurlikeQRSolver slvr;
			Index leftBlockCols = qr.leftBlockCols();
			Index rightBlockCols = RD.cols() - leftBlockCols;
			slvr.setSparseBlockParams(RD.rows() - 2 * rightBlockCols, leftBlockCols);
			slvr.getLeftSolver().setPattern(RD.rows() - 2 * rightBlockCols, leftBlockCols, 6, 3);
			slvr.compute(RD);

			//SparseMatrix<Scalar, RowMajor, Index> factType = RD;
			//std::cout << "Q * R - RD = " << (slvr.matrixQ() * slvr.matrixR() - factType).norm() << std::endl;
			//std::cout << "Q.T * RD - R = " << (slvr.matrixQ().transpose() * factType - slvr.matrixR()).norm() << std::endl;
			
			// awf moved here from 2nd last line below -- sdiag was not signalling rank deficiency in R (last 3 rows all zero)
			sdiag = slvr.colsPermutation() * slvr.matrixR().diagonal();
			
			// Solve the triangular system for z. If the system is 
			// singular, then obtain a least squares solution
			Index nsing;
			for (nsing = 0; nsing < n && sdiag(nsing) != 0; nsing++) {}
			
			// Update qtb as well
			wa = slvr.matrixQ().transpose() * wa;
			wa.head(n) = wa.head(n) * slvr.colsPermutation();
			wa.tail(2 * n - nsing).setZero();
			wa.head(nsing) = slvr.matrixR().topLeftCorner(nsing, nsing).template triangularView<Upper>().solve/*InPlace*/(wa.head(nsing));
			
			//// Permute the components of z back to components of x
			wa.head(n) = slvr.colsPermutation() * wa.head(n);
			//wa.head(n) = ;
			//Matrix<Scalar, Dynamic, 1> wa2 = Map<Matrix<Scalar, Dynamic, 1>, 0, InnerStride<2>>(wa.data(), n);
			//x = iPerm * wa2; //wa.head(n)
			x = iPerm * wa.head(n);
		}
	} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_LMQRSOLV_H