//
// Copyright (C) 2017 Jan Svoboda <jan.svoboda@usi.ch>
// Copyright (C) 2016 Andrew Fitzgibbon <awf@microsoft.com>
//
// Reimplementation of Matlab template LM imlementation found in AWF Utility Library:
// https://github.com/awf/awful/blob/master/matlab/au_levmarq.m
//

#ifndef SIMPLELM_H
#define SIMPLELM_H

#include <Eigen/Eigen>
#include <Eigen/Sparse>


namespace Eigen {
	/**
	* \class LMFunctor
	*
	* \brief Functor template for SimpleLM
	*
	* This functor models a function
	*      ValueType f(InputType x);
	* where ValueType and InputType are, or are like, vectors of scalars.
	*
	* \tparam _Scalar The type of the scalars
	* \tparam NX The number of elements in the InputType
	* \tparam NY The number of elements in the ValueType
	*
	*/
	template <typename _Scalar, int NX = Dynamic, int NY = Dynamic>
	struct LMFunctor {
		typedef _Scalar Scalar;
		enum {
			InputsAtCompileTime = NX,
			ValuesAtCompileTime = NY
		};

		LMFunctor() : m_inputs(InputsAtCompileTime), m_values(ValuesAtCompileTime) {}
		LMFunctor(Index inputs, Index values) : m_inputs(inputs), m_values(values) {}

		const Index m_inputs, m_values;

		Index inputs() const { return m_inputs; }
		Index values() const { return m_values; }

		// Three possibly distinct datatypes.  Consider the functor's operator() to 
		// have a signature like this:
		//   ValueType f(InputType x);
		typedef Matrix<Scalar, InputsAtCompileTime, 1> InputType;
		typedef Matrix<Scalar, ValuesAtCompileTime, 1> ValueType;

		// In some optimization situations, the number of columns in the Jacobian may not be the
		// same as the number of scalars in x, for example if optimizing over a subset of parameters,
		// or if parameterizing an update in a tangent plane.  Thus there is a third type which 
		// represents the type of an update to x.   That will typically be a vector type.
		typedef Matrix<Scalar, InputsAtCompileTime, 1> StepType;

		// Following functions has to be defined in the derived classes
		//virtual int operator() (const InputType &x, ValueType& fvec) = 0;
		//virtual int df(const InputType &x, JacobianType& fjac) = 0;
		// Derived classes may choose to define initQRSolver, e.g. to set block size parameters, convergence tolerances.
		// In particular, sparse solvers can benefit by expressing problem structure (see e.g. ellipse_fitting test)
		//virtual void initQRSolver(QRSolver &) = 0;
		
		// For some StepTypes, derived classes may need to implement increment_in_place(InputType, StepType)
		void increment_in_place(InputType* x, StepType const& delta) {
			*x += delta;
		}

		// Norm of inputType scaled by diag.   With non-standard inputtypes, it's probably better to
		// ensure your problem scaling is good, and just return diag.stableNorm() here.
		Scalar estimateNorm(InputType const&x, StepType const& diag) {
			return x.cwiseProduct(diag).stableNorm();
		}
	};

	// Specialization of LevenbergMarquardtFunctor for dense Jacobian
	template <typename _Scalar, int NX = Dynamic, int NY = Dynamic>
	struct DenseLMFunctor : public LMFunctor<_Scalar, NX, NY>
	{
		typedef LMFunctor<_Scalar, NX, NY> Base;

		typedef Matrix<_Scalar, NX, NX> JacobianType;
		typedef ColPivHouseholderQR<JacobianType> QRSolver;

		DenseLMFunctor() : Base(NX, NY) {}
		DenseLMFunctor(int inputs, int values) : Base(inputs, values) {}

		void initQRSolver(QRSolver &) {}
	};

	// Specialization of LevenbergMarquardtFunctor for general sparse Jacobian.
	// InputType and ValueType will generally be full, so are left as dense vectors.
	template <typename _Scalar, typename _Index = Index>
	struct SparseLMFunctor : public LMFunctor<_Scalar, Dynamic, Dynamic>
	{
		typedef LMFunctor<_Scalar, Dynamic, Dynamic> Base;

		typedef _Index Index;

		typedef SparseMatrix<_Scalar, ColMajor, _Index> JacobianType;
		typedef SparseQR<JacobianType, COLAMDOrdering<int> > QRSolver;

		SparseLMFunctor(Index inputs, Index values) : Base(inputs, values) {}

		void initQRSolver(QRSolver &) {}
	};

	namespace SimpleLMInfo {
		enum Status {
			NotStarted = -2,
			Running = -1,
			Success = 0,
			ExceededLambdaMax = 1,
			TooManyFunctionEvaluation = 2,
			MaxItersReached = 3
		};

		std::string statusToString(const Status &status) {
			switch (status) {
			case Status::NotStarted:
				return "Not Started";
			case Status::Running:
				return "Running";
			case Status::Success:
				return "Success (Energy Flatlined)";
			case Status::ExceededLambdaMax:
				return "Success (Exceeded Maximum Lambda)";
			case Status::TooManyFunctionEvaluation:
				return "Too Many Function Evaluations";
			case Status::MaxItersReached:
				return "Maximum Iterations Reached";
			}
		}

		void outputHeader() {
			//std::cout << "##################################################" << std::endl;
			//std::cout << "################### Simple LM ####################" << std::endl;
			//std::cout << "##################################################" << std::endl;
			std::cout << "--------------------------------------------------" << std::endl;
		}

		void outputFooter() {
			//std::cout << "##################################################" << std::endl;
			std::cout << "--------------------------------------------------" << std::endl;
		}

		void outputIterHeader() {
			std::cout << " Iter";
			std::cout << std::setw(15) << "f";
			std::cout << std::setw(15) << "|df|_2";
			std::cout << std::setw(15) << "Elapsed" << std::endl;
			std::cout << "--------------------------------------------------" << std::endl;
		}

		template<typename Index, typename Scalar>
		void outputIter(const Index iterIdx, const Scalar &fVal, const Scalar &normGrad, const Scalar &elapsedTime) {
			std::cout << std::setw(5) << iterIdx;
			std::cout << std::setw(15) << fVal;
			std::cout << std::setw(15) << normGrad;
			std::cout << std::setw(14) << elapsedTime << "s";
			std::cout << std::endl;
		}
	}

	/**
	* \brief Performs non linear optimization over a non-linear function,
	* using a simple variant of the Levenberg Marquardt algorithm.
	*
	* In each inner iteration:
	* Does a variety of line-searches using current Jacobian,
	* varying lambda each time, as well as lambdaIncrease.
	* This attempts to get to large lambda as fast as possible
	* if we are rejecting, so that the only convergence
	* criterion is large lambda (i.e. we stop when gradient
	* descent with a tiny step cannot reduce the function).
	*
	* Check wikipedia for more information on LM algorithm.
	* http://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm
	*/
	template <typename _FunctorType, bool Verbose = false>
	class SimpleLM {
	public:
		// Lambda initial parameters
		template <typename Scalar>
		struct Lambda {
			Scalar min;
			Scalar max;
			Scalar decrease;
			Scalar increaseBase;
			Scalar init;

			Lambda() 
				: min(1e-12), max(1e10), decrease(2), increaseBase(2), init(1e-4) {
			}
		};

		// LM initial parameters 
		template <typename Scalar, typename Index>
		struct LMParams {
			Lambda<Scalar> lambda;
			Scalar tolFun;
			Index maxIter;
			Index maxFunEv;

			LMParams()
				: lambda(Lambda<Scalar>()), tolFun(1e-8), maxIter(1e6), maxFunEv(1e6) {
			}
		};

		// Optimization parameters - variable
		template <typename Scalar, typename Index>
		struct OptimParams {
			Scalar lambda;
			Scalar lambdaIncrease;
			Index funEvals;
			Index iter;

			void initialize(LMParams<Scalar, Index> &lmParams) {
				this->lambda = lmParams.lambda.init;
				this->lambdaIncrease = lmParams.lambda.increaseBase;
				this->funEvals = 0;
				this->iter = 0;
			}
		};

		typedef _FunctorType FunctorType;
		typedef typename FunctorType::QRSolver QRSolver;
		typedef typename FunctorType::JacobianType JacobianType;
		typedef typename JacobianType::Scalar Scalar;
		typedef typename JacobianType::StorageIndex StorageIndex;
		typedef typename JacobianType::RealScalar RealScalar;
		typedef typename QRSolver::PermutationMatrixType PermutationType;
		typedef typename FunctorType::InputType InputType;
		typedef typename FunctorType::ValueType ValueType;
		typedef typename FunctorType::StepType StepType;

		// How many values to keep in energy history
		const Index EnergyHistorySize = 2;

		SimpleLM(FunctorType &functor)
			: m_functor(functor),
			m_status(SimpleLMInfo::Status::NotStarted) {
		}
		
		LMParams<Scalar, StorageIndex>& lmParams() {
			return this->m_lmParams;
		}

		SimpleLMInfo::Status minimize(InputType &x) {
			if(Verbose) {
				SimpleLMInfo::outputHeader();
			}

			// Initialize optimization parameters
			m_optParams.initialize(m_lmParams);

			// Intialize QR solver
			m_functor.initQRSolver(m_solver);

			// Allocate variables
			Index nc = m_functor.inputs();
			Index nr = m_functor.values();
			m_residuals.resize(nr);
			m_jacobian.resize(nr, nc);
			m_jacLambda.resize(nr + nc, nc);
			m_jacLambda.setZero();

			// Allocate vector of history fvals
			m_energyHistory = std::vector<Scalar>(EnergyHistorySize, Scalar(0));

			SparseMatrix<Scalar, RowMajor, StorageIndex> I(m_jacobian.cols(), m_jacobian.cols());
			I.setIdentity();

			// Iterate
			m_status = SimpleLMInfo::Status::Running;
			ValueType JtRes, jacColNorms;
			InputType xTest;
			bool stopNow = false;
			
			if (Verbose) {
				SimpleLMInfo::outputIterHeader();
			}
			clock_t iterStart;
			while (true) {
				if (Verbose) {
					iterStart = clock();
				}

				// Another iteration
				m_optParams.iter++;

				// If max iters were reached
				if (m_optParams.iter > m_lmParams.maxIter) {
					m_status = SimpleLMInfo::Status::MaxItersReached;
					break;
				}
				// Too many function evaluatoins
				if (m_optParams.funEvals > m_lmParams.maxFunEv) {
					m_status = SimpleLMInfo::Status::TooManyFunctionEvaluation;
					break;
				}

				// Evaluate functions (residuals)
				m_functor(x, m_residuals);
				m_optParams.funEvals++;

				// Compute sum of squares of the residual
				m_energy = m_residuals.squaredNorm();

				// Compute Jacobian
				m_functor.df(x, m_jacobian);

				// Compute this for later (see computation of "rho" in case of successful iteration)
				JtRes = m_jacobian.transpose() * m_residuals;

				// Compute gradient norm estimate 
				jacColNorms.resize(m_jacobian.cols());
				for (int c = 0; c < m_jacobian.cols(); c++) {
					jacColNorms(c) = m_jacobian.col(c).blueNorm();
				}
				m_normEst = jacColNorms.mean();
	
				// Prepare copy of x for testing
				xTest = x;
				while(true) {
					// Run QR on J with lambdas
					/// | J      | = | e |
					/// | l_diag | = | 0 |
					// Rowpermute the diagonal lambdas into Jacobian
					// Always place lambda below the last element of each column
					QRSolver::PermutationMatrixType rowPerm(m_jacobian.rows() + m_jacobian.cols());
					Index currRow = 0;
					for (Index c = 0; c < m_jacobian.cols(); c++) {
						JacobianType::InnerIterator colIt(m_jacobian, c);
						Index lastNnzIdx = 0;
						while (++colIt) { lastNnzIdx = colIt.index(); }

						// Don't permute the nnz elements in the column
						while (currRow <= lastNnzIdx + c) {
							rowPerm.indices()(currRow - c) = currRow;
							currRow++;
						}
						// Put current diagonal element on this position
						rowPerm.indices()(m_jacobian.rows() + c) = currRow;
						currRow++;
					}
					// Create concatenation of the Jacobian with the diagonal matrix of lambdas
					SparseMatrix<Scalar, RowMajor, StorageIndex> jacRm(m_jacLambda);
					jacRm.topRows(m_jacobian.rows()) = m_jacobian;
					jacRm.bottomRows(m_jacobian.cols()) = I * m_optParams.lambda;
					jacRm = rowPerm * jacRm;
					m_jacLambda = JacobianType(jacRm);

					// Perform QR decomposition of the modified jacobian
					// Ax = b -> QRx = b
					m_solver.compute(m_jacLambda);
					// Append zeros corresponding to lambdas to the residual vector and rowpermute
					ValueType resTmp(nr + nc);
					resTmp.setZero();
					for (int r = 0; r < nr; r++) {
						resTmp(rowPerm.indices()(r)) = m_residuals(r);
					}

					// Compute Q.T * b
					ValueType qtb = m_solver.matrixQ().transpose() * resTmp;

					// And compute the step
					m_dx = m_solver.matrixR().topLeftCorner(m_jacobian.cols(), m_jacobian.cols()).template triangularView<Upper>().solve(qtb.head(m_jacobian.cols()));
				
					// Permute the updates so that the vector ordering corresponds to the ordering of x
					m_dx = m_solver.colsPermutation() * m_dx;
										
					// Compute new step test values
					// Perform test update
					m_dx = -m_dx;	// We want to go in the negative gradient direction
					m_functor.increment_in_place(&xTest, m_dx);
					// Test residuals
					ValueType residualsTest(nr);
					m_functor(xTest, residualsTest);
					m_optParams.funEvals++;
					// Test energy
					Scalar energyTest = residualsTest.squaredNorm();

					// Decide what to do next
					if (energyTest < m_energy) {
						Scalar rhoScale = m_dx.transpose() * (m_optParams.lambda * m_dx + JtRes);
						Scalar rho = (m_energy - energyTest) / rhoScale;
						Scalar lambdaMul = 1.0 - std::pow(2.0 * rho - 1.0, 3);
						
						m_optParams.lambda *= std::max(1.0 / 3.0, lambdaMul);
						m_optParams.lambda = std::max(m_optParams.lambda, m_lmParams.lambda.min);

						// Reset lambda incraese
						m_optParams.lambdaIncrease = m_lmParams.lambda.increaseBase;
						// Update current fval
						m_energy = energyTest;
						// Insert to the history
						m_energyHistory[m_optParams.iter % EnergyHistorySize] = m_energy;
						// Break here
						break;
					} else {
						if (m_optParams.lambda > m_lmParams.lambda.max) {
							m_status = SimpleLMInfo::Status::ExceededLambdaMax;
							stopNow = true;
							break;
						}
						// Increase lambda
						m_optParams.lambda *= m_optParams.lambdaIncrease;

						// Successive rejects should be exponential, so now lambdaIncrease 
						// becomes 1e1, 1e2, 1e4, 1e8, ...
						m_optParams.lambdaIncrease = std::pow(m_optParams.lambdaIncrease, 1.5);
					}
				}

				// If stop was requested, stop it!
				if (stopNow) {
					break;
				}

				// Check for energy flatlining
				if (m_optParams.iter > EnergyHistorySize) {
					Scalar meanf = 0.0;
					for (auto it = m_energyHistory.begin(); it != m_energyHistory.end(); ++it) {
						meanf += *it;
					}
					meanf /= m_energyHistory.size();
					if (std::abs(m_energy - meanf) < m_lmParams.tolFun * m_energy) {
						m_status = SimpleLMInfo::Status::Success;
						break;
					}
				}

				if (Verbose) {
					SimpleLMInfo::outputIter<Index, Scalar>(m_optParams.iter, m_energy, m_normEst, double(clock() - iterStart) / CLOCKS_PER_SEC);
				}

				// After successive iteration, record new x and loop
				x = xTest;
			}

			if (Verbose) {
				SimpleLMInfo::outputFooter();
			}

			return m_status;
		}

	private:
		// Optimization functor
		FunctorType m_functor;
		// QR solver
		QRSolver m_solver;
		// Jacobian matrix
		JacobianType m_jacobian;
		// Jacobian matrix combined with diagonal matrix of lambdas
		JacobianType m_jacLambda;
		// Residuals
		ValueType m_residuals;
		// Current function value (objective)
		Scalar m_energy;
		// Gradient norm estimate
		Scalar m_normEst;
		// Step
		StepType m_dx;

		// History of fvals
		std::vector<Scalar> m_energyHistory;

		// LM paramters
		LMParams<Scalar, StorageIndex> m_lmParams;
		// Optimization parameters
		OptimParams<Scalar, StorageIndex> m_optParams;

		// LM status
		SimpleLMInfo::Status m_status;
	};
}

#endif