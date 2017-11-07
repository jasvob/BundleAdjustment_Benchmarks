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

	template <typename _FunctorType>
	class SimpleLM {
	public:
		enum Status {
			NotStarted = -2,
			Running = -1,
			Success = 0,
			TooManyFunctionEvaluation = 1,
			MaxItersReached = 2,
			ExceededLambdaMax = 3
		};

		template <typename Scalar>
		struct LMParams {
			Scalar lambdaMin;
			Scalar lambdaMax;
			Scalar lambdaDecrease;
			Scalar lambdaIncreaseBase;
			Scalar lambdaInit;
			Scalar tolFun;

			LMParams()
				: lambdaMin(1e-12), lambdaMax(1e10), lambdaDecrease(2), lambdaIncreaseBase(10), lambdaInit(1e-4), tolFun(1e-8) {
			}
		};

		template <typename Scalar, typename Index>
		struct OptimParams {
			Scalar lambda;
			Scalar lambdaIncrease;
			Index funEvals;
			Index iter;

			void initialize(LMParams<Scalar> &lmParams) {
				this->lambda = lmParams.lambdaInit;
				this->lambdaIncrease = lmParams.lambdaIncreaseBase;
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

		const Index HistorySize = 10;

		SimpleLM(FunctorType &functor)
			: m_functor(functor),
			m_maxIter(1e6),
			m_maxFunEv(1e6),
			m_fdDelta(1e-5),
			m_status(Status::NotStarted) {

		}

		Status minimize(InputType &x) {
			// Initialize optimization parameters
			m_optParams.initialize(m_lmParams);

			// Intialize QR solver
			m_functor.initQRSolver(m_solver);

			// Allocate variables
			Index n = m_functor.inputs();
			Index m = m_functor.values();
			m_residuals.resize(m);
			m_jacobian.resize(m, n);
			m_jacLambda.resize(m + n, n);
			m_jacLambda.setZero();

			// Allocate vector of history fvals
			m_fValHistory = std::vector<Scalar>(10, Scalar(0));

			SparseMatrix<Scalar, RowMajor, StorageIndex> I(m_jacobian.cols(), m_jacobian.cols());
			I.setIdentity();

			// Iterate
			m_status = Status::Running;
			ValueType JtRes;
			bool stopNow = false;
			while (true) {
				// Another iteration
				m_optParams.iter++;

				// If max iters were reached
				if (m_optParams.iter > m_maxIter) {
					m_status = Status::MaxItersReached;
					break;
				}
				// Too many function evaluatoins
				if (m_optParams.funEvals > m_maxFunEv) {
					m_status = Status::TooManyFunctionEvaluation;
					break;
				}

				// Evaluate functions (residuals)
				m_functor(x, m_residuals);
				m_optParams.funEvals++;

				// Compute sum of squares of the residual
				m_fVal = m_residuals.squaredNorm();	

				// Compute Jacobian
				m_functor.df(x, m_jacobian);

				JtRes = m_jacobian.transpose() * m_residuals;

				// Compute norm estimate and preconditioner for PCG
				m_colNorms.resize(m_jacobian.cols());
				for (int c = 0; c < m_jacobian.cols(); c++) {
					m_colNorms(c) = m_jacobian.col(c).blueNorm();
				}
				//m_colNorms = m_jacobian.colwise().blueNorm();	// There's no colwise for sparse matrices ...
				m_normEst = m_colNorms.mean();

				// Prepare copy of x for testing
				InputType xTest = x;
				while(true) {
					// Run QR on J with lambdas
					/// | J      | = | e |
					/// | l_diag | = | 0 |
					//VectorType diagLambda = VectorType::Ones(m_jacobian.cols());
					//diagLambda *= m_optParams.lambda;
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
					// Conservative resize jacobian so that it can be concatenated with the new rows
					//m_jacobian.conservativeResize(m_jacobian.rows() + m_jacobian.cols(), m_jacobian.cols());
					SparseMatrix<Scalar, RowMajor, StorageIndex> jacRm(m_jacLambda);
					jacRm.topRows(m_jacobian.rows()) = m_jacobian;
					jacRm.bottomRows(m_jacobian.cols()) = I * m_optParams.lambda;
					jacRm = rowPerm * jacRm;
					m_jacLambda = JacobianType(jacRm);

					// Perform QR decomposition of the modified jacobian
					// Ax = b -> QRx = b
					m_solver.compute(m_jacLambda);
					// Append zeros corresponding to lambdas to the residual vector and rowpermute
					ValueType resTmp(m + n);
					resTmp.setZero();
					for (int r = 0; r < m; r++) {
						resTmp(rowPerm.indices()(r)) = m_residuals(r);
					}
					// Compute Q.T * b
					ValueType qtb = m_solver.matrixQ().transpose() * resTmp;
					qtb.head(n) = m_solver.colsPermutation() * qtb.head(n);
					// And compute the step
					m_dx = m_solver.matrixR().topLeftCorner(m_jacobian.cols(), m_jacobian.cols()).template triangularView<Upper>().solve(qtb.head(m_jacobian.cols()));

					// Compute new step test values
					// Perform test update
					//m_dx = -m_dx;
					m_functor.increment_in_place(&xTest, m_dx);
					// Test residuals
					ValueType residualsTest(m);
					m_functor(xTest, residualsTest);
					m_optParams.funEvals++;
					// Test energy
					Scalar fValTest = residualsTest.squaredNorm();

					// Decide what to do next
					if (fValTest < m_fVal) {
						Scalar rhoScale = m_dx.transpose() * (m_optParams.lambda * m_dx + JtRes);
						Scalar rho = (m_fVal - fValTest) / rhoScale;
						Scalar lambdaMul = 1 - (2 * rho - 1);
						lambdaMul = lambdaMul * lambdaMul * lambdaMul;

						m_optParams.lambda *= std::max(1.0 / 3.0, lambdaMul);
						m_optParams.lambda = std::max(m_optParams.lambda, m_lmParams.lambdaMin);

						// Reset lambda incraese
						m_optParams.lambdaIncrease = 2; // m_lmParams.lambdaIncreaseBase
						// Update current fval
						m_fVal = fValTest;
						// Insert to the history
						m_fValHistory[m_optParams.iter % HistorySize] = m_fVal;
						// Break here
						break;
					} else {
						if (m_optParams.lambda > m_lmParams.lambdaMax) {
							m_status = Status::ExceededLambdaMax;
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

				// Check cgce
				if (m_optParams.iter > HistorySize) {
					Scalar meanf = 0.0;
					for (auto it = m_fValHistory.begin(); it != m_fValHistory.end(); ++it) {
						meanf += *it;
					}
					meanf /= m_fValHistory.size();
					if (std::abs(m_fVal - meanf) < m_lmParams.tolFun * m_fVal) {
						m_status = Status::Success;
						break;
					}
				}

				// After successive iteration, record new x and loop
				x = xTest;
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
		JacobianType m_jacLambda;
		// Residuals
		ValueType m_residuals;
		// Current function value (objective)
		Scalar m_fVal;
		// Jacobian column norms 
		ValueType m_colNorms;
		// Norm estimate
		Scalar m_normEst;
		// Step
		StepType m_dx;

		// History of fvals
		std::vector<Scalar> m_fValHistory;

		StorageIndex m_maxIter;
		StorageIndex m_maxFunEv;
		Scalar m_fdDelta;

		// LM paramters
		LMParams<Scalar> m_lmParams;
		// Optimization parameters
		OptimParams<Scalar, StorageIndex> m_optParams;

		// LM status
		Status m_status;
	};
}

#endif