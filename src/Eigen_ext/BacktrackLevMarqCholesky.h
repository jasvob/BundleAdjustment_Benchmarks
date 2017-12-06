//
// Copyright (C) 2017 Jan Svoboda <jan.svoboda@usi.ch>
// Copyright (C) 2016 Andrew Fitzgibbon <awf@microsoft.com>
//
// BacktrackLevMarqCholesky: Simple implementation of Levenberg Marquardt, with backtracking to update trust parameter.
// Useful for large sparse problems, and problems where function evaluation is not much cheaper (maybe not 10x) than
// evaluation of function and Jacobian. For many problems, the Jacobian can be filled in very cheaply at the same time
// as the function is computed (see examples/ellipse_fitting.cpp)
//
// Reimplementation of Matlab template LM imlementation found in AWF Utility Library:
// https://github.com/awf/awful/blob/master/matlab/au_levmarq.m
//

#ifndef BACKTRACK_LEVMARQ_CHOLESKY_H
#define BACKTRACK_LEVMARQ_CHOLESKYH

#include <Eigen/Eigen>
#include <Eigen/Sparse>
#include <iostream>
#include <iomanip>

#include <Eigen/SparseCholesky>
#include <Eigen/src/CholmodSupport/CholmodSupport.h>

namespace Eigen {
  namespace BacktrackLevMarqCholeskyInfo {
    enum Status {
      NotStarted = -2,
      Running = -1,
      Success = 0,
      ExceededLambdaMax = 1,
      TooManyFunctionEvaluation = 2,
      MaxItersReached = 3
    };

    inline std::string statusToString(const Status &status) {
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

    inline void outputHeader() {
      std::cout << "############################## Backtrack LevMarq ###############################" << std::endl;
      std::cout << "--------------------------------------------------------------------------------" << std::endl;
    }

    inline void outputFooter() {
      std::cout << "--------------------------------------------------------------------------------" << std::endl;
    }

    inline void outputIterHeader() {
      std::cout << " Iter";
      std::cout << std::setw(15) << "Status";
      std::cout << std::setw(15) << "f";
      std::cout << std::setw(15) << "rho";
      std::cout << std::setw(15) << "lambda";
      std::cout << std::setw(15) << "Elapsed" << std::endl;
      std::cout << "--------------------------------------------------------------------------------" << std::endl;
    }

    template<typename Index, typename Scalar>
    void outputIter(const Index iterIdx, const std::string &status, const Scalar &fVal, const Scalar &normGrad, const Scalar &lambda, const Scalar &elapsedTime) {
      std::cout << std::setw(5) << iterIdx;
      std::cout << std::setw(15) << status;
      std::cout << std::setw(15) << fVal;
      std::cout << std::setw(15) << normGrad;
      std::cout << std::setw(15) << lambda;
      std::cout << std::setw(14) << elapsedTime << "s";
      std::cout << std::endl;
    }
  }

  /**
  * \brief Performs non linear optimization over a non-linear function,
  * using a simple variant of the Levenberg Marquardt algorithm.
  *
  * This implements a "classic" Levenberg-Marquardt algorithm as described in "Numerical Recipes", 
  * on Wikipedia [https://en.wikipedia.org/wiki/Levenberg-Marquardt_algorithm] and widely implemented in computer vision applications. 
  * Every step solves a regularized system of normal equations, and the regularization 
  * parameter is adjusted depending on the rate of increase/decrease of the objective function.
  * It is based on QR decomposition of the augmented Jacobian [J; lambda D] rather 
  * than Cholesky decomposition of J'J + \lambda^2 D'D, so avoids squaring the condition number.
  * For large structured sparse problems, it may be faster than TrustRegionLevMarq. See examples/ellipse_fitting.cpp for usage.
  *
  */
  template <typename _FunctorType, bool Verbose = false>
  class BacktrackLevMarqCholesky {
  public:
    typedef _FunctorType FunctorType;
    typedef typename FunctorType::JacobianType JacobianType;
    typedef typename JacobianType::Scalar Scalar;
    typedef typename JacobianType::StorageIndex StorageIndex;
    typedef typename JacobianType::RealScalar RealScalar;
    typedef typename FunctorType::InputType InputType;
    typedef typename FunctorType::ValueType ValueType;
    typedef typename FunctorType::StepType StepType;

    // Lambda initial parameters
    struct Lambda {
      Scalar minVal;
      Scalar maxVal;
      Scalar decrease;
      Scalar increaseBase;
      Scalar init;

      Lambda() 
        : minVal(1e-10), maxVal(1e10), decrease(10), increaseBase(2), init(1e-3) {
      }
    };

    // LM initial parameters 
    struct LMParams {
      Lambda lambda;
      Scalar tolFun;
      int maxIter;
      int maxFunEv;

    LMParams()
        : lambda(Lambda()), tolFun(1e-8), maxIter(1e6), maxFunEv(1e6) {
      }
    };

    // Optimization parameters - variable
    struct OptimParams {
      Scalar lambda;
      Scalar lambdaIncrease;
      int funEvals;
      int iter;

      void initialize(LMParams &lmParams) {
        this->lambda = lmParams.lambda.init;
        this->lambdaIncrease = lmParams.lambda.increaseBase;
        this->funEvals = 0;
        this->iter = 0;
      }
    };

    // How many values to keep in energy history
    const Index EnergyHistorySize = 2;

  private:
    // Optimization functor
    FunctorType m_functor;
    // QR solver
    Eigen::SimplicialLDLT<JacobianType, Eigen::Lower> m_solver;
    // Jacobian matrix
    JacobianType m_jacobian;
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
    LMParams m_lmParams;
    // Optimization parameters
    OptimParams m_optParams;

    // LM status
    BacktrackLevMarqCholeskyInfo::Status m_status;

  public:

    BacktrackLevMarqCholesky(FunctorType &functor)
      : m_functor(functor),
      m_status(BacktrackLevMarqCholeskyInfo::Status::NotStarted) {
    }
    
    LMParams& lmParams() {
      return this->m_lmParams;
    }

    BacktrackLevMarqCholeskyInfo::Status minimize(InputType &x) {
      if(Verbose) {
        BacktrackLevMarqCholeskyInfo::outputHeader();
      }

      // Initialize optimization parameters
      m_optParams.initialize(m_lmParams);

      // Allocate variables
      Index nParams = m_functor.inputs();
      Index nResiduals = m_functor.values();
      m_residuals.resize(nResiduals);
      m_jacobian.resize(nResiduals, nParams);
      m_dx.resize(nParams);

      // Allocate vector of history fvals
      m_energyHistory = std::vector<Scalar>(EnergyHistorySize, Scalar(0));

      SparseMatrix<Scalar, RowMajor, StorageIndex> I(nParams, nParams);
      I.setIdentity();

      // Iterate
      m_status = BacktrackLevMarqCholeskyInfo::Status::Running;
      InputType xTest;
      bool stopNow = false;
      
      if (Verbose) {
        BacktrackLevMarqCholeskyInfo::outputIterHeader();
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
          m_status = BacktrackLevMarqCholeskyInfo::Status::MaxItersReached;
          break;
        }
        // Too many function evaluatoins
        if (m_optParams.funEvals > m_lmParams.maxFunEv) {
          m_status = BacktrackLevMarqCholeskyInfo::Status::TooManyFunctionEvaluation;
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
        ValueType JtRes = -m_jacobian.transpose() * m_residuals;

        // Compute gradient norm estimate 
        ValueType jacColNorms(nParams);
        for (int c = 0; c < nParams; c++) {
          jacColNorms(c) = m_jacobian.col(c).blueNorm();
        }
        m_normEst = jacColNorms.mean();
  
        // Create concatenation of the Hessian augmented with the diagonal matrix of lambdas
        JacobianType JtJ = m_jacobian.transpose() * m_jacobian;
        // Set initial lambda based on tau * max(J_colnorm)
        // Assume tau = 1e-6
        if(m_optParams.iter == 1) {
          m_optParams.lambda = 1e-12 * JtJ.diagonal().maxCoeff(); //FixMe: jasvob - or 1e-3?
        }

        while(true) {
          if (Verbose) {
            iterStart = clock();
          }

          // Run Cholesky on J'J with lambdas
          /// | J'J + lambda * I | = | e |
          JacobianType Il = I * m_optParams.lambda;
          JacobianType JtJl = JtJ + Il;

          // Perform Cholesky factorization of the modified Hessian
          m_solver.compute(JtJl);

          // And compute the step
          m_dx.setZero();
          m_dx = m_solver.solve(JtRes);

          // Permute the updates so that the vector ordering corresponds to the ordering of x
          m_dx = m_solver.permutationP() * m_dx;

          // Compute new step test values
          xTest = x;
          // Perform test update
          m_functor.increment_in_place(&xTest, m_dx);
          // Test residuals
          ValueType residualsTest(nResiduals);
          m_functor(xTest, residualsTest);
          m_optParams.funEvals++;
          // Test energy
          Scalar energyTest = residualsTest.squaredNorm();

          // Decide what to do next
          if (energyTest < m_energy) {
            Scalar rhoScale = m_dx.transpose() * (m_optParams.lambda * m_dx + JtRes);
            Scalar rho = (m_energy - energyTest) / rhoScale;

            Scalar lambdaMul = Scalar(1.0) - std::pow(Scalar(2.0) * rho - Scalar(1.0), Scalar(3.0));
            m_optParams.lambda *= std::max<Scalar>(Scalar(1.0) / Scalar(3.0), lambdaMul);
            m_optParams.lambda = std::max<Scalar>(m_optParams.lambda, m_lmParams.lambda.minVal);

			      if (Verbose) {
				      BacktrackLevMarqCholeskyInfo::outputIter<Index, Scalar>(m_optParams.iter, "Accepted", m_energy, rho, m_optParams.lambda, double(clock() - iterStart) / CLOCKS_PER_SEC);
			      }

			      // Reset lambda increase
			      m_optParams.lambdaIncrease = m_lmParams.lambda.increaseBase;
            // Update current fval
            m_energy = energyTest;
            // Insert to the history
            m_energyHistory[m_optParams.iter % EnergyHistorySize] = m_energy;

            // Break here
            break;
          } else {
            if (Verbose) {
              BacktrackLevMarqCholeskyInfo::outputIter<Index, Scalar>(m_optParams.iter, "Rejected", m_energy, 0, m_optParams.lambda, double(clock() - iterStart) / CLOCKS_PER_SEC);
            }

            if (m_optParams.lambda > m_lmParams.lambda.maxVal) {
              m_status = BacktrackLevMarqCholeskyInfo::Status::ExceededLambdaMax;
              stopNow = true;
              break;
            }
            // Increase lambda
            m_optParams.lambda *= m_optParams.lambdaIncrease;

            // Successive rejects grow exponentially
            m_optParams.lambdaIncrease = std::pow(m_optParams.lambdaIncrease, 1.5);
          }
        }

        // If stop was requested, stop it!
        if (stopNow) {
          break;
        }

        // Check for energy flatlining
        if (m_optParams.iter > EnergyHistorySize) {
          Scalar maxf = *(std::max_element(m_energyHistory.begin(), m_energyHistory.end()));
          if (std::abs(m_energy - maxf) < m_lmParams.tolFun * m_energy) {
            m_status = BacktrackLevMarqCholeskyInfo::Status::Success;
            break;
          }
        }

        // After successive iteration, record new x and loop
        x = xTest;
      }

      if (Verbose) {
        BacktrackLevMarqCholeskyInfo::outputFooter();
      }

      return m_status;
    }
  };
}

#endif