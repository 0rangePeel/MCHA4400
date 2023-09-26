#include <Eigen/Core>
#include "Event.h"
#include "State.h"
#include "funcmin.hpp"
#include "Measurement.h"

Measurement::Measurement(double time, const Eigen::VectorXd & y)
    : Event(time)
    , useQuasiNewton(true)
    , y_(y)
{}

Measurement::Measurement(double time, const Eigen::VectorXd & y, const Gaussian<double> & noise)
    : Event(time)
    , useQuasiNewton(true)
    , y_(y)
    , noise_(noise)
{}

double Measurement::costJointDensity(const Eigen::VectorXd & x, const State & state)
{
    double logprior = state.density.log(x);
    double loglik = logLikelihood(state, x);
    return -(logprior + loglik);
}

double Measurement::costJointDensity(const Eigen::VectorXd & x, const State & state, Eigen::VectorXd & g)
{
    Eigen::VectorXd logpriorGrad(x.size());
    double logprior = state.density.log(x, logpriorGrad);
    Eigen::VectorXd loglikGrad(x.size());
    double loglik = logLikelihood(state, x, loglikGrad);
    g = -(logpriorGrad + loglikGrad);
    return -(logprior + loglik);
}

double Measurement::costJointDensity(const Eigen::VectorXd & x, const State & state, Eigen::VectorXd & g, Eigen::MatrixXd & H)
{
    Eigen::VectorXd logpriorGrad(x.size());
    Eigen::MatrixXd logpriorHess(x.size(),x.size());
    double logprior = state.density.log(x, logpriorGrad, logpriorHess);

    Eigen::VectorXd loglikGrad(x.size());
    Eigen::MatrixXd loglikHess(x.size(),x.size());
    double loglik = logLikelihood(state, x, loglikGrad, loglikHess);

    g = -(logpriorGrad + loglikGrad);
    H = -(logpriorHess + loglikHess);
    return -(logprior + loglik);
}

#include <Eigen/SVD>

void Measurement::update(State & state)
{
    const Eigen::Index n = state.size();
    Eigen::MatrixXd Q(n, n);
    Eigen::VectorXd v(n);
    Eigen::VectorXd g(n);
    Eigen::VectorXd x = state.density.mean(); // Set initial decision variable to prior mean
    Eigen::MatrixXd & S = state.density.sqrtCov();

    constexpr int verbosity = 0; // 0:none, 1:dots, 2:summary, 3:iter
    //bool useQuasiNewton = 0;
    if (useQuasiNewton)
    {
        // Generate eigendecomposition of initial Hessian (inverse of prior covariance)
        // via an SVD of S = U*D*V.', i.e., (S.'*S)^{-1} = (V*D*U.'*U*D*V.')^{-1} = V*D^{-2}*V.'
        // This avoids the loss of precision associated with directly computing the eigendecomposition of (S.'*S)^{-1}
        //Eigen::JacobiSVD<Eigen::MatrixXd> svd(S, Eigen::ComputeFullV);
        //v = svd.singularValues().array().square().inverse();
        //Q = svd.matrixV();



        // Current      : If we were doing landmark SLAM with a quasi-Newton method,
        //                we can purposely introduce negative eigenvalues for newly
        //                initialised landmarks to force the Hessian and hence initial
        //                covariance to be approximated correctly.

        // Create J vector and place -1 for each newLandmark states
        Eigen::VectorXd J(n);
        J.fill(1.0);
        for (int i = 0; i < state.numNewLandmarks*6; i++) {
            J(n - i - 1) = -1;
        }

        // Eigen Decomposition
        // Lecture Landmark SLAM 1 p17
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenH(S.transpose()*J.asDiagonal()*S);
        v = eigenH.eigenvalues().array().inverse();
        Q = eigenH.eigenvectors();

        assert(Q.rows() == n);
        assert(Q.cols() == n);
        assert(v.size() == n);

        // Create cost function with prototype V = costFunc(x, g)
        auto costFunc = [&](const Eigen::VectorXd & x, Eigen::VectorXd & g){ return costJointDensity(x, state, g); };

        // Minimise cost
        int ret = funcmin::SR1TrustEig(costFunc, x, g, Q, v, verbosity);
        assert(ret == 0);
    }
    else
    {
        // Create cost function with prototype V = costFunc(x, g, H)
        auto costFunc = [&](const Eigen::VectorXd & x, Eigen::VectorXd & g, Eigen::MatrixXd & H){ return costJointDensity(x, state, g, H); };

        // Minimise cost
        int ret = funcmin::NewtonTrustEig(costFunc, x, g, Q, v, verbosity);
        assert(ret == 0);
    }
    // Set posterior mean to maximum a posteriori (MAP) estimate
    state.density.mean() = x;
    // Post-calculate posterior square-root covariance from Hessian eigendecomposition
    S = v.array().rsqrt().matrix().asDiagonal()*Q.transpose();
    Eigen::HouseholderQR<Eigen::Ref<Eigen::MatrixXd>> qr(S);    // In-place QR decomposition
    S = S.triangularView<Eigen::Upper>();                       // Safe aliasing   
}
