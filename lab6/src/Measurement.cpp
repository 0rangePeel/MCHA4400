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
    double loglik = logLikelihood(x);
    return -(logprior + loglik);
}

double Measurement::costJointDensity(const Eigen::VectorXd & x, const State & state, Eigen::VectorXd & g)
{
    Eigen::VectorXd logpriorGrad(x.size());
    double logprior = state.density.log(x, logpriorGrad);

    Eigen::VectorXd loglikGrad(x.size());
    double loglik = logLikelihood(x, loglikGrad);

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
    double loglik = logLikelihood(x, loglikGrad, loglikHess);

    g = -(logpriorGrad + loglikGrad);
    H = -(logpriorHess + loglikHess);
    return -(logprior + loglik);
}

#include <Eigen/SVD>

void Measurement::update(State & state)
{
    const Eigen::Index n = state.size();
    Eigen::MatrixXd Q(n,n);
    Eigen::VectorXd v(n);
    Eigen::VectorXd g(n);
    Eigen::VectorXd x = state.density.mean(); // Set initial decision variable to prior mean
    Eigen::MatrixXd & S = state.density.sqrtCov();

    constexpr int verbosity = 1; // 0:none, 1:dots, 2:summary, 3:iter
    if (useQuasiNewton)
    {
        // Generate eigendecomposition of initial Hessian (inverse of prior covariance)
        // via an SVD of S = U*D*V.', i.e., (S.'*S)^{-1} = (V*D*U.'*U*D*V.')^{-1} = V*D^{-2}*V.'
        // This avoids the loss of precision associated with directly computing the eigendecomposition of (S.'*S)^{-1}
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(S, Eigen::ComputeFullV);
        v = svd.singularValues().array().square().inverse();
        Q = svd.matrixV();

        assert(Q.rows() == n);
        assert(Q.cols() == n);
        assert(v.size() == n);

        // Foreshadowing: If we were doing landmark SLAM with a quasi-Newton method,
        //                we can purposely introduce negative eigenvalues for newly
        //                initialised landmarks to force the Hessian and hence initial
        //                covariance to be approximated correctly.

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

Eigen::VectorXd Measurement::simulate(const Eigen::VectorXd & x) const
{
    Gaussian p(predict(x), noise_.sqrtCov());
    return p.simulate();
}

double Measurement::logLikelihood(const Eigen::VectorXd & x) const
{
    Eigen::VectorXd h = predict(x);
    Gaussian likelihood(h, noise_.sqrtCov());
    
    // Evaluate log N(y; h(x), R)
    double logLik = likelihood.log(y_);
    return logLik;
}

double Measurement::logLikelihood(const Eigen::VectorXd & x, Eigen::VectorXd & g) const
{
    // i) Analyical derivatives
    Eigen::MatrixXd dhdx; // this is J from MeasurementRADAR.cpp
    Eigen::VectorXd h = predict(x, dhdx);
    Gaussian likelihood(h, noise_.sqrtCov());

    // Evaluate log N(y; h(x), R) and d/dy log N(y; h(x), R)
    Eigen::VectorXd loglikGrad;
    double logLik = likelihood.log(y_, loglikGrad);
    // Note:
    //  d                        d     
    // -- log N(y; h(x), R) = - -- log N(y; h(x), R)
    // dh                       dy

    // Gradient of log likelihood:
    //
    //         d 
    // g_i = ---- log N(y; h(x), R)
    //       dx_i
    //
    //             dh_k     d
    // g_i = sum_k ---- * ---- log N(y; h(x), R)
    //             dx_i   dh_k
    //
    //               dh_k     d
    // g_i = - sum_k ---- * ---- log N(y; h(x), R)
    //               dx_i   dy_k
    //

    g = -dhdx.transpose() * loglikGrad;

    return logLik;
}

double Measurement::logLikelihood(const Eigen::VectorXd & x, Eigen::VectorXd & g, Eigen::MatrixXd & H) const
{
    // i) Analyical derivatives
    Eigen::MatrixXd dhdx;
    Eigen::Tensor<double, 3> d2hdx2;
    Eigen::VectorXd h = predict(x, dhdx, d2hdx2);
    Gaussian likelihood(h, noise_.sqrtCov());

    // Evaluate log N(y; h(x), R), d/dy log N(y; h(x), R) and d^2/dy^2 log N(y; h(x), R)
    Eigen::VectorXd loglikGrad;
    Eigen::MatrixXd logLikHess;
    double logLik = likelihood.log(y_, loglikGrad, logLikHess);
    // Note:
    //  d                        d     
    // -- log N(y; h(x), R) = - -- log N(y; h(x), R)
    // dh                       dy
    //
    //  d^2                      d^2                  
    // ---- log N(y; h(x), R) = ---- log N(y; h(x), R)
    // dh^2                     dy^2                  

    // Gradient of log likelihood:
    //
    //         d 
    // g_i = ---- log N(y; h(x), R)
    //       dx_i
    //
    //             dh_k     d
    // g_i = sum_k ---- * ---- log N(y; h(x), R)
    //             dx_i   dh_k
    //
    //               dh_k     d
    // g_i = - sum_k ---- * ---- log N(y; h(x), R)
    //               dx_i   dy_k
    //

    g = -dhdx.transpose() * loglikGrad;

    // Hessian of log likelihood:
    //
    //              d                                 d  ( dh_k     d                    )
    // H_{ij} = --------- log N(y; h(x), R) = sum_k ---- ( ---- * ---- log N(y; h(x), R) )
    //          dx_i dx_j                           dx_j ( dx_i   dh_k                   )
    //
    //                      dh_k   d^2 log N(y; h(x), R)   dh_k          d^2 h_k      d                   
    // H_{ij} = sum_k sum_l ---- * --------------------- * ---- + sum_k --------- * ---- log N(y; h(x), R) 
    //                      dx_i         dh_k dh_l         dx_j         dx_i dx_j   dh_k                   
    //
    //                      dh_k   d^2 log N(y; h(x), R)   dh_k          d^2 h_k      d                  
    // H_{ij} = sum_k sum_l ---- * --------------------- * ---- - sum_k --------- * ---- log N(y; h(x), R)
    //                      dx_i         dy_k dy_l         dx_j         dx_i dx_j   dy_k   
    //
    
    int nx = dhdx.cols();
    int nh = dhdx.rows();
    

    Eigen::MatrixXd linearTerm;
    Eigen::MatrixXd nonlinearTerm;
    Eigen::VectorXd temp(nh);

    H.resize(nx,nx);
    linearTerm.resize(nx,nx);
    nonlinearTerm.resize(nx,nx);

    linearTerm = dhdx.transpose()*logLikHess*dhdx;

    for (int i = 0; i < nx; i++){
        for (int j = 0; j < nx; j++){
            for (int k = 0; k < nh; ++k) {
                temp(k) = d2hdx2(k, i, j) * loglikGrad(k);
            }
            nonlinearTerm(i, j) = temp.sum();        
        }
    }

    H = linearTerm - nonlinearTerm;

    //H(i,j) = d2hdx2(0,i,j) * loglikGrad - dhdx(i) * logLikHess;

    // Hint: In Matlab, this operation would look like the following:
    //       nh = length(h);
    //       nx = length(x);
    //       H = dhdx.'*logLikHess*dhdx - reshape(sum(d2hdx2 .* reshape(loglikGrad, [nh, 1, 1]), 1), [nx, nx]);

    return logLik;
}
