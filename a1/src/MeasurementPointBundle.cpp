#include <cstddef>
#include <numeric>
#include <vector>
#include <Eigen/Core>
#include "State.h"
#include "Camera.h"
#include "StateSLAMPointLandmarks.h"
#include "MeasurementPointBundle.h"

MeasurementPointBundle::MeasurementPointBundle(double time, const Eigen::VectorXd & y, const Camera & camera)
    : Measurement(time, y)
    , camera_(camera)
{
    // SR is an upper triangular matrix such that SR.'*SR = R is the measurement noise covariance
    const Eigen::Index & ny = y.size();
    Eigen::MatrixXd SR = 1.0*Eigen::MatrixXd::Identity(ny, ny); // TODO: Assignment(s)
    noise_ = Gaussian(SR);

    // useQuasiNewton = false;
}

double MeasurementPointBundle::logLikelihood(const State & state, const Eigen::VectorXd & x) const
{
    const StateSLAMPointLandmarks & stateSLAM = dynamic_cast<const StateSLAMPointLandmarks &>(state);
    return logLikelihoodImpl(x, stateSLAM, state.getIdxLandmarks());
}

double MeasurementPointBundle::logLikelihood(const State & state, const Eigen::VectorXd & x, Eigen::VectorXd & g) const
{
    // Evaluate gradient for SR1 and Newton methods
    const StateSLAMPointLandmarks & stateSLAM = dynamic_cast<const StateSLAMPointLandmarks &>(state);

    g.resize(x.size());
    g.setZero();
    Eigen::VectorX<autodiff::dual> xdual = x.cast<autodiff::dual>();
    autodiff::dual fdual;
    g = gradient(&MeasurementPointBundle::logLikelihoodImpl<autodiff::dual>, wrt(xdual), at(this, xdual, stateSLAM, state.getIdxLandmarks()), fdual);
    return val(fdual);
}

double MeasurementPointBundle::logLikelihood(const State & state, const Eigen::VectorXd & x, Eigen::VectorXd & g, Eigen::MatrixXd & H) const
{
    // Evaluate Hessian for Newton method  
    const StateSLAMPointLandmarks & stateSLAM = dynamic_cast<const StateSLAMPointLandmarks &>(state);

    H.resize(x.size(), x.size());
    H.setZero();
    Eigen::VectorX<autodiff::dual2nd> xdual = x.cast<autodiff::dual2nd>();
    autodiff::dual2nd fdual;
    H = hessian(&MeasurementPointBundle::logLikelihoodImpl<autodiff::dual2nd>, wrt(xdual), at(this, xdual, stateSLAM, state.getIdxLandmarks()), fdual, g);
    return val(fdual);
}

void MeasurementPointBundle::update(State & state)
{
    StateSLAM & stateSLAM = dynamic_cast<StateSLAM &>(state);

    // TODO: Assignment(s)
    // Identify landmarks with matching features (data association)
    // Remove failed landmarks from map (consecutive failures to match)
    // Identify surplus features that do not correspond to landmarks in the map
    // Initialise up to Nmax – N new landmarks from best surplus features
    
    Measurement::update(state);  // Do the actual measurement update
}