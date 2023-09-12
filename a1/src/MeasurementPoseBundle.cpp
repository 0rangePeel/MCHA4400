#include <cstddef>
#include <numeric>
#include <vector>
#include <Eigen/Core>
#include "State.h"
#include "Camera.h"
#include "StateSLAM.h"
#include "StateSLAMPoseLandmarks.h"
#include "MeasurementPoseBundle.h"

#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>

MeasurementPoseBundle::MeasurementPoseBundle(double time, const Eigen::VectorXd & y, const Camera & camera)
    : Measurement(time, y)
    , camera_(camera)
{
    // SR is an upper triangular matrix such that SR.'*SR = R is the measurement noise covariance
    const Eigen::Index & ny = y.size();
    Eigen::MatrixXd SR = 1.0*Eigen::MatrixXd::Identity(ny, ny); // TODO: Assignment(s)
    noise_ = Gaussian(SR);

    // useQuasiNewton = false;
}

double MeasurementPoseBundle::logLikelihood(const State & state, const Eigen::VectorXd & x) const
{
    const StateSLAMPoseLandmarks & stateSLAM = dynamic_cast<const StateSLAMPoseLandmarks &>(state);

    // Select visible landmarks
    std::vector<std::size_t> idxLandmarks(stateSLAM.numberLandmarks());
    // TODO: Assignment(s)
    std::iota(idxLandmarks.begin(), idxLandmarks.end(), 0); // Select all landmarks

    Eigen::VectorXd h = stateSLAM.predictFeatureBundle(x, camera_, idxLandmarks);

    Gaussian likelihood(h, noise_.sqrtCov());
    return likelihood.log(y_);
}

double MeasurementPoseBundle::logLikelihood(const State & state, const Eigen::VectorXd & x, Eigen::VectorXd & g) const
{
    // Evaluate gradient for SR1 and Newton methods
    // TODO: Assignment(s)
    /*
    const StateSLAMPoseLandmarks & stateSLAM = dynamic_cast<const StateSLAMPoseLandmarks &>(state);
    g.resize(x.size());
    g.setZero();
    Eigen::VectorX<autodiff::dual> xdual = x.cast<autodiff::dual>();
    autodiff::dual fdual;
    g = gradient(&MeasurementPoseBundle::logLikelihoodImpl<autodiff::dual>, wrt(xdual), at(this, xdual, camera_, stateSLAM, idxLandmark, j), fdual);
    return val(fdual);
    */

    g.resize(x.size());
    g.setZero();
    return logLikelihood(state, x);
}

double MeasurementPoseBundle::logLikelihood(const State & state, const Eigen::VectorXd & x, Eigen::VectorXd & g, Eigen::MatrixXd & H) const
{
    // Evaluate Hessian for Newton method
    // TODO: Assignment(s)
    /*
    const StateSLAMPoseLandmarks & stateSLAM = dynamic_cast<const StateSLAMPoseLandmarks &>(state);
    H.resize(x.size());
    H.setZero();
    Eigen::VectorX<autodiff::dual> xdual = x.cast<autodiff::dual>();
    autodiff::dual fdual;
    H = gradient(&MeasurementPoseBundle::logLikelihoodImpl<autodiff::dual>, wrt(xdual), at(this, xdual, camera_, stateSLAM, idxLandmark, j), fdual);
    return val(fdual);
    */

    H.resize(x.size(), x.size());
    H.setZero();
    return logLikelihood(state, x, g);
}

void MeasurementPoseBundle::update(State & state)
{
    StateSLAM & stateSLAM = dynamic_cast<StateSLAM &>(state);

    // TODO: Assignment(s)
    // Identify landmarks with matching features (data association)
    // Remove failed landmarks from map (consecutive failures to match)
    // Identify surplus features that do not correspond to landmarks in the map
    // Initialise up to Nmax – N new landmarks from best surplus features
    
    Measurement::update(state);  // Do the actual measurement update
}