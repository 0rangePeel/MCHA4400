#include <cstddef>
#include <numeric>
#include <vector>
#include <Eigen/Core>
#include "rotation.hpp"
#include "State.h"
#include "Camera.h"
#include "StateSLAM.h"
#include "StateSLAMPoseLandmarks.h"
#include "MeasurementPoseBundle.h"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgcodecs.hpp>

#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>

MeasurementPoseBundle::MeasurementPoseBundle(double time, const Eigen::VectorXd & y, const Camera & camera)
    : Measurement(time, y)
    , camera_(camera)
{
    
    // SR is an upper triangular matrix such that SR.'*SR = R is the measurement noise covariance
    const Eigen::Index & ny = y.size();
    double rms = 1.27127;
    Eigen::MatrixXd SR = rms*Eigen::MatrixXd::Identity(ny, ny); // TODO: Assignment(s)
    noise_ = Gaussian(SR);

    // useQuasiNewton = false;
    
}

double MeasurementPoseBundle::logLikelihood(const State & state, const Eigen::VectorXd & x) const
{
    const StateSLAMPoseLandmarks & stateSLAM = dynamic_cast<const StateSLAMPoseLandmarks &>(state);

    Pose cameraPose;
    // Extract Thetanb from x
    Eigen::Vector3d thetanb = x.segment(9, 3);
    
    // Get Pose Matrix
    Eigen::Matrix3d Rnb = rpy2rot(thetanb);

    // Fill Pose from camera.h
    Eigen::Matrix3d Rnc = Rnb * camera_.Rbc;
    cv::eigen2cv(Rnc, cameraPose.Rnc);

    Eigen::Vector3d rBNn = x.segment(6, 3);
    cv::eigen2cv(rBNn, cameraPose.rCNn); // "Assume that B and C coincide"

    std::vector<std::size_t> idxLandmarks(stateSLAM.numberLandmarks());
    for (std::size_t j = 0; j < stateSLAM.numberLandmarks(); ++j)
    {
        Eigen::Vector3d murPNn = stateSLAM.landmarkPositionDensity(j).mean();
        cv::Vec3d rPNn;
        cv::eigen2cv(murPNn, rPNn);
        if (camera_.isWorldWithinFOV(rPNn, cameraPose))
        {
            std::cout << "Landmark " << j << " is expected to be within camera FOV" << std::endl;
            idxLandmarks.push_back(j);
        }
        else
        {
            std::cout << "Landmark " << j << " is NOT expected to be within camera FOV" << std::endl;
        }
    }
    
    // TODO: Assignment(s)
    std::iota(idxLandmarks.begin(), idxLandmarks.end(), 0); // Select all landmarks

    Eigen::VectorXd h = stateSLAM.predictFeatureBundle(x, camera_, idxLandmarks);

    Gaussian likelihood(h, noise_.sqrtCov());

    //TODO
    //return likelihood.log(y_);
    return 0;
}

double MeasurementPoseBundle::logLikelihood(const State & state, const Eigen::VectorXd & x, Eigen::VectorXd & g) const
{
    // Evaluate gradient for SR1 and Newton methods
    // TODO: Assignment(s)
    //const StateSLAMPoseLandmarks & stateSLAM = dynamic_cast<const StateSLAMPoseLandmarks &>(state);
    g.resize(x.size());
    g.setZero();
    Eigen::VectorX<autodiff::dual> xdual = x.cast<autodiff::dual>();
    autodiff::dual fdual;
    //g = gradient(&MeasurementPoseBundle::logLikelihoodImpl<autodiff::dual>, wrt(xdual), at(this, xdual, stateSLAM, idxLandmark, j), fdual);
    return val(fdual);
    /*
    g.resize(x.size());
    g.setZero();
    return logLikelihood(state, x);
    */
}

double MeasurementPoseBundle::logLikelihood(const State & state, const Eigen::VectorXd & x, Eigen::VectorXd & g, Eigen::MatrixXd & H) const
{
    // Evaluate Hessian for Newton method
    // TODO: Assignment(s)
    
    const StateSLAMPoseLandmarks & stateSLAM = dynamic_cast<const StateSLAMPoseLandmarks &>(state);
    H.resize(x.size(), x.size());
    H.setZero();
    Eigen::VectorX<autodiff::dual2nd> xdual = x.cast<autodiff::dual2nd>();
    autodiff::dual2nd fdual;
    //H = hessian(&MeasurementPoseBundle::logLikelihoodImpl<autodiff::dual2nd>, wrt(xdual), at(this, xdual, stateSLAM, idxLandmark, j), fdual, g);
    return val(fdual);
    
    /*
    H.resize(x.size(), x.size());
    H.setZero();
    return logLikelihood(state, x, g);   
    */ 
}

void MeasurementPoseBundle::update(State & state)
{
    StateSLAM & stateSLAM = dynamic_cast<StateSLAM &>(state);

    // TODO: Assignment(s)
    // Identify landmarks with matching features (data association)
    // Remove failed landmarks from map (consecutive failures to match)
    // Identify surplus features that do not correspond to landmarks in the map
    // Initialise up to Nmax – N new landmarks from best surplus features

    // What we actually need
    // 
    
    
    Measurement::update(state);  // Do the actual measurement update
}