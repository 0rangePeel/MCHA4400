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
    double rms = 1.27127; // Manually put in from Camera Calibration
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
    //for (std::size_t j = 0; j < state.stateSLAM.numberLandmarks(); ++j)
    /*
    for (std::size_t j = 0; j < state.idxLandmarks.size(); ++j)
    {
        Eigen::Vector3d murPNn = stateSLAM.landmarkPositionDensity(j).mean();
        cv::Vec3d rPNn;
        cv::eigen2cv(murPNn, rPNn);
        if (camera_.isWorldWithinFOV(rPNn, cameraPose))
        {
            std::cout << "Landmark " << j << " is expected to be within camera FOV" << std::endl;
            state.idxLandmarks.push_back(j);
        }
        else
        {
            std::cout << "Landmark " << j << " is NOT expected to be within camera FOV" << std::endl;
        }
    }
    */
    
    // TODO: Assignment(s)

    std::iota(idxLandmarks.begin(), idxLandmarks.end(), 0); // Select all landmarks

    Eigen::VectorXd h = stateSLAM.predictFeatureTagBundle(x, camera_, idxLandmarks);

    for (int i = 0; i < h.size(); i++){
        std::cout << h(i) << std::endl;
    }

    Gaussian likelihood(h, noise_.sqrtCov());

    return likelihood.log(y_);

    // or

    //return MeasurementPoseBundle.logLikelihoodImpl(x, stateSLAM, idxLandmarks);
}

double MeasurementPoseBundle::logLikelihood(const State & state, const Eigen::VectorXd & x, Eigen::VectorXd & g) const
{
    // Evaluate gradient for SR1 and Newton methods
    const StateSLAMPoseLandmarks & stateSLAM = dynamic_cast<const StateSLAMPoseLandmarks &>(state);
    // Select visible landmarks
    //std::vector<std::size_t> idxLandmarks(stateSLAM.numberLandmarks());
    // Select all landmarks
    //std::iota(idxLandmarks.begin(), idxLandmarks.end(), 0); 

    g.resize(x.size());
    g.setZero();
    Eigen::VectorX<autodiff::dual> xdual = x.cast<autodiff::dual>();
    autodiff::dual fdual;
    //std::vector<std::size_t> idxLandmarks = state.getIdxLandmarks();
    //g = gradient(&MeasurementPoseBundle::logLikelihoodImpl<autodiff::dual>, wrt(xdual), at(this, xdual, stateSLAM, idxLandmarks), fdual);
    g = gradient(&MeasurementPoseBundle::logLikelihoodImpl<autodiff::dual>, wrt(xdual), at(this, xdual, stateSLAM, state.getIdxLandmarks()), fdual);
    return val(fdual);
}

double MeasurementPoseBundle::logLikelihood(const State & state, const Eigen::VectorXd & x, Eigen::VectorXd & g, Eigen::MatrixXd & H) const
{
    // Evaluate Hessian for Newton method  
    const StateSLAMPoseLandmarks & stateSLAM = dynamic_cast<const StateSLAMPoseLandmarks &>(state);
    // Select visible landmarks
    //std::vector<std::size_t> idxLandmarks(stateSLAM.numberLandmarks());
    // Select all landmarks
    //std::iota(idxLandmarks.begin(), idxLandmarks.end(), 0); 

    H.resize(x.size(), x.size());
    H.setZero();
    Eigen::VectorX<autodiff::dual2nd> xdual = x.cast<autodiff::dual2nd>();
    autodiff::dual2nd fdual;
    //H = hessian(&MeasurementPoseBundle::logLikelihoodImpl<autodiff::dual2nd>, wrt(xdual), at(this, xdual, stateSLAM, idxLandmarks), fdual, g);
    //H = hessian(&MeasurementPoseBundle::logLikelihoodImpl<autodiff::dual2nd>, wrt(xdual), at(this, xdual, stateSLAM, state.getIdxLandmarks()), fdual, g);
    return val(fdual);
}

void MeasurementPoseBundle::update(State & state)
{
    StateSLAM & stateSLAM = dynamic_cast<StateSLAM &>(state);
/*
    // If idxLandmark is size 0 then put first idsLandmark inside idsHistLandmark
    if (state.getIdsHistLandmarks.empty()) {
        // If it's empty, simply copy all values from idsLandmarks
        state.setIdsHistLandmarks() = idsLandmarks;
    } else 
    {
        // Get idsLandmarks and see if any mismatch between idsHistLandmarks
        // If mismatch is there, append idsLandmark to end of idxLandmark
        // Loop through idsLandmarks and check if each value is in idsHistLandmarks
        for (const int& id : state.getIdsLandmarks()) {
            if (std::find(state.idsHistLandmarks.begin(), state.idsHistLandmarks.end(), id) == state.idsHistLandmarks.end()) {
                // id is not in idsHistLandmarks, so append it
                state.idsHistLandmarks.push_back(id);
            }
        }
    }

    // Create idxLandmarks after everything completed by using idsLandmarks then verfering to idsHistLandmarks 
    // Loop through idsLandmarks and look up values in idsHistLandmarks
    for (const int& id : state.idsLandmarks) {
        auto it = std::find(state.idsHistLandmarks.begin(), state.idsHistLandmarks.end(), id);
        if (it != state.idsHistLandmarks.end()) {
            // id found in idsHistLandmarks, append its position to idxLandmarks
            std::size_t position = std::distance(state.idsHistLandmarks.begin(), it);
            state.idxLandmarks.push_back(position);
        }
    }
  */  
    Measurement::update(state);  // Do the actual measurement update
}