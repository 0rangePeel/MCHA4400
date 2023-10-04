#include <cstddef>
#include <numeric>
#include <vector>
#include <Eigen/Core>
#include "rotation.hpp"
#include "State.h"
#include "Camera.h"
#include "StateSLAM.h"
#include "StateSLAMPoseLandmarks.h"
#include "MeasurementTagBundle.h"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgcodecs.hpp>

#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>

MeasurementTagBundle::MeasurementTagBundle(double time, const Eigen::VectorXd & y, const Camera & camera)
    : Measurement(time, y)
    , camera_(camera)
{  
    // SR is an upper triangular matrix such that SR.'*SR = R is the measurement noise covariance
    const Eigen::Index & ny = y.size();
    //double rms = 1.27127; // Manually put in from Camera Calibration
    double rms = 3.5;
    //double rms = ;
    Eigen::MatrixXd SR = rms*Eigen::MatrixXd::Identity(ny, ny);
    noise_ = Gaussian(SR);

    useQuasiNewton = true;  
}

double MeasurementTagBundle::logLikelihood(const State & state, const Eigen::VectorXd & x) const
{
    const StateSLAMPoseLandmarks & stateSLAM = dynamic_cast<const StateSLAMPoseLandmarks &>(state);
    return logLikelihoodImpl(x, stateSLAM, state.getIdxLandmarks());
}

double MeasurementTagBundle::logLikelihood(const State & state, const Eigen::VectorXd & x, Eigen::VectorXd & g) const
{
    // Evaluate gradient for SR1 and Newton methods
    const StateSLAMPoseLandmarks & stateSLAM = dynamic_cast<const StateSLAMPoseLandmarks &>(state);

    g.resize(x.size());
    g.setZero();
    Eigen::VectorX<autodiff::dual> xdual = x.cast<autodiff::dual>();
    autodiff::dual fdual;
    g = gradient(&MeasurementTagBundle::logLikelihoodImpl<autodiff::dual>, wrt(xdual), at(this, xdual, stateSLAM, state.getIdxLandmarks()), fdual);
    return val(fdual);
}

double MeasurementTagBundle::logLikelihood(const State & state, const Eigen::VectorXd & x, Eigen::VectorXd & g, Eigen::MatrixXd & H) const
{
    // Evaluate Hessian for Newton method  
    const StateSLAMPoseLandmarks & stateSLAM = dynamic_cast<const StateSLAMPoseLandmarks &>(state);

    H.resize(x.size(), x.size());
    H.setZero();
    Eigen::VectorX<autodiff::dual2nd> xdual = x.cast<autodiff::dual2nd>();
    autodiff::dual2nd fdual;
    H = hessian(&MeasurementTagBundle::logLikelihoodImpl<autodiff::dual2nd>, wrt(xdual), at(this, xdual, stateSLAM, state.getIdxLandmarks()), fdual, g);
    return val(fdual);
}

void MeasurementTagBundle::update(State & state)
{
    Pose cameraPose;
    StateSLAM & stateSLAM = dynamic_cast<StateSLAM &>(state);

    state.numNewLandmarks = 0;

    // Get landmarks
    std::vector<int> idsLandmarks = state.getIdsLandmarks();
    std::vector<int> idsHistLandmarks = state.getIdsHistLandmarks();
    std::vector<std::size_t> idxLandmarks = state.getIdxLandmarks();

    // Initialise New Landmarks //
    // If idxLandmark is size 0 then put first idsLandmark inside idsHistLandmark
    if (state.getIdsHistLandmarks().empty()) {
        // If it's empty, simply copy all values from idsLandmarks
        for (int i = 0; i < idsLandmarks.size(); i++) {
            int id = idsLandmarks[i];
            state.modifyIdsHistLandmarks(id);
            idsHistLandmarks.push_back(id);
            state.numNewLandmarks = state.numNewLandmarks + 1;
        }
    } 
    else 
    {
        // Get idsLandmarks and see if any mismatch between idsHistLandmarks
        // If mismatch is there, append idsLandmark to end of idsHistLandmark
        // Loop through idsLandmarks and check if each value is in idsHistLandmarks
        for (const int& id : idsLandmarks) {
            auto it = std::find(idsHistLandmarks.begin(), idsHistLandmarks.end(), id);
            if (it == idsHistLandmarks.end()) {
                state.modifyIdsHistLandmarks(id);
                idsHistLandmarks.push_back(id);
                state.numNewLandmarks = state.numNewLandmarks + 1;
            }
        }
    }

    // Increase size of sqrtCov matrix and mu array to match number of ArUco Tags
    if (stateSLAM.density.sqrtCov().rows() < (12 + 6*idsHistLandmarks.size()))
    {
        int densitySize = stateSLAM.density.sqrtCov().rows();
        int idsHistSize = 12 + 6*idsHistLandmarks.size();
        stateSLAM.density.sqrtCov().conservativeResizeLike(Eigen::MatrixXd::Zero(idsHistSize,idsHistSize));

        // For the newly added diagonal elements, initialise them to 10 instead of the initial zeros.
        /*
        for (int i = densitySize; i < idsHistSize; ++i){
            stateSLAM.density.sqrtCov()(i, i) = 1000.0;
        }
        */
        
        
        for (int i = densitySize; i < idsHistSize; i=i+6){
            stateSLAM.density.sqrtCov()(i + 0, i + 0) = 1000.0;
            stateSLAM.density.sqrtCov()(i + 1, i + 1) = 1000.0;
            stateSLAM.density.sqrtCov()(i + 2, i + 2) = 1000.0;
            stateSLAM.density.sqrtCov()(i + 3, i + 3) = M_PI/2;
            stateSLAM.density.sqrtCov()(i + 4, i + 4) = M_PI/2;
            stateSLAM.density.sqrtCov()(i + 5, i + 5) = M_PI/2;
        }

        //std::cout << stateSLAM.density.sqrtCov() << std::endl;
        
        stateSLAM.density.mean().conservativeResizeLike(Eigen::VectorXd::Zero(idsHistSize));
        
        // Place landmarks in front of the camera by some amount and rotate 180 back towards the camera
        Eigen::Vector3d rBNn    = stateSLAM.density.mean().segment(6,3);
        Eigen::Vector3d Thetanb = stateSLAM.density.mean().segment(9,3);
        Eigen::Matrix3d Rnb     = rpy2rot(Thetanb);
        Eigen::Vector3d rotVec(0,0,M_PI);// Rotate the around n3
        Eigen::Vector3d unitVec(1,0,0);

        for (int i = densitySize; i < idsHistSize; i=i+6){
            stateSLAM.density.mean().segment<3>(i) = Rnb*unitVec + rBNn;
            stateSLAM.density.mean().segment<3>(i+3) = Thetanb + rotVec;
        }
         
        std::cout << "New Aruco ID Detected" << std::endl;
    }

    // Data Association //
    // Create idxLandmarks after everything completed by using idsLandmarks then verfering to idsHistLandmarks 
    // Loop through idsLandmarks and look up values in idsHistLandmarks
    // Reset idxLandmarks
    std::vector<std::size_t> emptyVector; // Create an empty vector
    state.setIdxLandmarks(emptyVector);   // Set idxLandmarks to the empty vector

    for (const int& id : idsLandmarks) {
        auto it = std::find(idsHistLandmarks.begin(), idsHistLandmarks.end(), id);
        if (it != idsHistLandmarks.end()) {
            // id found in idsHistLandmarks, append its position to idxLandmarks
            std::size_t position = std::distance(idsHistLandmarks.begin(), it);
            state.modifyIdxLandmarks(position);
        }
    }
       
    Measurement::update(state);  // Do the actual measurement update
}