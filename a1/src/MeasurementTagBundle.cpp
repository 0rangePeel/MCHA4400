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
    double rms = 1.27127; // Manually put in from Camera Calibration
    //double rms = 5; // Manually put in from Camera Calibration
    Eigen::MatrixXd SR = rms*Eigen::MatrixXd::Identity(ny, ny); // TODO: Assignment(s)
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
    StateSLAM & stateSLAM = dynamic_cast<StateSLAM &>(state);

    std::cout << "muSize " << stateSLAM.density.mean().size() << std::endl;
    std::cout << "sqrtSize " << stateSLAM.density.sqrtCov().rows() << " " << stateSLAM.density.sqrtCov().cols() << std::endl;

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
            //std::cout << "Print Id" << std::endl;
            std::cout << id << std::endl;
        }
    } 
    else 
    {
        // Get idsLandmarks and see if any mismatch between idsHistLandmarks
        // If mismatch is there, append idsLandmark to end of idsHistLandmark
        // Loop through idsLandmarks and check if each value is in idsHistLandmarks
        for (const int& id : idsLandmarks) {
            //std::cout << "id " << id << std::endl;
            auto it = std::find(idsHistLandmarks.begin(), idsHistLandmarks.end(), id);
            if (it == idsHistLandmarks.end()) {
                //std::cout << id << " not found in the vector" << std::endl;
                state.modifyIdsHistLandmarks(id);
                idsHistLandmarks.push_back(id);
            }
        }
    }

    //std::cout << "sqrtSize* " << (12 + 6*idsHistLandmarks.size()) << std::endl;

    // Increase size of sqrtCov matrix and mu array to match number of ArUco Tags
    if (stateSLAM.density.sqrtCov().rows() < (12 + 6*idsHistLandmarks.size()))
    {
        int densitySize = stateSLAM.density.sqrtCov().rows();
        int idsHistSize = 12 + 6*idsHistLandmarks.size();
        stateSLAM.density.sqrtCov().conservativeResizeLike(Eigen::MatrixXd::Zero(idsHistSize,idsHistSize));

        // For the newly added diagonal elements, initialise them to 1000 instead of the initial zeros.
        for (int i = densitySize; i < idsHistSize; ++i)
        {
            stateSLAM.density.sqrtCov()(i, i) = 1000.0;
        }
        stateSLAM.density.mean().conservativeResizeLike(Eigen::VectorXd::Zero(idsHistSize));
        std::cout << "Increase sqrtCov " << std::endl;
    }

    std::cout << "sqrtSize after " << stateSLAM.density.sqrtCov().rows() << std::endl;
    
    std::cout << "idsSize " << idsLandmarks.size() << std::endl;
    std::cout << "idsHistSize " << idsHistLandmarks.size() << std::endl;

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
            //std::cout << position << std::endl;
            state.modifyIdxLandmarks(position);
            //idxLandmarks.push_back(position); // this is not needed
        }
    }
    //std::cout << "idxSize " << idxLandmarks.size() << std::endl;

    std::cout << "ids" << std::endl;
    const std::vector<int>& idsLandmarkstemp = state.getIdsLandmarks();
    for (const int& idstemp : idsLandmarkstemp) {
        std::cout << idstemp << std::endl;
    }
    std::cout << "idsHist" << std::endl;
    const std::vector<int>& idsHistLandmarkstemp = state.getIdsHistLandmarks();
    for (const int& idsHisttemp : idsHistLandmarkstemp) {
        std::cout << idsHisttemp << std::endl;
    }
    std::cout << "idx" << std::endl;
    const std::vector<std::size_t>& idxLandmarkstemp = state.getIdxLandmarks();
    for (std::size_t idxtemp : idxLandmarkstemp) {
        std::cout << idxtemp << std::endl;
    }


    Eigen::MatrixXd sqrtCov = stateSLAM.density.sqrtCov();
    int n = sqrtCov.rows(); // Assuming a square matrix
    // Print the diagonal elements
    std::cout << "Diagonal elements of S:" << std::endl;
    for (int i = 0; i < n; ++i) {
        std::cout << sqrtCov(i, i) << " ";
    }
    std::cout << std::endl;

    // Assuming stateSLAM.density.mean() is an Eigen array
    Eigen::ArrayXd mean = stateSLAM.density.mean();
    // Print the elements of the mean array
    std::cout << "Mean: ";
    for (int i = 0; i < mean.size(); ++i) {
        std::cout << mean(i) << " ";
    }
    std::cout << std::endl;
       
    Measurement::update(state);  // Do the actual measurement update
}

    /*
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