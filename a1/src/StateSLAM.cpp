#include <cstddef>
#include <cmath>
#include <vector>
#include <Eigen/Core>
#include <opencv2/core/mat.hpp>
#include "Gaussian.hpp"
#include "State.h"
#include "StateSLAM.h"
#include "rotation.hpp"

#include "MeasurementPointBundle.h"
#include "MeasurementPoseBundle.h"


#include <unsupported/Eigen/CXX11/Tensor>
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>

StateSLAM::StateSLAM(const Gaussian<double> & density)
    : State(density)
{}

// Evaluate f(x) from the SDE dx = f(x)*dt + dw
Eigen::VectorXd StateSLAM::dynamics(const Eigen::VectorXd & x) const
{
    assert(size() == x.size());
    /*
    * State containing body velocities, body pose and landmark states
    *
    *     [ vBNb     ]  Body translational velocity (body-fixed)
    *     [ omegaBNb ]  Body angular velocity (body-fixed)
    * x = [ rBNn     ]  Body position (world-fixed)
    *     [ Thetanb  ]  Body orientation (world-fixed)
    *     [ m        ]  Landmark map states (undefined in this class)
    *
    */
    //
    //  dnu/dt =          0 + dwnu/dt
    // deta/dt = JK(eta)*nu +       0
    //   dm/dt =          0 +       0
    // \_____/   \________/   \_____/
    //  dx/dt  =    f(x)    +  dw/dt
    //
    //        [          0 ]
    // f(x) = [ JK(eta)*nu ]
    //        [          0 ] for all map states
    //
    //        [                    0 ]
    //        [                    0 ]
    // f(x) = [    Rnb(thetanb)*vBNb ]
    //        [ TK(thetanb)*omegaBNb ]
    //        [                    0 ] for all map states
    //

    /*
    Eigen::VectorXd f(x.size());
    f.setZero();

    Eigen::Vector3d vBNb        = x.segment(0,3);
    Eigen::Vector3d omegaBNb    = x.segment(3,3);
    Eigen::Vector3d rBNb        = x.segment(6,3);
    Eigen::Vector3d Thetanb     = x.segment(9,3);

    Eigen::Matrix3d Rnb         = rpy2rot(Thetanb);

    Eigen::Matrix3d Tk; 

    //Thetanb(0) = phi
    //Thetanb(1) = theta    
    //Thetanb(2) = psi

    using std::cos, std::sin, std::tan;

    Tk << 1, sin(Thetanb(0))*tan(Thetanb(1)), cos(Thetanb(0))*tan(Thetanb(1)),
          0,                 cos(Thetanb(0)),                -sin(Thetanb(0)),
          0, sin(Thetanb(0))/cos(Thetanb(1)), cos(Thetanb(0))/cos(Thetanb(1));

    Eigen::Vector3d firstCalculation = Rnb * vBNb;
    Eigen::Vector3d secondCalculation = Tk * omegaBNb;

    f.segment(6, 3) = firstCalculation;
    f.segment(9, 3) = secondCalculation;

    return f;
    */
   //std::cout << "Dynamics Time !!!" << std::endl;
   return dynamicsImpl(x);
}

// Evaluate f(x) and its Jacobian J = df/fx from the SDE dx = f(x)*dt + dw
Eigen::VectorXd StateSLAM::dynamics(const Eigen::VectorXd & x, Eigen::MatrixXd & J) const
{
    Eigen::VectorXd f = dynamics(x);

    // Jacobian J = df/dx
    //    
    //     [  0                  0 0 ]
    // J = [ JK d(JK(eta)*nu)/deta 0 ]
    //     [  0                  0 0 ]
    //

    J.resize(f.size(), x.size());
    J.setZero();

    Eigen::VectorX<autodiff::dual> xdual = x.cast<autodiff::dual>();
    Eigen::VectorX<autodiff::dual> fdual;
    J = jacobian(&StateSLAM::dynamicsImpl<autodiff::dual>, wrt(xdual), at(this, xdual), fdual);

    return fdual.cast<double>();
}

cv::Mat & StateSLAM::view()
{
    return view_;
};

const cv::Mat & StateSLAM::view() const
{
    return view_;
};

Gaussian<double> StateSLAM::bodyPositionDensity() const
{
    return density.marginal(Eigen::seqN(6, 3));
}

Gaussian<double> StateSLAM::bodyOrientationDensity() const
{
    return density.marginal(Eigen::seqN(9, 3));
}

Gaussian<double> StateSLAM::bodyTranslationalVelocityDensity() const
{
    return density.marginal(Eigen::seqN(0, 3));
}

Gaussian<double> StateSLAM::bodyAngularVelocityDensity() const
{
    return density.marginal(Eigen::seqN(3, 3));
}

#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>

Eigen::Vector3d StateSLAM::cameraPosition(const Camera & cam, const Eigen::VectorXd & x, Eigen::MatrixXd & J)
{
    Eigen::Vector3<autodiff::dual> rCNn_dual;
    Eigen::VectorX<autodiff::dual> x_dual = x.cast<autodiff::dual>();
    J = jacobian(cameraPosition<autodiff::dual>, wrt(x_dual), at(cam, x_dual), rCNn_dual);
    return rCNn_dual.cast<double>();
};

Gaussian<double> StateSLAM::cameraPositionDensity(const Camera & cam) const
{
    auto f = [&](const Eigen::VectorXd & x, Eigen::MatrixXd & J) { return cameraPosition(cam, x, J); };
    return density.transform(f);
}

Eigen::Vector3d StateSLAM::cameraOrientationEuler(const Camera & cam, const Eigen::VectorXd & x, Eigen::MatrixXd & J)
{
    Eigen::Vector3<autodiff::dual> Thetanc_dual;
    Eigen::VectorX<autodiff::dual> x_dual = x.cast<autodiff::dual>();
    J = jacobian(cameraOrientationEuler<autodiff::dual>, wrt(x_dual), at(cam, x_dual), Thetanc_dual);
    return Thetanc_dual.cast<double>();
};

Gaussian<double> StateSLAM::cameraOrientationEulerDensity(const Camera & cam) const
{
    auto f = [&](const Eigen::VectorXd & x, Eigen::MatrixXd & J) { return cameraOrientationEuler(cam, x, J); };
    return density.transform(f);    
}

Gaussian<double> StateSLAM::landmarkPositionDensity(std::size_t idxLandmark) const
{
    assert(idxLandmark < numberLandmarks());
    std::size_t idx = landmarkPositionIndex(idxLandmark);
    return density.marginal(Eigen::seqN(idx, 3));
}

#include "Camera.h"
#include "rotation.hpp"

// Image feature location for a given landmark and Jacobian
Eigen::Vector2d StateSLAM::predictFeature(const Eigen::VectorXd & x, Eigen::MatrixXd & J, const Camera & cam, std::size_t idxLandmark) const
{
    // Set elements of J
    Eigen::VectorX<autodiff::dual> xdual = x.cast<autodiff::dual>();
    Eigen::Vector2<autodiff::dual> fdual;
    // Define a lambda function that captures the arguments and calls the .h implementation
    auto predictFeatureLambda = [&](const Eigen::VectorX<autodiff::dual> & xdual) {
        return predictFeature(xdual, cam, idxLandmark);
    };

    // Compute the jacobian using the lambda function
    J = jacobian(predictFeatureLambda, wrt(xdual), at(xdual), fdual);

    return fdual.cast<double>(); // cast return value to double
    
    //return predictFeature(x, cam, idxLandmark);
    // Note: If you use autodiff, return the evaluated function value (cast with double scalar type) instead of calling predictFeature as above
}

// Density of image feature location for a given landmark
Gaussian<double> StateSLAM::predictFeatureDensity(const Camera & cam, std::size_t idxLandmark) const
{
    const auto func = [&](const Eigen::VectorXd & x, Eigen::MatrixXd & J) { return predictFeature(x, J, cam, idxLandmark); };
    return density.transform(func);
}

// Density of image feature location for a given landmark with added noise
Gaussian<double> StateSLAM::predictFeatureDensity(const Camera & cam, std::size_t idxLandmark, const Gaussian<double> & noise) const
{
    const auto func = [&](const Eigen::VectorXd & x, Eigen::MatrixXd & J) { return predictFeature(x, J, cam, idxLandmark); };
    return density.transform(func, noise);
}

// Image feature locations for a bundle of landmarks
Eigen::VectorXd StateSLAM::predictFeatureBundle(const Eigen::VectorXd & x, Eigen::MatrixXd & J, const Camera & cam, const std::vector<std::size_t> & idxLandmarks) const
{
    const std::size_t & nL = idxLandmarks.size();
    const std::size_t & nx = size();
    assert(x.size() == nx);

    Eigen::VectorXd h(2*nL);
    J.resize(2*nL, nx);
    for (std::size_t i = 0; i < nL; ++i)
    {
        Eigen::MatrixXd Jfeature;
        Eigen::Vector2d rQOi = predictFeature(x, Jfeature, cam, idxLandmarks[i]);
        // Set pair of elements of h
        // TODO: Lab 8
        // Set pair of rows of J
        // TODO: Lab 8
    }
    return h;
}

// Density of image features for a set of landmarks
Gaussian<double> StateSLAM::predictFeatureBundleDensity(const Camera & cam, const std::vector<std::size_t> & idxLandmarks) const
{
    auto func = [&](const Eigen::VectorXd & x, Eigen::MatrixXd & J) { return predictFeatureBundle(x, J, cam, idxLandmarks); };
    return density.transform(func);
}

// Density of image features for a set of landmarks
Gaussian<double> StateSLAM::predictFeatureBundleDensity(const Camera & cam, const std::vector<std::size_t> & idxLandmarks, const Gaussian<double> & noise) const
{
    auto func = [&](const Eigen::VectorXd & x, Eigen::MatrixXd & J) { return predictFeatureBundle(x, J, cam, idxLandmarks); };
    return density.transform(func, noise);
}

// Corner feature location for a given ArUco landmark and Jacobian
Eigen::Vector2d StateSLAM::predictFeatureTag(const Eigen::VectorXd & x, Eigen::MatrixXd & J, const Camera & cam, std::size_t idxLandmark, int j) const
{
    // Set elements of J
    Eigen::VectorX<autodiff::dual> xdual = x.cast<autodiff::dual>();
    Eigen::Vector2<autodiff::dual> fdual;
    // Define a lambda function that captures the arguments and calls the .h implementation
    auto predictFeatureLambda = [&](const Eigen::VectorX<autodiff::dual> & xdual) {
        return predictFeatureTag(xdual, cam, idxLandmark, j);
    };

    // Compute the jacobian using the lambda function
    J = jacobian(predictFeatureLambda, wrt(xdual), at(xdual), fdual);

    return fdual.cast<double>(); // cast return value to double
    
    //return predictFeature(x, cam, idxLandmark);
    // Note: If you use autodiff, return the evaluated function value (cast with double scalar type) instead of calling predictFeature as above
}
