#ifndef STATESLAM_H
#define STATESLAM_H

#include <iostream>
#include <cmath>
#include <vector>
#include <Eigen/Core>
#include <opencv2/core/mat.hpp>
#include "Gaussian.hpp"
#include "Camera.h"
#include "State.h"
#include "rotation.hpp"

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
class StateSLAM : public State
{
public:
    explicit StateSLAM(const Gaussian<double> & density);
    virtual StateSLAM * clone() const = 0;
    virtual Eigen::VectorXd dynamics(const Eigen::VectorXd & x) const override;
    virtual Eigen::VectorXd dynamics(const Eigen::VectorXd & x, Eigen::MatrixXd & J) const override;

    virtual Gaussian<double> bodyPositionDensity() const;
    virtual Gaussian<double> bodyOrientationDensity() const;
    virtual Gaussian<double> bodyTranslationalVelocityDensity() const;
    virtual Gaussian<double> bodyAngularVelocityDensity() const;

    template <typename Scalar> static Eigen::Vector3<Scalar> cameraPosition(const Camera & cam, const Eigen::VectorX<Scalar> & x);
    static Eigen::Vector3d cameraPosition(const Camera & cam, const Eigen::VectorXd & x, Eigen::MatrixXd & J);
    template <typename Scalar> static Eigen::Matrix3<Scalar> cameraOrientation(const Camera & cam, const Eigen::VectorX<Scalar> & x);
    template <typename Scalar> static Eigen::Vector3<Scalar> cameraOrientationEuler(const Camera & cam, const Eigen::VectorX<Scalar> & x);
    static Eigen::Vector3d cameraOrientationEuler(const Camera & cam, const Eigen::VectorXd & x, Eigen::MatrixXd & J);

    virtual Gaussian<double> cameraPositionDensity(const Camera & cam) const;
    virtual Gaussian<double> cameraOrientationEulerDensity(const Camera & cam) const;

    virtual std::size_t numberLandmarks() const = 0;
    virtual Gaussian<double> landmarkPositionDensity(std::size_t idxLandmark) const;
    virtual std::size_t landmarkPositionIndex(std::size_t idxLandmark) const = 0;

    template <typename Scalar> Eigen::VectorX<Scalar> dynamicsImpl(const Eigen::VectorX<Scalar> & x) const;

    template <typename Scalar> Eigen::Vector2<Scalar> predictFeature(const Eigen::VectorX<Scalar> & x, const Camera & cam, std::size_t idxLandmark) const;
    Eigen::Vector2d predictFeature(const Eigen::VectorXd & x, Eigen::MatrixXd & J, const Camera & cam, std::size_t idxLandmark) const;
    Gaussian<double> predictFeatureDensity(const Camera & cam, std::size_t idxLandmark) const;
    Gaussian<double> predictFeatureDensity(const Camera & cam, std::size_t idxLandmark, const Gaussian<double> & noise) const;

    template <typename Scalar> Eigen::VectorX<Scalar> predictFeatureBundle(const Eigen::VectorX<Scalar> & x, const Camera & cam, const std::vector<std::size_t> & idxLandmarks) const;
    Eigen::VectorXd predictFeatureBundle(const Eigen::VectorXd & x, Eigen::MatrixXd & J, const Camera & cam, const std::vector<std::size_t> & idxLandmarks) const;
    Gaussian<double> predictFeatureBundleDensity(const Camera & cam, const std::vector<std::size_t> & idxLandmarks) const;
    Gaussian<double> predictFeatureBundleDensity(const Camera & cam, const std::vector<std::size_t> & idxLandmarks, const Gaussian<double> & noise) const;

    template <typename Scalar> Eigen::Vector2<Scalar> predictFeatureTag(const Eigen::VectorX<Scalar> & x, const Camera & cam, std::size_t idxLandmark, const int j) const;

    template <typename Scalar> Eigen::VectorX<Scalar> predictFeatureTagBundle(const Eigen::VectorX<Scalar> & x, const Camera & cam, const std::vector<std::size_t> & idxLandmarks) const;
    Eigen::VectorXd predictFeatureTagBundle(const Eigen::VectorXd & x, Eigen::MatrixXd & J, const Camera & cam, const std::vector<std::size_t> & idxLandmarks) const;

    cv::Mat & view();
    const cv::Mat & view() const;
protected:
    cv::Mat view_;
};

#include "rotation.hpp"

template <typename Scalar>
Eigen::VectorX<Scalar> StateSLAM::dynamicsImpl(const Eigen::VectorX<Scalar> & x) const
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
    
    Eigen::VectorX<Scalar> f(x.size());
    f.setZero();

    Eigen::Vector3<Scalar> vBNb        = x.template segment<3>(0);
    Eigen::Vector3<Scalar> omegaBNb    = x.template segment<3>(3);
    Eigen::Vector3<Scalar> rBNn        = x.template segment<3>(6);
    Eigen::Vector3<Scalar> Thetanb     = x.template segment<3>(9);

    Eigen::Matrix3<Scalar> Rnb         = rpy2rot(Thetanb);

    Eigen::Matrix3<Scalar> Tk; 

    //Thetanb(0) = phi
    //Thetanb(1) = theta    
    //Thetanb(2) = psi

    using std::cos, std::sin, std::tan;

    Tk << 1, sin(Thetanb(0))*tan(Thetanb(1)), cos(Thetanb(0))*tan(Thetanb(1)),
          0,                 cos(Thetanb(0)),                -sin(Thetanb(0)),
          0, sin(Thetanb(0))/cos(Thetanb(1)), cos(Thetanb(0))/cos(Thetanb(1));

    //std::cout <<"Tk"<< std::endl;
    //std::cout << Tk << std::endl;

    f.template segment<3>(6) = Rnb * vBNb;
    f.template segment<3>(9) = Tk * omegaBNb;

    return f;
}

template <typename Scalar>
Eigen::Vector3<Scalar> StateSLAM::cameraPosition(const Camera & cam, const Eigen::VectorX<Scalar> & x)
{
    Eigen::Vector3<Scalar> rBNn = x.template segment<3>(6);
    Eigen::Vector3<Scalar> Thetanb = x.template segment<3>(9);
    Eigen::Matrix3<Scalar> Rnb = rpy2rot(Thetanb); 
    Eigen::Vector3<Scalar> rCNn = rBNn + Rnb*cam.rCBb;
    return rCNn;
}

template <typename Scalar>
Eigen::Matrix3<Scalar> StateSLAM::cameraOrientation(const Camera & cam, const Eigen::VectorX<Scalar> & x)
{
    Eigen::Vector3<Scalar> Thetanb = x.template segment<3>(9);
    Eigen::Matrix3<Scalar> Rnb = rpy2rot(Thetanb); 
    Eigen::Matrix3<Scalar> Rnc = Rnb*cam.Rbc;
    return Rnc;
}

template <typename Scalar>
Eigen::Vector3<Scalar> StateSLAM::cameraOrientationEuler(const Camera & cam, const Eigen::VectorX<Scalar> & x)
{
    return rot2rpy(cameraOrientation(cam, x));
}

// Image feature location for a given landmark
template <typename Scalar>
Eigen::Vector2<Scalar> StateSLAM::predictFeature(const Eigen::VectorX<Scalar> & x, const Camera & cam, std::size_t idxLandmark) const
{
    // Obtain camera pose from state
    Eigen::Vector3<Scalar> rCNn = cameraPosition(cam, x);
    Eigen::Matrix3<Scalar> Rnc = cameraOrientation(cam, x);

    // Obtain landmark position from state
    std::size_t idx = landmarkPositionIndex(idxLandmark);
    Eigen::Vector3<Scalar> rPNn = x.template segment<3>(idx);

    // Camera vector
    Eigen::Vector3<Scalar> rPCc;
    
    rPCc = Rnc.transpose() * (rPNn - rCNn);

    // Pixel coordinates
    Eigen::Vector2<Scalar> rQOi;
    
    rQOi = cam.vectorToPixel(rPCc);

    return rQOi;
}

// Image feature locations for a bundle of landmarks
template <typename Scalar>
Eigen::VectorX<Scalar> StateSLAM::predictFeatureBundle(const Eigen::VectorX<Scalar> & x, const Camera & cam, const std::vector<std::size_t> & idxLandmarks) const
{
    const std::size_t & nL = idxLandmarks.size();
    const std::size_t & nx = size();
    assert(x.size() == nx);

    Eigen::VectorX<Scalar> h(2*nL);
    for (std::size_t i = 0; i < nL; ++i)
    {
        Eigen::Vector2<Scalar> rQOi = predictFeature(x, cam, idxLandmarks[i]);
        // Set pair of elements of h
        // TODO: Lab 8
    }
    return h;
}

// Image feature location for a given ARUCO Tag point
template <typename Scalar>
Eigen::Vector2<Scalar> StateSLAM::predictFeatureTag(const Eigen::VectorX<Scalar> & x, const Camera & cam, std::size_t idxLandmark, int j) const
{
    // Obtain camera pose from state
    Eigen::Vector3<Scalar> rCNn = cameraPosition(cam, x);
    Eigen::Matrix3<Scalar> Rnc = cameraOrientation(cam, x);

    // Obtain landmark position from state
    // idx is equuivalent to mj which is a 6x1 which contains
    // position and rotation of tag centre
    // This is placed into rjNn and thetanj respectively
    std::size_t idx = landmarkPositionIndex(idxLandmark);
    Eigen::Vector3<Scalar> rjNn = x.template segment<3>(idx);
    Eigen::Vector3<Scalar> thetanj = x.template segment<3>(idx+3);
    Eigen::Matrix3<Scalar> Rnj = rpy2rot(thetanj);

    Eigen::Vector3<Scalar> rjcNj;

    const double l = 0.166;
    
    // Corner offset given which corner from j
    switch (j) {
        case 0:
            rjcNj << -l/2, l/2, 0;
            break;
        case 1:
            rjcNj << l/2, l/2, 0;
            break;
        case 2:
            rjcNj << l/2, -l/2, 0;
            break;
        case 3:
            rjcNj<< -l/2, -l/2, 0;
            break;
        default:
            std::cout << "Invalid choice. Please enter a number between 0 and 3." << std::endl;
    }

    // Equation 9 from Assignemnt 1
    Eigen::Vector3<Scalar> rjcNn = Rnj * rjcNj + rjNn;
    // Camera vector
    Eigen::Vector3<Scalar> rPCc = Rnc.transpose() * (rjcNn - rCNn);
    // Pixel coordinates
    Eigen::Vector2<Scalar> rQOi = cam.vectorToPixel(rPCc);

    return rQOi;
}

// Image feature location for a bundle of ARUCO Tag point
template <typename Scalar>
Eigen::VectorX<Scalar> StateSLAM::predictFeatureTagBundle(const Eigen::VectorX<Scalar> & x, const Camera & cam, const std::vector<std::size_t> & idxLandmarks) const
{
    const std::size_t & nL = idxLandmarks.size();
    const std::size_t & nx = size();
    assert(x.size() == nx);

    Eigen::VectorX<Scalar> h(8*nL);
    std::size_t h_index = 0; // Initialize the index for h
    for (std::size_t i = 0; i < nL; ++i)
    {
        for (int j = 0; j < 4; j++)
        {
            Eigen::Vector2<Scalar> rQOi = predictFeatureTag(x, cam, idxLandmarks[i], j);
            // Set pair of elements in h
            h(h_index) = rQOi(0); // First element of rQOi
            h(h_index + 1) = rQOi(1); // Second element of rQOi

            h_index += 2; // Increment the index by 2 to move to the next pair
        }
    }
    return h;
}

#endif
