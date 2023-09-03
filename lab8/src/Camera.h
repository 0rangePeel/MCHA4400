#ifndef CAMERA_H
#define CAMERA_H

#include <vector>
#include <filesystem>
#include <Eigen/Core>
#include <opencv2/core/types.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/persistence.hpp>
#include "serialisation.hpp"

struct Pose
{
    cv::Matx33d Rnc;
    cv::Vec3d rCNn;
};

struct Chessboard
{
    cv::Size boardSize;
    float squareSize;

    void write(cv::FileStorage & fs) const;                 // OpenCV serialisation
    void read(const cv::FileNode & node);                   // OpenCV serialisation

    std::vector<cv::Point3f> gridPoints() const;
    friend std::ostream & operator<<(std::ostream &, const Chessboard &);
};

struct Camera;

struct ChessboardImage
{
    ChessboardImage(const cv::Mat &, const Chessboard &, const std::filesystem::path & = "");
    cv::Mat image;
    std::filesystem::path filename;
    Pose cameraPose;                                        // Extrinsic camera parameters
    std::vector<cv::Point2f> corners;                       // Chessboard corners in image [rQOi]
    bool isFound;
    void drawCorners(const Chessboard &);
    void drawBox(const Chessboard &, const Camera &);
    void recoverPose(const Chessboard &, const Camera &);
};

struct ChessboardData
{
    explicit ChessboardData(const std::filesystem::path &); // Load from config file

    Chessboard chessboard;
    std::vector<ChessboardImage> chessboardImages;

    void drawCorners();
    void drawBoxes(const Camera &);
    void recoverPoses(const Camera &);
};

namespace Eigen {
using Matrix23d = Eigen::Matrix<double, 2, 3>;
using Matrix26d = Eigen::Matrix<double, 2, 6>;
using Vector6d = Eigen::Matrix<double, 6, 1>;
template <typename Scalar> using Vector6 = Eigen::Matrix<Scalar, 6, 1>;
}

struct Camera
{
    void calibrate(ChessboardData &);                       // Calibrate camera from chessboard data
    void printCalibration() const;

    cv::Vec3d worldToVector(const cv::Vec3d & rPNn, const Pose & pose) const;
    template <typename Scalar> Eigen::Vector3<Scalar> worldToVector(const Eigen::Vector3<Scalar> & rPNn, const Eigen::Vector6<Scalar> & eta) const;

    cv::Vec2d worldToPixel(const cv::Vec3d &, const Pose &) const;
    template <typename Scalar> Eigen::Vector2<Scalar> worldToPixel(const Eigen::Vector3<Scalar> & rPNn, const Eigen::Vector6<Scalar> & eta) const;
    Eigen::Vector2d worldToPixel(const Eigen::Vector3d & rPNn, const Eigen::Vector6d & eta, Eigen::Matrix23d & JrPNn, Eigen::Matrix26d & Jeta) const;

    cv::Vec2d vectorToPixel(const cv::Vec3d &) const;
    template <typename Scalar> Eigen::Vector2<Scalar> vectorToPixel(const Eigen::Vector3<Scalar> &) const;
    Eigen::Vector2d vectorToPixel(const Eigen::Vector3d &, Eigen::Matrix23d &) const;

    cv::Vec3d pixelToVector(const cv::Vec2d &) const;

    bool isWorldWithinFOV(const cv::Vec3d & rPNn, const Pose & pose) const;
    bool isVectorWithinFOV(const cv::Vec3d & rPCc) const;

    void calcFieldOfView();
    void write(cv::FileStorage &) const;                    // OpenCV serialisation
    void read(const cv::FileNode &);                        // OpenCV serialisation

    cv::Mat cameraMatrix;                                   // Camera matrix
    cv::Mat distCoeffs;                                     // Lens distortion coefficients
    int flags = 0;                                          // Calibration flags
    cv::Size imageSize;                                     // Image size
    double hFOV = 0.0;                                      // Horizonal field of view
    double vFOV = 0.0;                                      // Vertical field of view
    double dFOV = 0.0;                                      // Diagonal field of view

    Eigen::Vector3d rCBb = Eigen::Vector3d::Zero();         // TODO: Assignment(s)
    Eigen::Matrix3d Rbc = Eigen::Matrix3d::Identity();      // TODO: Assignment(s)
};

template <typename Scalar>
Eigen::Vector3<Scalar> Camera::worldToVector(const Eigen::Vector3<Scalar> & rPNn, const Eigen::Vector6<Scalar> & eta) const
{
    Eigen::Vector3<Scalar> rCNn = eta.template head<3>();
    Eigen::Vector3<Scalar> Thetanc = eta.template tail<3>();
    Eigen::Matrix3<Scalar> Rnc = rpy2rot(Thetanc);
    Eigen::Vector3<Scalar> rPCc = Rnc.transpose()*(rPNn - rCNn);
    return rPCc;
}

template <typename Scalar>
Eigen::Vector2<Scalar> Camera::worldToPixel(const Eigen::Vector3<Scalar> & rPNn, const Eigen::Vector6<Scalar> & eta) const
{
    return vectorToPixel(worldToVector(rPNn, eta));
}

#include <cmath>
#include <opencv2/calib3d.hpp>

template <typename Scalar>
Eigen::Vector2<Scalar> Camera::vectorToPixel(const Eigen::Vector3<Scalar> & rPCc) const
{
    bool isRationalModel    = (flags & cv::CALIB_RATIONAL_MODEL) == cv::CALIB_RATIONAL_MODEL;
    bool isThinPrismModel   = (flags & cv::CALIB_THIN_PRISM_MODEL) == cv::CALIB_THIN_PRISM_MODEL;
    assert(isRationalModel && isThinPrismModel);

    Eigen::Vector2<Scalar> rQOi; 
    
    // Extract the x, y, and z components of rPCc
    Scalar x = rPCc(0);  // Assuming x() returns the x-component
    Scalar y = rPCc(1);  // Assuming y() returns the y-component
    Scalar z = rPCc(2);  // Assuming z() returns the z-component

    Scalar u = x/z;
    Scalar v = y/z;
    Scalar r = hypot(u,v);

    // Apply distortion model to correct pixel coordinates
    Scalar r2 = r * r;
    Scalar r4 = r2 * r2;
    Scalar r6 = r2 * r4;

    // Assuming you have distortion coefficients in distCoeffs
    Scalar k1 = distCoeffs.at<double>(0);  // First distortion coefficient
    Scalar k2 = distCoeffs.at<double>(1);  // Second distortion coefficient
    Scalar p1 = distCoeffs.at<double>(2);  // Third distortion coefficient
    Scalar p2 = distCoeffs.at<double>(3);  // Fourth distortion coefficient
    Scalar k3 = distCoeffs.at<double>(4);  // Fifth distortion coefficient
    Scalar k4 = distCoeffs.at<double>(5);  // Sixth distortion coefficient
    Scalar k5 = distCoeffs.at<double>(6);  // Seventh distortion coefficient
    Scalar k6 = distCoeffs.at<double>(7);  // Eigth distortion coefficient
    Scalar s1 = distCoeffs.at<double>(8);  // Nineth distortion coefficient
    Scalar s2 = distCoeffs.at<double>(9);  // Tenth distortion coefficient
    Scalar s3 = distCoeffs.at<double>(10);  // Eleventh distortion coefficient
    Scalar s4 = distCoeffs.at<double>(11);  // Twelvth distortion coefficient

    /*
    std::cout << "Camera.h k1 = " << k1 << std::endl;
    std::cout << "Camera.h k2 = " << k2 << std::endl;
    std::cout << "Camera.h p1 = " << p1 << std::endl;
    std::cout << "Camera.h p2 = " << p2 << std::endl;
    std::cout << "Camera.h k3 = " << k3 << std::endl;
    std::cout << "Camera.h k4 = " << k4 << std::endl;
    std::cout << "Camera.h k5 = " << k5 << std::endl;
    std::cout << "Camera.h k6 = " << k6 << std::endl;
    std::cout << "Camera.h s1 = " << s1 << std::endl;
    std::cout << "Camera.h s2 = " << s2 << std::endl;
    std::cout << "Camera.h s3 = " << s3 << std::endl;
    std::cout << "Camera.h s4 = " << s4 << std::endl;
    */
    
    // Assuming you have the camera matrix (3x3 matrix) as cameraMatrix 
    Scalar fx = cameraMatrix.at<double>(0, 0);  // Focal length in x
    Scalar fy = cameraMatrix.at<double>(1, 1);  // Focal length in y
    Scalar cx = cameraMatrix.at<double>(0, 2);  // Principal point x-coordinate
    Scalar cy = cameraMatrix.at<double>(1, 2);  // Principal point y-coordinate
    
    /*
    std::cout << "Camera.h cameraMatrix = " << cameraMatrix << std::endl;
    std::cout << "Camera.h fx = " << fx << std::endl;
    std::cout << "Camera.h fy = " << fy << std::endl;
    std::cout << "Camera.h cx = " << cx << std::endl;
    std::cout << "Camera.h cy = " << cy << std::endl;
    std::cout << "Camera.h r = " << r << std::endl;
    */
    
    // Calculate Radial Distortion
    Scalar alpha = k1*r2 + k2*r4 + k3*r6;
    Scalar beta = k4*r2 + k5*r4 + k6*r6;
    Scalar c = (1 + alpha)/(1 + beta);

    // Calculate Decentering Distortion
    Scalar dist_Top = 2 * p1 * u * v + p2 * ( r2 + 2 * u * u);
    Scalar dist_Bottom = p1 * ( r2 + 2 * v * v) + 2 * p2 * u * v;

    // Calculate Thin Prism Distortion
    Scalar thin_Top = s1 * r2 + s2 * r4;
    Scalar thin_Bottom = s3 * r2 + s4 * r4;

    // Full Distortion
    Scalar u_ = c * u + dist_Top + thin_Top; 
    Scalar v_ = c * v + dist_Bottom + thin_Bottom; 

    /*
    std::cout << "Camera.h u_ = " << u_ << std::endl;
    std::cout << "Camera.h v_ = " << v_ << std::endl;
    */
   
    // Final Values
    rQOi[0] = fx * u_ + cx;
    rQOi[1] = fy * v_ + cy;

    return rQOi;
}

#endif

