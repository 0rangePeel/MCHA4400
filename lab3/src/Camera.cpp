#include <cassert>
#include <cstddef>
#include <cmath>
#include <limits>
#include <iostream>
#include <vector>
#include <filesystem>
#include <regex>
#include <opencv2/core/types.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/persistence.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/calib3d.hpp>
#include "Camera.h"

void Chessboard::write(cv::FileStorage& fs) const
{
    fs << "{"
       << "grid_width"  << boardSize.width
       << "grid_height" << boardSize.height
       << "square_size" << squareSize
       << "}";
}

void Chessboard::read(const cv::FileNode& node)
{
    node["grid_width"]  >> boardSize.width;
    node["grid_height"] >> boardSize.height;
    node["square_size"] >> squareSize;
}

std::vector<cv::Point3f> Chessboard::gridPoints() const
{
    std::vector<cv::Point3f> rPNn_all;
    rPNn_all.reserve(boardSize.height*boardSize.width);
    for (int i = 0; i < boardSize.height; ++i)
        for (int j = 0; j < boardSize.width; ++j)
            rPNn_all.push_back(cv::Point3f(j*squareSize, i*squareSize, 0));   
    return rPNn_all; 
}

std::ostream & operator<<(std::ostream & os, const Chessboard & chessboard)
{
    return os << "boardSize: " << chessboard.boardSize << ", squareSize: " << chessboard.squareSize;
}

ChessboardImage::ChessboardImage(const cv::Mat & image_, const Chessboard & chessboard, const std::filesystem::path & filename_)
    : image(image_)
    , filename(filename_)
    , isFound(false)
{
    cv::Mat grayImage;
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);

    // TODO:
    //  - Detect chessboard corners in image and set the corners member
    //  - (optional) Do subpixel refinement of detected corners
    // Using cv::findChessboardCorners and cv::cornerSubPix

    // Detect chessboard corners in image and set the corners member
    std::vector<cv::Point2f> corners;
    //bool found = cv::findChessboardCorners(image, cv::Size(8,8), corners,cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_FAST_CHECK);
    bool found = cv::findChessboardCorners(grayImage, chessboard.boardSize, corners,cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_FAST_CHECK);

    if (found) {
        // If corners are found, set the isFound flag to true and store the corners in the ChessboardImage object
        isFound = true;
        //cornersImage = corners;

        // Optional subpixel refinement of detected corners
        //cv::cornerSubPix(image, cornersImage, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1));
        cv::cornerSubPix(grayImage, corners, cv::Size(5, 5), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1));
    }

}

void ChessboardImage::drawCorners(const Chessboard & chessboard)
{
    cv::drawChessboardCorners(image, chessboard.boardSize, corners, isFound);
}

void ChessboardImage::drawBox(const Chessboard & chessboard, const Camera & camera)
{
    // TODO
}

void ChessboardImage::recoverPose(const Chessboard & chessboard, const Camera & camera)
{
    std::vector<cv::Point3f> rPNn_all = chessboard.gridPoints();

    cv::Mat Thetacn, rNCc;
    cv::solvePnP(rPNn_all, corners, camera.cameraMatrix, camera.distCoeffs, Thetacn, rNCc);

    cv::Mat Rcn;
    cv::Rodrigues(Thetacn, Rcn);
    cameraPose.Rnc = cv::Mat(Rcn.t());
    cameraPose.rCNn = cv::Mat(-Rcn.t()*rNCc);
}

ChessboardData::ChessboardData(const std::filesystem::path & configPath)
{
    assert(std::filesystem::exists(configPath));
    cv::FileStorage fs(configPath.string(), cv::FileStorage::READ);
    assert(fs.isOpened());
    cv::FileNode node = fs["chessboard_data"];
    node["chessboard"] >> chessboard;
    std::cout << "Chessboard: " << chessboard << std::endl;

    std::string pattern;
    node["file_regex"] >> pattern;
    fs.release();
    std::regex re(pattern, std::regex_constants::basic | std::regex_constants::icase);
    
    // Populate chessboard images from regex
    std::filesystem::path root = configPath.parent_path();
    std::cout << "Scanning directory " << root.string() << " for file pattern \"" << pattern << "\"" << std::endl;
    chessboardImages.clear();
    if (std::filesystem::exists(root) && std::filesystem::is_directory(root))
    {
        for (const auto & p : std::filesystem::recursive_directory_iterator(root))
        {
            if (std::filesystem::is_regular_file(p))
            {
                if (std::regex_match(p.path().filename().string(), re))
                {
                    std::cout << "Loading " << p.path().filename().string() << "..." << std::flush;
                    cv::Mat image = cv::imread(p.path().string());
                    std::cout << "done, detecting chessboard..." << std::flush;
                    ChessboardImage ci(image, chessboard, p.path().filename());
                    std::cout << (ci.isFound ? "found" : "not found") << std::endl;
                    if (ci.isFound)
                    {
                        chessboardImages.push_back(ci);
                    }
                }
            }
        }
    }
}

void ChessboardData::drawCorners()
{
    for (auto & chessboardImage : chessboardImages)
    {
        chessboardImage.drawCorners(chessboard);
    }
}

void ChessboardData::drawBoxes(const Camera & camera)
{
    for (auto & chessboardImage : chessboardImages)
    {
        chessboardImage.drawBox(chessboard, camera);
    }
}

void ChessboardData::recoverPoses(const Camera & camera)
{
    for (auto & chessboardImage : chessboardImages)
    {
        chessboardImage.recoverPose(chessboard, camera);
    }
}

void Camera::calibrate(ChessboardData & chessboardData)
{
    std::vector<cv::Point3f> rPNn_all = chessboardData.chessboard.gridPoints();

    std::vector<std::vector<cv::Point2f>> rQOi_all;
    for (const auto & chessboardImage : chessboardData.chessboardImages)
    {
        rQOi_all.push_back(chessboardImage.corners);
    }
    assert(!rQOi_all.empty());

    imageSize = chessboardData.chessboardImages[0].image.size();
    
    flags = cv::CALIB_RATIONAL_MODEL | cv::CALIB_THIN_PRISM_MODEL;

    // Find intrinsic and extrinsic camera parameters
    cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
    distCoeffs = cv::Mat::zeros(12, 1, CV_64F);
    std::vector<cv::Mat> Thetacn_all, rNCc_all;
    double rms;
    std::cout << "Calibrating camera..." << std::flush;
    
    // Prepare a vector of vector of 3D points for cv::calibrateCamera
    std::vector<std::vector<cv::Point3f>> objectPoints(chessboardData.chessboardImages.size(), rPNn_all);

    // TODO: Calibrate camera from detected chessboard corners
    // Calibrate the camera
    rms = cv::calibrateCamera(objectPoints, rQOi_all, imageSize, cameraMatrix, distCoeffs, rNCc_all, Thetacn_all, flags);

    std::cout << "done" << std::endl;
    
    // Calculate horizontal, vertical and diagonal field of view
    calcFieldOfView();

    // Write extrinsic parameters for each chessboard image
    assert(chessboardData.chessboardImages.size() == rNCc_all.size());
    assert(chessboardData.chessboardImages.size() == Thetacn_all.size());
    for (std::size_t i = 0; i < chessboardData.chessboardImages.size(); ++i)
    {
        cv::Mat Rcn;
        cv::Rodrigues(Thetacn_all[i], Rcn);
        Pose & cameraPose = chessboardData.chessboardImages[i].cameraPose;
        
        
        // TODO: Set the camera orientation and position (extrinsic camera parameters)
        // cameraPose.Rnc = ???;
        cameraPose.Rnc = cv::Matx33d(Rcn).t();

        // cameraPose.rCNn = ???;
        cameraPose.rCNn = -cameraPose.Rnc * cv::Vec3d(rNCc_all[i]);

    }
    
    printCalibration();
    std::cout << std::setw(30) << "RMS reprojection error: " << rms << std::endl;

    assert(cv::checkRange(cameraMatrix));
    assert(cv::checkRange(distCoeffs));
}

void Camera::printCalibration() const
{
    std::bitset<8*sizeof(flags)> bitflag(flags);
    std::cout << std::endl << "Calibration data:" << std::endl;
    std::cout << std::setw(30) << "Bit flags: " << bitflag << std::endl;
    std::cout << std::setw(30) << "cameraMatrix:\n" << cameraMatrix << std::endl;
    std::cout << std::setw(30) << "distCoeffs:\n" << distCoeffs.t() << std::endl;
    std::cout << std::setw(30) << "Focal lengths: " 
              << "(fx, fy) = "
              << "("<< cameraMatrix.at<double>(0, 0) << ", "<< cameraMatrix.at<double>(1, 1) << ")"
              << std::endl;       
    std::cout << std::setw(30) << "Principal point: " 
              << "(cx, cy) = "
              << "("<< cameraMatrix.at<double>(0, 2) << ", "<< cameraMatrix.at<double>(1, 2) << ")"
              << std::endl;     
    std::cout << std::setw(30) << "Field of view (horizontal): " << 180.0/CV_PI*hFOV << " deg" << std::endl;
    std::cout << std::setw(30) << "Field of view (vertical): " << 180.0/CV_PI*vFOV << " deg" << std::endl;
    std::cout << std::setw(30) << "Field of view (diagonal): " << 180.0/CV_PI*dFOV << " deg" << std::endl;
}

void Camera::calcFieldOfView()
{
    // In the Camera::calcFieldOfView function, calculate the horizontal, vertical and diagonal field
    // of view of the camera, using the intrinsic parameters of the camera. This can be done by finding
    // the angles between direction vectors that correspond to some of the edge and corner pixels in
    // the image with the help of Camera::pixelToVector.

    assert(cameraMatrix.rows == 3);
    assert(cameraMatrix.cols == 3);
    assert(cameraMatrix.type() == CV_64F);

    // TODO:
    /*
    hFOV = 0.0;
    vFOV = 0.0;
    dFOV = 0.0;
    */

    // Define four corners and center of the image in pixel coordinates
    cv::Vec2d topLeft(0.0, 0.0);
    cv::Vec2d topRight(imageSize.width - 1.0, 0.0);
    cv::Vec2d bottomLeft(0.0, imageSize.height - 1.0);
    cv::Vec2d bottomRight(imageSize.width - 1.0, imageSize.height - 1.0);
    cv::Vec2d center(imageSize.width / 2.0, imageSize.height / 2.0);

    // Convert pixel coordinates to unit vectors in camera coordinates
    cv::Vec3d topDir = pixelToVector(topLeft);
    cv::Vec3d rightDir = pixelToVector(topRight);
    cv::Vec3d bottomDir = pixelToVector(bottomLeft);
    //cv::Vec3d leftDir = pixelToVector(bottomRight);
    cv::Vec3d centerDir = pixelToVector(center);

    // Calculate horizontal and vertical FOV as the angles between the corresponding vectors
    hFOV = std::acos(topDir.dot(rightDir));
    vFOV = std::acos(topDir.dot(bottomDir));

    // Calculate diagonal FOV as the angle between the top-left and bottom-right corner vectors
    dFOV = std::acos(topDir.dot(centerDir));

    std::cout << "hFOV: " << hFOV << " vFOV: " << vFOV << " dFOV: " << dFOV << std::endl;

}

cv::Vec3d Camera::worldToVector(const cv::Vec3d & rPNn, const Pose & pose) const
{
    cv::Vec3d uPCc;
    // TODO: Compute the unit vector uPCc from the world position rPNn and camera pose
    // by doing the following mathematically
    // uPCc = rPCc/norm(rPCc)
    // Convert the world point to a camera point
    cv::Vec3d rPCc = pose.Rnc.t() * (rPNn - pose.rCNn);

    // Normalize the vector
    uPCc = rPCc / cv::norm(rPCc);
    
    return uPCc;
}

cv::Vec2d Camera::worldToPixel(const cv::Vec3d & rPNn, const Pose & pose) const
{
    return vectorToPixel(worldToVector(rPNn, pose));
}

cv::Vec2d Camera::vectorToPixel(const cv::Vec3d & rPCc) const
{
    //In the Camera::vectorToPixel function, use cv::projectPoints to map from a vector in
    //camera coordinates, rPCc, to a pixel location in image coordinates, rQOi
    //cv::Vec2d rQOi;
    // TODO: Compute the pixel location (rQOi) for the given unit vector (uPCc)
    //return rQOi;
    // Project the given vector (rPCc) to a pixel location in image coordinates (rQOi)

    cv::Vec2d rQOi;

    // Form a vector of object points from the unit vector
    std::vector<cv::Vec3d> objectPoints{rPCc};

    // Vector to hold output image points
    std::vector<cv::Vec2d> imagePoints;

    // Project the 3D points to 2D image points
    cv::projectPoints(objectPoints, cv::Vec3d::zeros(), cv::Vec3d::zeros(), cameraMatrix, distCoeffs, imagePoints);

    // Extract the first (and only) projected image point
    rQOi = imagePoints[0];


    return rQOi;
}

cv::Vec3d Camera::pixelToVector(const cv::Vec2d & rQOi) const
{
    //In the Camera::pixelToVector function, return the unit vector uPCc that corresponds to the
    //provided pixel coordinates rQOi. This can be done with the help of the cv::undistortPoints function.
    //cv::Vec3d uPCc;
    // TODO: Compute unit vector (uPCc) for the given pixel location (rQOi)
    // Undistort the pixel coordinates to remove lens distortion

    cv::Mat rQOiMat = (cv::Mat_<double>(1, 2) << rQOi[0], rQOi[1]);
    cv::Mat rQOiUndistorted;
    cv::undistortPoints(rQOiMat, rQOiUndistorted, cameraMatrix, distCoeffs);

    // Convert the undistorted pixel coordinates to a unit vector in camera coordinates
    cv::Vec3d uPCc = cv::Vec3d(rQOiUndistorted.at<double>(0, 0), rQOiUndistorted.at<double>(0, 1), 1.0);
    uPCc /= cv::norm(uPCc); // Normalize the vector to get the unit vector

    return uPCc;
}

bool Camera::isVectorWithinFOV(const cv::Vec3d & rPCc) const
{
    // In the Camera::isVectorWithinFOV function, determine if the 
    // provided vector rPCc lies within the camera field of view.

    /////////// TODO: Check if uPCc lies in the image //////////////////////

    cv::Vec3d uPCc = rPCc/cv::norm(rPCc);
    assert(std::abs(cv::norm(uPCc) - 1.0) < 100*std::numeric_limits<double>::epsilon());

    // Project the vector to pixel coordinates
    cv::Vec2d rQOi = vectorToPixel(uPCc);

    // Check if the pixel is within the image bounds
    if (rQOi[0] >= 0 && rQOi[0] < imageSize.width && rQOi[1] >= 0 && rQOi[1] < imageSize.height)
    {
        return false;
    }
    else
    {
        return true;
    }
}

bool Camera::isWorldWithinFOV(const cv::Vec3d & rPNn, const Pose & pose) const
{
    return isVectorWithinFOV(worldToVector(rPNn, pose));
}

void Camera::write(cv::FileStorage & fs) const
{
    fs << "{"
       << "camera_matrix"           << cameraMatrix
       << "distortion_coefficients" << distCoeffs
       << "flags"                   << flags
       << "imageSize"               << imageSize
       << "}";
}

void Camera::read(const cv::FileNode & node)
{
    node["camera_matrix"]           >> cameraMatrix;
    node["distortion_coefficients"] >> distCoeffs;
    node["flags"]                   >> flags;
    node["imageSize"]               >> imageSize;

    calcFieldOfView();

    assert(cameraMatrix.cols == 3);
    assert(cameraMatrix.rows == 3);
    assert(cameraMatrix.type() == CV_64F);
    assert(distCoeffs.cols == 1);
    assert(distCoeffs.type() == CV_64F);
}
