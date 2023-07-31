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
    // TODO:
    //  - Detect chessboard corners in image and set the corners member
    //  - (optional) Do subpixel refinement of detected corners
    // Convert image to grayscale for corner detection
    cv::Mat grayImage;
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
    
    // Detect the corners in the image
    isFound = cv::findChessboardCorners(grayImage, chessboard.boardSize, corners, 
        cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK);

    if (isFound)
    {
        // Optional: refine the corner locations at the subpixel level
        cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1);
        cv::cornerSubPix(grayImage, corners, cv::Size(5,5), cv::Size(-1,-1), criteria);
    }

}

void ChessboardImage::drawCorners(const Chessboard & chessboard)
{
    cv::drawChessboardCorners(image, chessboard.boardSize, corners, isFound);
}

void ChessboardImage::drawBox(const Chessboard & chessboard, const Camera & camera)
{
    // TODO
    this->recoverPose(chessboard, camera);

    cv::Vec3d startPoint(0, 0, 0);
    cv::Vec2d origin = camera.worldToPixel(startPoint, cameraPose);

    cv::Vec3d nextPoint(9*0.022, 6*0.022, 0);
    cv::Vec2d point = camera.worldToPixel(nextPoint, cameraPose);

    
    cv::line(image, cv::Point (origin[0], origin[1]), cv::Point (point[0], point[1]), cv::Scalar(0,255,0));
    
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
    assert(cameraMatrix.rows == 3);
    assert(cameraMatrix.cols == 3);
    assert(cameraMatrix.type() == CV_64F);

    // TODO:
    // Get the center, corner, and edge points in the image
    cv::Vec2d centerPoint(imageSize.width / 2.0, imageSize.height / 2.0);
    cv::Vec2d cornerPoint(0, 0);
    cv::Vec2d edgePointX(imageSize.width, imageSize.height / 2.0);
    cv::Vec2d edgePointY(imageSize.width / 2.0, imageSize.height);

    // Compute the direction vectors for these points
    cv::Vec3d centerVector = pixelToVector(centerPoint);
    cv::Vec3d cornerVector = pixelToVector(cornerPoint);
    cv::Vec3d edgeVectorX = pixelToVector(edgePointX);
    cv::Vec3d edgeVectorY = pixelToVector(edgePointY);

    // Compute the dot product between the center vector and other vectors
    double centerCornerDotProduct = centerVector.dot(cornerVector);
    double centerEdgeDotProductX = centerVector.dot(edgeVectorX);
    double centerEdgeDotProductY = centerVector.dot(edgeVectorY);

    // Compute the magnitudes of the vectors
    double centerVectorMagnitude = cv::norm(centerVector);
    double cornerVectorMagnitude = cv::norm(cornerVector);
    double edgeVectorMagnitudeX = cv::norm(edgeVectorX);
    double edgeVectorMagnitudeY = cv::norm(edgeVectorY);

    // Compute the field of view angles using the dot product formula: dot(v1, v2) = ||v1|| ||v2|| cos(theta)
    hFOV = std::acos(centerEdgeDotProductX / (centerVectorMagnitude * edgeVectorMagnitudeX));
    vFOV = std::acos(centerEdgeDotProductY / (centerVectorMagnitude * edgeVectorMagnitudeY));
    dFOV = std::acos(centerCornerDotProduct / (centerVectorMagnitude * cornerVectorMagnitude));
    
}

cv::Vec3d Camera::worldToVector(const cv::Vec3d & rPNn, const Pose & pose) const
{
    cv::Vec3d uPCc;
    // TODO: Compute the unit vector uPCc from the world position rPNn and camera pose
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
    cv::Vec2d rQOi;
    // TODO: Compute the pixel location (rQOi) for the given unit vector (uPCc)
    cv::Vec3d uPCc = rPCc/cv::norm(rPCc);

    // Form a vector of object points from the unit vector
    std::vector<cv::Vec3d> objectPoints{uPCc};

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
    cv::Vec3d uPCc;
    // TODO: Compute unit vector (uPCc) for the given pixel location (rQOi)
    // Form a vector of image points from the given pixel coordinate
    std::vector<cv::Vec2d> imagePoints{rQOi};
    
    // Vector to hold output object points
    std::vector<cv::Vec2d> objectPoints;

    // Convert the 2D image points to normalized 2D points
    cv::undistortPoints(imagePoints, objectPoints, cameraMatrix, distCoeffs);

    // Convert the 2D point to a 3D unit vector
    uPCc = cv::Vec3d(objectPoints[0][0], objectPoints[0][1], 1.0);

    // Normalize the vector
    uPCc = uPCc / cv::norm(uPCc);
    
    return uPCc;
}

bool Camera::isVectorWithinFOV(const cv::Vec3d & rPCc) const
{
    cv::Vec3d uPCc = rPCc/cv::norm(rPCc);
    assert(std::abs(cv::norm(uPCc) - 1.0) < 100*std::numeric_limits<double>::epsilon());
    
    /////////// TODO: Check if uPCc lies in the image //////////////////////
    // Project the vector to pixel coordinates
    cv::Vec2d rQOi = vectorToPixel(uPCc);

    // Check if the pixel is within the image bounds
    if (rQOi[0] >= 0 && rQOi[0] < hFOV && rQOi[1] >= 0 && rQOi[1] < vFOV)
    {
        return true;
    }
    else
    {
        return false;
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