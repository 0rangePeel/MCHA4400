#include <iostream>
#include <vector>
#include <algorithm>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/aruco.hpp>
#include "imagefeatures.h"
#include "Camera.h"
#include <Eigen/Core>

PointFeature::PointFeature()
    : score(0)
    , x(0)
    , y(0)
{}

PointFeature::PointFeature(const double & score_, const double & x_, const double & y_)
    : score(score_)
    , x(x_)
    , y(y_)
{}

bool PointFeature::operator<(const PointFeature & other) const
{
    return (score > other.score);
}

std::vector<PointFeature> detectFeatures(const cv::Mat & img, const int & maxNumFeatures)
{
    std::vector<PointFeature> features;
    // TODO: Lab 8
    // Choose a suitable feature detector
    // Save features above a certain texture threshold
    // Sort features by texture
    // Cap number of features to maxNumFeatures

    // Shi and Tomasi

    //cv::Mat imgout = img.clone();

    // Convert to greyscale
    cv::Mat grayImg;
    cv::cvtColor(img, grayImg, cv::COLOR_BGR2GRAY);

    // Shi tuning parameters
    int myShiTomasi_qualityLevel = 40;
    int max_qualityLevel = 500;

    int blockSize = 3;
    int apertureSize = 3;

    double myShiTomasi_minVal;
    double myShiTomasi_maxVal;

    cv::Mat myShiTomasi_dst;
    cornerMinEigenVal( grayImg, myShiTomasi_dst, blockSize, apertureSize );
    minMaxLoc( myShiTomasi_dst, &myShiTomasi_minVal, &myShiTomasi_maxVal );

    int NumFeatures = 0;

    // Initialise Shi Vectors
    std::vector<int> x; // rows
    std::vector<int> y; // columns
    std::vector<double> score; // columns

    myShiTomasi_qualityLevel = MAX(myShiTomasi_qualityLevel, 1);
    for(int i = 0; i < grayImg.rows; i++) {
        for(int j = 0; j < grayImg.cols; j++) {
            if(myShiTomasi_dst.at<float>(i,j) > myShiTomasi_minVal + ( myShiTomasi_maxVal - myShiTomasi_minVal )*myShiTomasi_qualityLevel/max_qualityLevel) {
                circle(img, cv::Point(j,i), 5, cv::Scalar(0,0,255), 2, 8, 0);
                NumFeatures += 1;
                x.push_back(j);
                y.push_back(i);
                score.push_back(myShiTomasi_dst.at<float>(i,j));
            }
        }
    }

    std::cout << "Features Detected: " << NumFeatures << std::endl;

    // Initialize a vector to store the indices of the original array
    std::vector<int> indices(NumFeatures);
    for (int i = 0; i < NumFeatures; ++i) {
        indices[i] = i;
    }

    // Perform a selection sort on the indices based on the values in the original array
    for (int i = 0; i < NumFeatures - 1; ++i) {
        int maxIndex = i;
        for (int j = i + 1; j < NumFeatures; ++j) {
            if (score[indices[j]] > score[indices[maxIndex]]) {
                maxIndex = j;
            }
        }
        if (maxIndex != i) {
            std::swap(indices[i], indices[maxIndex]);
        }
    }

    // Print Max Features
    for (int i = 0; i < std::min(NumFeatures,maxNumFeatures); ++i) {
        // std::cout << "  idx: " << i << "  at point: (" << x[indices[i]] << "," << y[indices[i]] << ")      Eigenvalue: " << score[indices[i]] << std::endl;
        PointFeature feature(score[indices[i]], x[indices[i]], y[indices[i]]);
        features.push_back(feature);
    }
    

    return features;
}

ArUcoResult detectAndDrawArUco(const cv::Mat &img, const Camera &cam) {
    ArUcoResult result;

    cv::Mat imgout = img.clone();

    cv::aruco::DetectorParameters detectorParams = cv::aruco::DetectorParameters();
    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    cv::aruco::ArucoDetector detector(dictionary, detectorParams);
    
    std::vector<int> ids;
    std::vector<std::vector<cv::Point2f>> corners, rejected;

    detector.detectMarkers(img, corners, ids, rejected);

    
    // Optimised aruco tag corner detector
    Eigen::VectorXd y_array(8*corners.size());
    for (int i = 0; i < corners.size(); i++){
        for (int j = 0; j < corners[i].size(); j++){
            y_array(8 * i + 2 * j) = corners[i][j].x;
            y_array(8 * i + 2 * j + 1) = corners[i][j].y;
        }  
    }
    

    // To get cool pnp stuff //
    
    /*
    cv::Mat cameraMatrix, distCoeffs;

    cameraMatrix = cam.cameraMatrix;
    distCoeffs = cam.distCoeffs.t();

    //float markerLength = 0.05;
    float markerLength = 0.166;

    // Set coordinate system
    cv::Mat objPoints(4, 1, CV_32FC3);
    objPoints.ptr<cv::Vec3f>(0)[0] = cv::Vec3f(-markerLength/2.f, markerLength/2.f, 0);
    objPoints.ptr<cv::Vec3f>(0)[1] = cv::Vec3f(markerLength/2.f, markerLength/2.f, 0);
    objPoints.ptr<cv::Vec3f>(0)[2] = cv::Vec3f(markerLength/2.f, -markerLength/2.f, 0);
    objPoints.ptr<cv::Vec3f>(0)[3] = cv::Vec3f(-markerLength/2.f, -markerLength/2.f, 0);
    
    Eigen::VectorXd y_array;
    // If at least one marker detected
    if (ids.size() > 0){
        cv::aruco::drawDetectedMarkers(imgout, corners, ids);
        int nMarkers = corners.size();
        std::vector<cv::Vec3d> rvecs(nMarkers), tvecs(nMarkers);
        // Calculate pose for each marker
        for (int i = 0; i < nMarkers; i++) {
            solvePnP(objPoints, corners.at(i), cameraMatrix, distCoeffs, rvecs.at(i), tvecs.at(i));
            for (int j = 0; j < corners[i].size(); j++) {
                    y_array.conservativeResize(y_array.size() + 2);
                    y_array(y_array.size() - 2) = corners[i][j].x;
                    y_array(y_array.size() - 1) = corners[i][j].y;
            }
        }
        // Draw axis for each marker - note 0.01 is the length of the axis
        for(unsigned int i = 0; i < ids.size(); i++) {
            cv::drawFrameAxes(imgout, cameraMatrix, distCoeffs, rvecs[i], tvecs[i], 0.01);
        }
    }
    */

    // set results
    result.imgout = imgout;
    result.ids = ids;
    result.corners = corners;
    result.y = y_array;

    return result;
}
