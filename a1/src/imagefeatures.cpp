#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/aruco.hpp>
#include "imagefeatures.h"
#include "Camera.h"

ArUcoResult detectAndDrawArUco(const cv::Mat &img, int maxNumFeatures, const Camera &cam) {
    ArUcoResult result;

    cv::Mat imgout = img.clone();

    cv::Mat cameraMatrix, distCoeffs;

    cameraMatrix = cam.cameraMatrix;
    distCoeffs = cam.distCoeffs.t();

    float markerLength = 0.05;

    // Set coordinate system
    cv::Mat objPoints(4, 1, CV_32FC3);
    objPoints.ptr<cv::Vec3f>(0)[0] = cv::Vec3f(-markerLength/2.f, markerLength/2.f, 0);
    objPoints.ptr<cv::Vec3f>(0)[1] = cv::Vec3f(markerLength/2.f, markerLength/2.f, 0);
    objPoints.ptr<cv::Vec3f>(0)[2] = cv::Vec3f(markerLength/2.f, -markerLength/2.f, 0);
    objPoints.ptr<cv::Vec3f>(0)[3] = cv::Vec3f(-markerLength/2.f, -markerLength/2.f, 0);

    cv::aruco::DetectorParameters detectorParams = cv::aruco::DetectorParameters();
    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    cv::aruco::ArucoDetector detector(dictionary, detectorParams);
    
    std::vector<int> ids;
    std::vector<std::vector<cv::Point2f>> corners, rejected;

    detector.detectMarkers(img, corners, ids, rejected);

    // If at least one marker detected
    if (ids.size() > 0){
        cv::aruco::drawDetectedMarkers(imgout, corners, ids);
        int nMarkers = corners.size();
        std::vector<cv::Vec3d> rvecs(nMarkers), tvecs(nMarkers);
        // Calculate pose for each marker
        for (int i = 0; i < nMarkers; i++) {
            solvePnP(objPoints, corners.at(i), cameraMatrix, distCoeffs, rvecs.at(i), tvecs.at(i));
        }
        // Draw axis for each marker - note 0.01 is the length of the axis
        for(unsigned int i = 0; i < ids.size(); i++) {
            cv::drawFrameAxes(imgout, cameraMatrix, distCoeffs, rvecs[i], tvecs[i], 0.01);
        }
    }

    // set results
    result.imgout = imgout;
    result.ids = ids;
    result.corners = corners;

    return result;
}