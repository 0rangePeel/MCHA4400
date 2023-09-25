#include <iostream>
#include <Eigen/Core>
#include <cmath>
#include "Camera.h"
#include "imagefeatures.h"
#include "checkfeatures.h"

bool isPointInsideEllipse(int x, int y, const Camera &cam) {
    //double a = 920.0; // Semi-major axis length
    //double b = 460.0; // Semi-minor axis length
    double a = 1920.0/2; // Semi-major axis length
    double b = 1080.0/2; // Semi-minor axis length
    //double centerX = 968.853; // x-coordinate of the ellipse center
    //double centerY = 581.045; // y-coordinate of the ellipse center
    double centerX = cam.cameraMatrix.at<double>(0, 2); // x-coordinate of the ellipse center
    double centerY = cam.cameraMatrix.at<double>(1, 2); // y-coordinate of the ellipse center

    // Bounding box check
    if (x < centerX - a || x > centerX + a || y < centerY - b || y > centerY + b) {
        return false;
    }

    // Distance check
    //double lhs = ((x - centerX) * (x - centerX) / (a * a)) + ((y - centerY) * (y - centerY) / (b * b));
    //double lhs = sqrt(((x - centerX) * (x - centerX) / (a * a)) + ((y - centerY) * (y - centerY) / (b * b)));
    double lhs = ((x - a) * (x - a) / (a * a)) + ((y - b) * (y - b) / (b * b));
    if (lhs <= 1.0) {
        return true;
    } else {
        return false;
    }
}

checkFeatureResult checkfeature(const ArUcoResult &arucoResult, const Camera &cam) {
    checkFeatureResult result;

    std::vector<int> ids = arucoResult.ids;
    Eigen::VectorXd y = arucoResult.y;

    for (int i = 0; i < y.size(); i = i + 8) {
        for (int j = 0; j < 8; j = j + 2){
            if (isPointInsideEllipse(y(i + j), y(i + j + 1), cam) == 0){
                //std::cout << "Aruco Tag: " << ids[i/8] << " has corner/corners which lies outside ellipse" << std::endl;
                //std::cout << "(" << y(i + j) << "," << y(i + j + 1) << ") - Lies outside the ellipse" << std::endl;
                // Remove the i-th aruco tag
                ids.erase(ids.begin() + i/8); 
                Eigen::VectorXd modifiedVector(y.size() - 8);
                modifiedVector << y.segment(0, i), y.segment(i + 8, y.size() - i - 8);
                y = modifiedVector;
                i = i - 8;
                break;
            }
        }
    }

    result.ids = ids;
    result.y = y;

    return result;
}