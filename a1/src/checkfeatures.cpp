#include <iostream>
#include <Eigen/Core>
#include <cmath>
#include "Camera.h"

bool isPointInsideEllipse(int x, int y, const Camera &cam) {
    double a = 920.0; // Semi-major axis length
    double b = 460.0; // Semi-minor axis length
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
    double lhs = sqrt(((x - centerX) * (x - centerX) / (a * a)) + ((y - centerY) * (y - centerY) / (b * b)));
    if (lhs <= 1.0) {
        return true;
    } else {
        return false;
    }
}