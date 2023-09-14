#ifndef IMAGEFEATURES_H
#define IMAGEFEATURES_H 

#include <opencv2/core.hpp>
#include "Camera.h"
#include <Eigen/Core>

struct ArUcoResult {
    cv::Mat imgout;
    std::vector<int> ids;
    std::vector<std::vector<cv::Point2f>> corners;
    Eigen::VectorXd y;
};

ArUcoResult detectAndDrawArUco(const cv::Mat &img, int maxNumFeatures, const Camera &cam);

#endif