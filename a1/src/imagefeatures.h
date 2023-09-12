#ifndef IMAGEFEATURES_H
#define IMAGEFEATURES_H 

#include <opencv2/core.hpp>
#include "Camera.h"

struct ArUcoResult {
    cv::Mat imgout;
    std::vector<int> ids;
    std::vector<std::vector<cv::Point2f>> corners;
};

ArUcoResult detectAndDrawArUco(const cv::Mat &img, int maxNumFeatures, const Camera &cam);

#endif