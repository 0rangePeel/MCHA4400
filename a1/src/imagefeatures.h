#ifndef IMAGEFEATURES_H
#define IMAGEFEATURES_H 

#include <opencv2/core.hpp>
#include "Camera.h"
#include <Eigen/Core>

struct PointFeature
{
    PointFeature();
    PointFeature(const double & score_, const double & x_, const double & y_);
    double score, x, y;
    bool operator<(const PointFeature & other) const;   // used for std::sort
};

std::vector<PointFeature> detectFeatures(const cv::Mat & img, const int & maxNumFeatures);

struct ArUcoResult {
    cv::Mat imgout;
    std::vector<int> ids;
    std::vector<std::vector<cv::Point2f>> corners;
    Eigen::VectorXd y;
};

ArUcoResult detectAndDrawArUco(const cv::Mat &img, const Camera &cam);

#endif