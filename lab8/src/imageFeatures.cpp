#include <iostream>
#include <vector>
#include <algorithm>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>

#include "imageFeatures.h"

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

    cv::Mat imgout = img.clone();

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
                circle(imgout, cv::Point(i,j), 5, cv::Scalar(0,0,255), 2, 8, 0);
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
