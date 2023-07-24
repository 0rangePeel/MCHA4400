#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/aruco.hpp>
#include "imagefeatures.h"

cv::Mat detectAndDrawHarris(const cv::Mat & img, int maxNumFeatures)
{
    cv::Mat imgout = img.clone();

    // Convert to greyscale
    cv::Mat grayImg;
    cv::cvtColor(img, grayImg, cv::COLOR_BGR2GRAY);

    // Harris tuning parameters
    int blockSize = 2;
    int apertureSize = 3;
    double k = 0.04;
    double thresh = 200.0;

    // Harris Preprocessing
    cv::Mat dst;
    cornerHarris(grayImg, dst, blockSize, apertureSize, k);

    cv::Mat dst_norm, dst_norm_scaled;
    normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    convertScaleAbs(dst_norm, dst_norm_scaled);
    
    int features = 0;

    // Initialise Harris Vectors
    std::vector<int> x; // rows
    std::vector<int> y; // columns
    std::vector<double> score; // columns

    // Run Harris Detector and plot circles on features
    for(int i = 0; i < dst_norm.rows ; i++) {
        for(int j = 0; j < dst_norm.cols; j++) {
            if((int) dst_norm.at<float>(i,j) > thresh) {
                circle(imgout, cv::Point(j,i), 5, cv::Scalar(0,0,255), 2, 8, 0);
                features += 1;
                x.push_back(i);
                y.push_back(j);
                score.push_back(dst.at<float>(i,j));
            }
        }
    }

    std::cout << "Features Detected: " << features << std::endl;


    // Initialize a vector to store the indices of the original array
    std::vector<int> indices(features);
    for (int i = 0; i < features; ++i) {
        indices[i] = i;
    }

    // Perform a selection sort on the indices based on the values in the original array
    for (int i = 0; i < features - 1; ++i) {
        int maxIndex = i;
        for (int j = i + 1; j < features; ++j) {
            if (score[indices[j]] > score[indices[maxIndex]]) {
                maxIndex = j;
            }
        }
        if (maxIndex != i) {
            std::swap(indices[i], indices[maxIndex]);
        }
    }

    // Uncomment for all feature information
    /*
    // Print all features
    for (int i = 0; i < features; ++i) {
        std::cout << "Entry " << i << ": x - " << x[i] << ": y - " << y[i] << ": score - " << score[i] << std::endl;
    }
    // Print the ordered array of positions (indices)
    std::cout << "Positions ordered from highest to lowest values: ";
    for (int i = 0; i < features; ++i) {
        std::cout << indices[i] << " ";
    }
    std::cout << std::endl;
    */

    // Print Max Features and plot on image
    for (int i =0; i < maxNumFeatures; ++i) {
        std::cout << "  idx: " << i << "  at point: (" << x[indices[i]] << "," << y[indices[i]] << ")      Harris Score: " << score[indices[i]] << std::endl;
        circle(imgout, cv::Point(y[indices[i]],x[indices[i]]), 5, cv::Scalar(0,255,0), 2, 8, 0);
        std::string indexText = std::to_string(i);
        cv::putText(imgout, indexText, cv::Point(y[indices[i]]+10, x[indices[i]]+10), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 255, 0), 2);
    }

    return imgout;
}

cv::Mat detectAndDrawShiAndTomasi(const cv::Mat & img, int maxNumFeatures)
{
    cv::Mat imgout = img.clone();

    // Convert to greyscale
    cv::Mat grayImg;
    cv::cvtColor(img, grayImg, cv::COLOR_BGR2GRAY);

    // Shi tuning parameters
    int myShiTomasi_qualityLevel = 50;
    int max_qualityLevel = 100;

    int blockSize = 3;
    int apertureSize = 3;

    double myShiTomasi_minVal;
    double myShiTomasi_maxVal;

    cv::Mat myShiTomasi_dst;
    cornerMinEigenVal( grayImg, myShiTomasi_dst, blockSize, apertureSize );
    minMaxLoc( myShiTomasi_dst, &myShiTomasi_minVal, &myShiTomasi_maxVal );

    int features = 0;

    // Initialise Shi Vectors
    std::vector<int> x; // rows
    std::vector<int> y; // columns
    std::vector<double> score; // columns

    myShiTomasi_qualityLevel = MAX(myShiTomasi_qualityLevel, 1);
    for(int i = 0; i < grayImg.rows; i++) {
        for(int j = 0; j < grayImg.cols; j++) {
            if(myShiTomasi_dst.at<float>(i,j) > myShiTomasi_minVal + ( myShiTomasi_maxVal - myShiTomasi_minVal )*myShiTomasi_qualityLevel/max_qualityLevel) {
                circle(imgout, cv::Point(j,i), 5, cv::Scalar(0,0,255), 2, 8, 0);
                features += 1;
                x.push_back(i);
                y.push_back(j);
                score.push_back(myShiTomasi_dst.at<float>(i,j));
            }
        }
    }

    std::cout << "Features Detected: " << features << std::endl;

    // Initialize a vector to store the indices of the original array
    std::vector<int> indices(features);
    for (int i = 0; i < features; ++i) {
        indices[i] = i;
    }

    // Perform a selection sort on the indices based on the values in the original array
    for (int i = 0; i < features - 1; ++i) {
        int maxIndex = i;
        for (int j = i + 1; j < features; ++j) {
            if (score[indices[j]] > score[indices[maxIndex]]) {
                maxIndex = j;
            }
        }
        if (maxIndex != i) {
            std::swap(indices[i], indices[maxIndex]);
        }
    }

    // Print Max Features and plot on image
    for (int i =0; i < maxNumFeatures; ++i) {
        std::cout << "  idx: " << i << "  at point: (" << x[indices[i]] << "," << y[indices[i]] << ")      Eigenvalue: " << score[indices[i]] << std::endl;
        circle(imgout, cv::Point(y[indices[i]],x[indices[i]]), 5, cv::Scalar(0,255,0), 2, 8, 0);
        std::string indexText = std::to_string(i);
        cv::putText(imgout, indexText, cv::Point(y[indices[i]]+10, x[indices[i]]+10), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 255, 0), 2);
    }
    
    return imgout;
}

cv::Mat detectAndDrawORB(const cv::Mat & img, int maxNumFeatures)
{
    cv::Mat imgout = img.clone();

    // Convert to greyscale
    cv::Mat grayImg;
    cv::cvtColor(img, grayImg, cv::COLOR_BGR2GRAY);

    // ORB Tuning Parameters
    int maxKeypoints = maxNumFeatures;

    // Initiate ORB detector
    cv::Ptr<cv::ORB> orb = cv::ORB::create(maxKeypoints);

    // Find the keypoints with ORB
    std::vector<cv::KeyPoint> kp;
    orb->detect(grayImg, kp, cv::Mat());

    // Compute the descriptors with ORB
    cv::Mat des;
    orb->compute(grayImg, kp, des);

    // Draw only keypoints location, not size and orientation
    cv::drawKeypoints(img, kp, imgout, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    std::cout << "Descriptor Width: " << des.cols << std::endl;

    std::cout << "Descriptor Height: " << des.rows << std::endl;

    // Print the ORB descriptors
    for (int i = 0; i < des.rows; ++i)
    {
        std::cout<<"[";
        for (int j = 0; j < des.cols; ++j)
        {
            //WTF
            uchar descriptor = des.at<uchar>(i,j);
            std::cout<<static_cast<int>(descriptor)<<" ";
            //Huh
            //std::cout << des.at<float>(i, j) << " ";
        }
        std::cout<<"]";
        std::cout << std::endl;
    }

    return imgout;
}

cv::Mat detectAndDrawArUco(const cv::Mat & img, int maxNumFeatures)
{
    cv::Mat imgout = img.clone();

    cv::aruco::DetectorParameters detectorParams = cv::aruco::DetectorParameters();
    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    cv::aruco::ArucoDetector detector(dictionary, detectorParams);
    
    std::vector<int> ids;
    std::vector<std::vector<cv::Point2f>> corners, rejected;

    detector.detectMarkers(img, corners, ids, rejected);

    cv::aruco::drawDetectedMarkers(imgout, corners, ids);

    int arucoNum = (int)ids.size();

    if (arucoNum > 0) {
        std::cout << "Number of ArUco in image: " << arucoNum << std::endl;
        for (int i = 0; i < arucoNum; ++i) {
            std::cout << "  idx: " << ids[i] << "  with corners: (" << corners[i][0];
            for (int j = 1; j < 4; j++){
                std::cout << ", " << corners[i][j];
            }
            std::cout << ")" << std::endl;
        }
    }
    else{
        std::cout << "Error: no Aruco tags found"  << std::endl;
    }
    
    return imgout;
}