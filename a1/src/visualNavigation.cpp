#include <filesystem>
#include <string>
#include <iostream>
#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgcodecs.hpp>
#include "BufferedVideo.h"
#include "visualNavigation.h"
#include "rotation.hpp"
#include "Camera.h"
#include "Plot.h"
#include "StateSLAMPointLandmarks.h"
#include "StateSLAMPoseLandmarks.h"
#include "imagefeatures.h"
#include "MeasurementTagBundle.h"
#include "checkfeatures.h"

void runVisualNavigationFromVideo(const std::filesystem::path & videoPath, const std::filesystem::path & cameraPath, int scenario, int interactive, const std::filesystem::path & outputDirectory)
{
    assert(!videoPath.empty());

    // Output video path
    std::filesystem::path outputPath;
    bool doExport = !outputDirectory.empty();
    if (doExport)
    {
        std::string outputFilename = videoPath.stem().string()
                                   + "_out"
                                   + videoPath.extension().string();
        outputPath = outputDirectory / outputFilename;
    }

    // Load camera calibration
    Camera cam;
    if (!std::filesystem::exists(cameraPath))
    {
        std::cout << "File: " << cameraPath << " does not exist" << std::endl;
    }
    cv::FileStorage fs(cameraPath.string(), cv::FileStorage::READ);
    assert(fs.isOpened());
    fs["camera"] >> cam;

    // Display loaded calibration data
    cam.printCalibration();

    // Open input video
    cv::VideoCapture cap(videoPath.string());
    assert(cap.isOpened());
    int nFrames = cap.get(cv::CAP_PROP_FRAME_COUNT);
    std::cout << "Total Number of Video Frames: " << nFrames << std::endl;
    assert(nFrames > 0);
    
    double fps = cap.get(cv::CAP_PROP_FPS);

    BufferedVideoReader bufferedVideoReader(5);
    bufferedVideoReader.start(cap);

    cv::VideoWriter videoOut;
    BufferedVideoWriter bufferedVideoWriter(3);
    if (doExport)
    {
        cv::Size frameSize;
        //frameSize.width     = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        frameSize.width     = 1624;
        //frameSize.height    = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        frameSize.height    = 540;
        double outputFps    = fps;
        int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        //videoOut.open(outputPath.string(), codec, outputFps, frameSize);
        videoOut.open(outputPath.string(), fourcc, outputFps, frameSize);
        bufferedVideoWriter.start(videoOut);
    }

    // Visual Navigation Initialisation //
    // Initial state mean
    /*
    Eigen::VectorXd mu(12);
    mu.setZero();
    // Initial state square-root covariance
    Eigen::MatrixXd S(12,12);
    S.setIdentity();  

    //StateSLAMPoseLandmarks state(Gaussian(mu, S));
    StateSLAMPointLandmarks state(Gaussian(mu, S));
    

    // Scenario Initialisation
    switch (scenario) {
        case 1:
            // Initial state mean
            Eigen::VectorXd mu(12);
            mu.setZero();
            mu(8)  = -1.7; // height (I may be projecting)
            mu(9)  = -3 * M_PI/180; // roll
            mu(10) = -4 * M_PI/180; // pitch
            mu(11) = -1 * M_PI/180; // yaw


            // Initial state square-root covariance
            Eigen::MatrixXd S(12,12);
            S.setIdentity();
            //S.diagonal().array() = 0.1;
            //S.diagonal().array() = 0.01;

            
            for (int i = 0; i < 6; ++i)
            {
                S(i, i) = 0.1;
            }
            for (int i = 6; i < 12; ++i)
            {
                S(i, i) = 0.005;
            }
            
            // Initialise state
            StateSLAMPoseLandmarks state(Gaussian(mu, S));

            break;
        case 2:
            Eigen::VectorXd mu(12);
            mu.setZero();
            // Initial state square-root covariance
            Eigen::MatrixXd S(12,12);
            S.setIdentity();  

            StateSLAMPointLandmarks state(Gaussian(mu, S));
            break;
        default:
            std::cout << "Incorrenct Scenario - Must be either 1 or 2" << std::endl;
    }
    */
    /*
    // Initial state mean
    Eigen::VectorXd mu(12);
    mu.setZero();
    mu(8)  = -1.7; // height (I may be projecting)
    mu(9)  = -3 * M_PI/180; // roll
    mu(10) = -4 * M_PI/180; // pitch
    mu(11) = -1 * M_PI/180; // yaw


    // Initial state square-root covariance
    Eigen::MatrixXd S(12,12);
    S.setIdentity();
    //S.diagonal().array() = 0.1;
    //S.diagonal().array() = 0.01;

    
    for (int i = 0; i < 6; ++i)
    {
        S(i, i) = 0.1;
    }
    for (int i = 6; i < 12; ++i)
    {
        S(i, i) = 0.005;
    }
    
    // Initialise state
    StateSLAMPoseLandmarks state(Gaussian(mu, S));
    */
    
    Eigen::VectorXd mu(12);
    mu.setZero();
    // Initial state square-root covariance
    Eigen::MatrixXd S(12,12);
    S.setIdentity();  

    StateSLAMPointLandmarks state(Gaussian(mu, S));
    



    // Initialise plot
    Plot plot(state, cam);

    int i = 0;
    double t = 0;

    while (true)
    {
        std::cout << "Frame: " << i+1 << std::endl;
        // Get next input frame
        cv::Mat imgin = bufferedVideoReader.read();
        if (imgin.empty())
        {
            break;
        }

        switch (scenario) {
            case 1: {
                std::cout << "Scenario 1" << std::endl;
                ArUcoResult arucoResult = detectAndDrawArUco(imgin, cam);
                checkFeatureResult checkfeatureResult = checkfeature(arucoResult,cam);

                //Set idsLandmarks 
                //state.setIdsLandmarks(arucoResult.ids);
                //MeasurementTagBundle MeasurementTagBundle(t, arucoResult.y, cam);

                state.setIdsLandmarks(checkfeatureResult.ids);
                MeasurementTagBundle MeasurementTagBundle(t, checkfeatureResult.y, cam);

                MeasurementTagBundle.process(state);
                break;
            }
            case 2: {
                std::cout << "Scenario 2" << std::endl;
                int maxNumFeatures = 10;
                std::vector<PointFeature> features = detectFeatures(imgin, maxNumFeatures);
                std::cout << features.size() << " features found in image"  << std::endl;
                assert(features.size() > 0);
                assert(features.size() <= maxNumFeatures);

                break;
            }
            default:
                std::cout << "Incorrenct Scenario - Must be either 1 or 2" << std::endl;
        }
        

        //cv::Mat outputframe = arucoResult.imgout;
        //state.view() = outputframe.clone();
        
        state.view() = imgin;
        
        // Set local copy of state for plot to use
        plot.setState(state);
       
        // -Update plot
        plot.render();

        i += 1;

        t = t + 1/fps;

        // Write output frame 
        if (doExport)
        {
            cv::Mat imgout = plot.getFrame();
            //std::cout << imgout.size() << std::endl;
            bufferedVideoWriter.write(imgout);
        }

        

        
        if (interactive == 2 || (interactive == 1 && i + 1 == nFrames))
        { 
            /*
            // Print the detected ARUCO tags and corners for when frame is stopped
            std::cout << "Detected Marker Corners:" << std::endl;
            for (int i = 0; i < arucoResult.corners.size(); i++) {
                std::cout << "Marker " << arucoResult.ids[i] << " Corners: ";
                for (int j = 0; j < arucoResult.corners[i].size(); j++) {
                    std::cout << "(" << arucoResult.corners[i][j].x << ", " << arucoResult.corners[i][j].y << ") ";
                }
                std::cout << std::endl;
            }
            */
            /*
            std::cout << "Measurement : y " << std::endl;
            for (int i = 0; i < arucoResult.y.size(); i++){
                std::cout << arucoResult.y(i) << std::endl;
            }
            */
            /*
            std::cout << "Elements of y:" << std::endl;
            for (int i = 0; i < arucoResult.y.size()/8; ++i) {
                std::cout << "Marker " << arucoResult.ids[i] << " Corners: ";
                for (int j = 0; j < 4; ++j){
                    std::cout << "(" << arucoResult.y(8 * i + 2 * j) << ", " << arucoResult.y(8 * i + 2 * j + 1) << ") ";
                }
                std::cout << std::endl;
            }
            */
            // Start handling plot GUI events (blocking)
            plot.start();
        }
        
    }

    bufferedVideoReader.stop();
    if (doExport)
    {
        std::cout << "File Succesfully Exported" << std::endl;
        bufferedVideoWriter.stop();
    }
}
