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
#include "MeasurementPoseBundle.h"

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
    assert(nFrames > 0);
    double fps = cap.get(cv::CAP_PROP_FPS);
    int codec = cap.get(cv::CAP_PROP_FOURCC);

    BufferedVideoReader bufferedVideoReader(5);
    bufferedVideoReader.start(cap);

    cv::VideoWriter videoOut;
    BufferedVideoWriter bufferedVideoWriter(3);
    if (doExport)
    {
        cv::Size frameSize;
        frameSize.width     = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        frameSize.height    = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        double outputFps    = fps;
        videoOut.open(outputPath.string(), codec, outputFps, frameSize);
        bufferedVideoWriter.start(videoOut);
    }

    // Visual navigation

    // Initialisation

    // Initial state mean
    Eigen::VectorXd mu(12);
    mu.setZero();

    // Initial state square-root covariance
    Eigen::MatrixXd S(12,12);
    S.setIdentity();

    // Initialise state
    StateSLAMPoseLandmarks state(Gaussian(mu, S));
    //StateSLAMPointLandmarks state(Gaussian(mu, S));

    // Initialise plot
    Plot plot(state, cam);

    int i = 0;
    double dt = 1/fps;
    double time = 0;

    while (true)
    {
        std::cout << "Frame: " << i << std::endl;
        // Get next input frame
        cv::Mat imgin = bufferedVideoReader.read();
        if (imgin.empty())
        {
            break;
        }

        // -Update plot
        // Get a copy of the image for plot to draw on

        ArUcoResult arucoResult = detectAndDrawArUco(imgin, 0, cam);

        state.setIdsLandmarks(arucoResult.ids);

        time = time + dt;

        MeasurementPoseBundle MeasurementPoseBundle(time, arucoResult.y, cam);

        std::cout << "visualNavigation.cpp - BeforeProcess" << std::endl;

        MeasurementPoseBundle.process(state);
        std::cout << "visualNavigation.cpp - AfterProcess" << std::endl;

        cv::Mat outputframe = arucoResult.imgout;
        //cv::Mat outputframe = imgin;

        std::cout << "visualNavigation.cpp - outputFrame" << std::endl;

        state.view() = outputframe.clone();
        std::cout << "visualNavigation.cpp - state.view" << std::endl;
        //state.view() = imgin;
        
        // Set local copy of state for plot to use
        plot.setState(state);
        std::cout << "visualNavigation.cpp - plot" << std::endl;
       
        // -Update plot
        plot.render();
        std::cout << "visualNavigation.cpp - render" << std::endl;
        

        i += 1;

        // Write output frame 
        if (doExport)
        {
            cv::Mat imgout = plot.getFrame();
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

            for (int i = 0; i < arucoResult.y.size(); i++){
                std::cout << arucoResult.y(i) << std::endl;
            }
            */
            
            

            // Start handling plot GUI events (blocking)
            plot.start();
        }
        
    }

    bufferedVideoReader.stop();
    if (doExport)
    {
         bufferedVideoWriter.stop();
    }
}
