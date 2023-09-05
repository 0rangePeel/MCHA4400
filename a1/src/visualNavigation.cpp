#include <filesystem>
#include <string>
#include <iostream>
#include <opencv2/core/mat.hpp>
#include "BufferedVideo.h"
#include "visualNavigation.h"
#include "Camera.h"
//#include "Plot.h"

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

    while (true)
    {
        // Get next input frame
        cv::Mat imgin = bufferedVideoReader.read();
        if (imgin.empty())
        {
            break;
        }

        // Process frame

        // Update state

        // Update plot
        // Get a copy of the image for plot to draw on
        //state.view() = chessboardImage.image.clone();

        // Update plot
        //plot.render();

        // Write output frame 
        if (doExport)
        {
            cv::Mat imgout /* = plot.getFrame()*/;
            bufferedVideoWriter.write(imgout);
        }
    }

    bufferedVideoReader.stop();
    if (doExport)
    {
         bufferedVideoWriter.stop();
    }
}
