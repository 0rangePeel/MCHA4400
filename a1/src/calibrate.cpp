#include <string>
#include <sstream>
#include <iomanip>
#include <cstdlib>
#include <iostream>
#include <filesystem>
#include "calibrate.h"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include "Camera.h"

void calibrateCamera(const std::filesystem::path & configPath)
{
    
    std::cout << std::endl << "calibrate.cpp: It's calibrate time" << std::endl<< std::endl;

    // Open the video file
    cv::VideoCapture video("../data/calibration.mov");

    // Check if the video opened successfully
    if (!video.isOpened()) {
        std::cout << std::endl << "Error: Could not open video file." << std::endl << std::endl;
    }
    else{
        std::cout << std::endl << "Video hase been loaded" << std::endl << std::endl ;
    }

    // Break the video into frames and save into calibrationImages file
    // Loop through the video frames and save them as JPG images
    
    // This controls how many frames are skipped before one is chosen
    // i.e 100 results in 14 total frames for calibration
    int frameSkip = 100;

    int frameCount = 0;
    cv::Mat frame;

    while (true) {
        video >> frame;

        if (frame.empty()) {
            break;
        }

        frameCount++;

        if (frameCount % frameSkip  != 0) {
            continue; // Skip frames that are not multiples of 100
        }

        // Generate the file name with a zero-padded index
        std::stringstream filenameStream;
        filenameStream << "../data" << "/frame" << std::setw(4) << std::setfill('0') << frameCount << ".jpg";
        std::cout << "Creating frame" << std::setw(4) << std::setfill('0') << frameCount << ".jpg" << std::endl;
        // Save the frame as a JPG image
        imwrite(filenameStream.str(), frame);

    }

    // Terminate video to save memory
    video.release();
    std::cout << std::endl << "All Calibration Frames Created" << std::endl << std::endl;
    

    // - Read XML at configPath
    // Read chessboard data using configuration file
    // - Parse XML and extract relevant frames from source video containing the chessboard
    ChessboardData chessboardData(configPath);

    // - Perform camera calibration
    // Calibrate camera from chessboard data
    Camera cam;
    cam.calibrate(chessboardData);

    // - Write the camera matrix and lens distortion parameters to camera.xml file in same directory as configPath
    // Write camera calibration to file
    std::filesystem::path cameraPath = configPath.parent_path() / "camera.xml";
    cv::FileStorage fs(cameraPath.string(), cv::FileStorage::WRITE);
    fs << "camera" << cam;
    fs.release();

    std::cout << std::endl << std::endl;

    // - Visualise the camera calibration results
    // Pop-up window which either draws a box or corners
    int verbose = 1; // Enable Pop-up
    int hasBox = 1; // Box = 1, Corners = 0;
    if (verbose == 1)
    {
        if (hasBox == 1)
        {
            chessboardData.drawBoxes(cam);
        }
        else
        {
            chessboardData.drawCorners();
        }

        for (const auto & chessboardImage : chessboardData.chessboardImages)
        {
            cv::imshow("Calibration images (press ESC, q or Q to quit)", chessboardImage.image);
            char c = static_cast<char>(cv::waitKey(0));
            if (c == 27 || c == 'q' || c == 'Q') // ESC, q or Q to quit, any other key to continue
                break;
        }
    }

    // - Delete Calibration Images
    // After processing, delete the JPG files
    for (const auto& entry : std::filesystem::directory_iterator("../data")) {
        if (entry.path().extension() == ".jpg") {
            std::filesystem::remove(entry.path());
            std::cout << "Deleted: " << entry.path() << std::endl;
        }
    }
    std::cout << std::endl << "Calibration Complete - All Calibration Frames Deleted" << std::endl << std::endl;
}