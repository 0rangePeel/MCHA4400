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
    //TODO - Make this function more robust
    // check for bugs and edge cases
    // add assertions
    // use config path to get to needed files
    std::cout << std::endl << "calibrate.cpp: It's calibrate time" << std::endl<< std::endl;

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
    int hasBox = 0; // Box = 1, Corners = 0;
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
}