#include <cstdlib>
#include <iostream>
#include <string>  
#include <filesystem>
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "imagefeatures.h"

int main(int argc, char *argv[])
{
    cv::String keys = 
        // Argument names | defaults | help message
        "{help h usage ?  |          | print this message}"
        "{@input          | <none>   | input can be a path to an image or video (e.g., ../data/lab.jpg)}"
        "{export e        |          | export output file to the ./out/ directory}"
        "{N               | 10       | maximum number of features to find}"
        "{detector d      | orb      | feature detector to use (e.g., harris, shi, aruco, orb)}"
        ;
    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("MCHA4400 Lab 2");

    if (parser.has("help"))
    {
        parser.printMessage();
        return EXIT_SUCCESS;
    }

    // Parse input arguments
    bool doExport = parser.has("export");
    int maxNumFeatures = parser.get<int>("N");
    cv::String detector = parser.get<std::string>("detector");
    std::filesystem::path inputPath = parser.get<std::string>("@input");

    // Check for syntax errors
    if (!parser.check())
    {
        parser.printMessage();
        parser.printErrors();
        return EXIT_FAILURE;
    }

    if (!std::filesystem::exists(inputPath))
    {
        std::cout << "File: " << inputPath.string() << " does not exist" << std::endl;
        return EXIT_FAILURE;
    }

    // Prepare output directory
    std::filesystem::path outputDirectory;
    if (doExport)
    {
        std::filesystem::path appPath = parser.getPathToApplication();
        outputDirectory = appPath / ".." / "out";

        // Create output directory if we need to
        if (!std::filesystem::exists(outputDirectory))
        {
            std::cout << "Creating directory " << outputDirectory.string() << std::endl;
            std::filesystem::create_directory(outputDirectory);
        }
        std::cout << "Output directory set to " << outputDirectory.string() << std::endl;
    }

    // Prepare output file path
    std::filesystem::path outputPath;
    if (doExport)
    {
        std::string outputFilename = inputPath.stem().string()
                                   + "_"
                                   + detector
                                   + inputPath.extension().string();
        outputPath = outputDirectory / outputFilename;
        std::cout << "Output name: " << outputPath.string() << std::endl;
    }

    // Check if input is an image or video (or neither)
    //bool isVideo = false; // TODO
    //bool isImage = false; // TODO

    ////
    // Assign if input file is neither
    //bool isVideo = false;
    //bool isImage = false;

    // Read the input file
    cv::VideoCapture cap(argv[1]);

    // Check if the input is a video
    bool isVideo = cap.get(cv::CAP_PROP_FRAME_COUNT) > 1;

    // Check if the input is an image
    //bool isImage = cap.get(cv::CAP_PROP_FRAME_COUNT) == 1;
    bool isImage = !isVideo;

    ////

    if (!isImage && !isVideo)
    {
        std::cout << "Could not read file: " << inputPath.string() << std::endl;
        return EXIT_FAILURE;
    }

    if (isImage)
    {
        // TODO: Call one of the detectAndDraw functions from imagefeatures.cpp according to the detector option specified at the command line
        cv::Mat image = cv::imread(inputPath.string(), cv::IMREAD_COLOR);
        cv::Mat imgout;

        // Get the width and height of the image
        int width = image.cols;
        int height = image.rows;



        if (detector == "harris") {
            std::cout << "Harris Feature Detector" << std::endl;
            std::cout << "Image Width: " << width << " pixels" << std::endl;
            std::cout << "Image Height: " << height << " pixels" << std::endl;
            std::cout << "Features Requested: " << maxNumFeatures << std::endl;
            imgout = detectAndDrawHarris(image, maxNumFeatures);
        } else if (detector == "shi") {
            std::cout << "Shi and Tomasi Feature Detector" << std::endl;
            std::cout << "Image Width: " << width << " pixels" << std::endl;
            std::cout << "Image Height: " << height << " pixels" << std::endl;
            std::cout << "Features Requested: " << maxNumFeatures << std::endl;
            imgout = detectAndDrawShiAndTomasi(image, maxNumFeatures);
        } else if (detector == "orb") {
            std::cout << "ORB Feature Detector" << std::endl;
            std::cout << "Image Width: " << width << " pixels" << std::endl;
            std::cout << "Image Height: " << height << " pixels" << std::endl;
            std::cout << "Features Requested: " << maxNumFeatures << std::endl;
            imgout = detectAndDrawORB(image, maxNumFeatures);
        } else if (detector == "aruco") {
            std::cout << "ArUco Feature Detector" << std::endl;
            std::cout << "Image Width: " << width << " pixels" << std::endl;
            std::cout << "Image Height: " << height << " pixels" << std::endl;
            imgout = detectAndDrawArUco(image, maxNumFeatures);
        } else {
            std::cout << "Invalid detector option. Available options: harris, shi, orb, aruco" << std::endl;
            return -1;
        }

        if (doExport)
        {
            // TODO: Write image returned from detectAndDraw to outputPath
            // Save the image to the specified outputPath
            bool success = cv::imwrite(outputPath.string(), imgout);
            if (!success) {
                std::cout << "Error writing image to: " << outputPath.string() << std::endl;
                return -1;
            }
            std::cout << "Image saved to: " << outputPath.string() << std::endl;
            }
        else
        {
            // TODO: Display image returned from detectAndDraw on screen and wait for keypress
            // Display the image on the screen and wait for a key press
            cv::imshow("Detected Features", imgout);
            cv::waitKey(0);
        }
    }

    if (isVideo)
    {
        cv::VideoCapture cap(inputPath.string());
        cv::VideoWriter outputVideo;

        // Check if the input video was opened successfully
        if (!cap.isOpened()) {
            std::cout << "Error opening input video: " << inputPath << std::endl;
            return -1;
        }

        int fps = static_cast<int>(cap.get(cv::CAP_PROP_FPS));
        int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

        if (doExport)
        {
            // TODO: Open output video for writing using the same fps as the input video
            //       and the codec set to cv::VideoWriter::fourcc('m', 'p', '4', 'v')            
            int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');

            // Open the output video for writing
            outputVideo.open(outputPath.string(), fourcc, fps, cv::Size(width, height));

            // Check if the output video was opened successfully
            if (!outputVideo.isOpened()) {
                std::cout << "Error opening output video: " << outputPath.string() << std::endl;
                return -1;
            }
        }

        cv::Mat inputframe;
        cv::Mat outputframe;
        while (true)
        {
            // TODO: Get next frame from input video
            cap >> inputframe;
            // TODO: If frame is empty, break out of the while loop
            if (!cap.read(inputframe))
            break;
            
            // TODO: Call one of the detectAndDraw functions from imagefeatures.cpp according to the detector option specified at the command line
            if (detector == "harris") {
                outputframe = detectAndDrawHarris(inputframe, maxNumFeatures);
            } else if (detector == "shi") {
                outputframe = detectAndDrawShiAndTomasi(inputframe, maxNumFeatures);
            } else if (detector == "orb") {
                outputframe = detectAndDrawORB(inputframe, maxNumFeatures);
            } else if (detector == "aruco") {
                outputframe = detectAndDrawArUco(inputframe, maxNumFeatures);
            } else {
                std::cout << "Invalid detector option. Available options: harris, shi, orb, aruco" << std::endl;
                return -1;
            }

            if (doExport)
            {
                // TODO: Write image returned from detectAndDraw to frame of output video
                outputVideo.write(outputframe);
            }
            else
            {
                // TODO: Display image returned from detectAndDraw on screen and wait for 1000/fps milliseconds
                cv::imshow("Processed Frame", outputframe);
                int delay = 1000 / fps;
                if (cv::waitKey(delay) == 27) // Press 'Esc' to stop processing and close the window
                    break;
            }
        }

        // TODO: release the input video object
        cap.release();

        if (doExport)
        {
            // TODO: release the output video object
            outputVideo.release();
        }
    }

    return EXIT_SUCCESS;
}



