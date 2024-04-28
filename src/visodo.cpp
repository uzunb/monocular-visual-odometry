/*

The MIT License

Copyright (c) 2015 Avi Singh

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

*/

#include <chrono>
#include <ctime>
#include <fstream>
#include <iostream>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <string>
#include <vector>

#include "Constant.hpp"
#include "vo_features.h"

using namespace cv;
using namespace std;

#define MAX_FRAME 4000
#define MIN_NUM_FEAT 5000

// IMP: Change the file directories (4 places) according to where your dataset is saved before
// running!

double getAbsoluteScale(int frame_id, int sequence_id, double z_cal) {
    string line;
    int i = 0;
    ifstream myfile(Constant::Kitti::GROUND_TRUTH_PATH);
    double x = 0, y = 0, z = 0;
    double x_prev, y_prev, z_prev;
    if (myfile.is_open()) {
        while ((getline(myfile, line)) && (i <= frame_id)) {
            z_prev = z;
            x_prev = x;
            y_prev = y;
            std::istringstream in(line);
            // cout << line << '\n';
            for (int j = 0; j < 12; j++) {
                in >> z;
                if (j == 7) y = z;
                if (j == 3) x = z;
            }

            i++;
        }
        myfile.close();
    }

    else {
        cout << "Unable to open file";
        return 0;
    }

    return sqrt((x - x_prev) * (x - x_prev) + (y - y_prev) * (y - y_prev) +
                (z - z_prev) * (z - z_prev));
}

void readGroundTruthPoses() {
    ifstream file(Constant::Kitti::GROUND_TRUTH_PATH);
    if (!file.is_open()) {
        cout << "Unable to open file" << endl;
        return;
    }

    string line;
    while (getline(file, line)) {
        cout << line << endl;
    }
}

int main(int argc, char** argv) {
    Mat img_1, img_2;
    Mat finalRotationVector, finalTranslationVector;

    ofstream myfile;
    myfile.open("results1_1.txt");

    double scale = 1.00;
    char filename1[4000];
    char filename2[4000];
    std::string imagePath = Constant::Kitti::IMAGE_PATH / "%06d.png";
    sprintf(filename1, imagePath.c_str(), 0);
    sprintf(filename2, imagePath.c_str(), 1);

    char text[100];
    int fontFace = FONT_HERSHEY_PLAIN;
    double fontScale = 1;
    int thickness = 1;
    cv::Point textOrg(10, 50);

    // read the first two frames from the dataset
    Mat img_1_c = imread(filename1);
    Mat img_2_c = imread(filename2);

    if (!img_1_c.data || !img_2_c.data) {
        std::cout << " --(!) Error reading images " << std::endl;
        return -1;
    }

    // we work with grayscale images
    cvtColor(img_1_c, img_1, COLOR_BGR2GRAY);
    cvtColor(img_2_c, img_2, COLOR_BGR2GRAY);

    // feature detection, tracking
    vector<Point2f> points1, points2;  // vectors to store the coordinates of the feature points
    featureDetection(img_1, points1);  // detect features in img_1
    vector<uchar> status;
    featureTracking(img_1, img_2, points1, points2, status);  // track those features to img_2

    // TODO: add a fucntion to load these values directly from KITTI's calib files
    //  WARNING: different sequences in the KITTI VO dataset have different intrinsic/extrinsic
    //  parameters
    double cameraFocalLength = 718.8560;
    cv::Point2d cameraPrincipalPoint(607.1928, 185.2157);

    // recovering the pose and the essential matrix
    Mat E, R, t, mask;
    E = findEssentialMat(points2, points1, cameraFocalLength, cameraPrincipalPoint, RANSAC, 0.999, 1.0, mask);
    recoverPose(E, points2, points1, R, t, cameraFocalLength, cameraPrincipalPoint, mask);

    Mat prevImage = img_2;
    Mat currentImage;
    vector<Point2f> prevFeatures = points2;
    vector<Point2f> currFeatures;

    char filename[100];

    finalRotationVector = R.clone();
    finalTranslationVector = t.clone();

    // namedWindow("Road facing camera", WINDOW_AUTOSIZE);
    // namedWindow("Trajectory Map", WINDOW_AUTOSIZE);

    Mat trajectoryMap = Mat::zeros(600, 600, CV_8UC3);

    float fps = 0.0;
    float averageFps = 0.0;
    auto begin = std::chrono::high_resolution_clock::now();

    for (int numFrame = 2; numFrame < MAX_FRAME; numFrame++) {

        sprintf(filename, imagePath.c_str(), numFrame);
        
        currentImage = imread(filename, IMREAD_GRAYSCALE);
        vector<uchar> status;
        featureTracking(prevImage, currentImage, prevFeatures, currFeatures, status);

        E = findEssentialMat(currFeatures, prevFeatures, cameraFocalLength, cameraPrincipalPoint, RANSAC, 0.999, 1.0, mask);
        recoverPose(E, currFeatures, prevFeatures, R, t, cameraFocalLength, cameraPrincipalPoint, mask);

        Mat prevPts(2, prevFeatures.size(), CV_64F), currPts(2, currFeatures.size(), CV_64F);

        for (int i = 0; i < prevFeatures.size(); i++) {
            // this (x,y) combination makes sense as observed from the source code of
            // triangulatePoints on GitHub
            prevPts.at<double>(0, i) = prevFeatures.at(i).x;
            prevPts.at<double>(1, i) = prevFeatures.at(i).y;

            currPts.at<double>(0, i) = currFeatures.at(i).x;
            currPts.at<double>(1, i) = currFeatures.at(i).y;
        }

        scale = getAbsoluteScale(numFrame, 0, t.at<double>(2));

        // cout << "Scale is " << scale << endl;

        if ((scale > 0.1) && (t.at<double>(2) > t.at<double>(0)) &&
            (t.at<double>(2) > t.at<double>(1))) {
            finalTranslationVector = finalTranslationVector + scale * (finalRotationVector * t);
            finalRotationVector = R * finalRotationVector;

        } else {
            // cout << "scale below 0.1, or incorrect translation" << endl;
        }

        // lines for printing results
        // myfile << finalTranslationVector.at<double>(0) << " " <<
        // finalTranslationVector.at<double>(1) << " " << finalTranslationVector.at<double>(2) <<
        // endl;

        // a redetection is triggered in case the number of feautres being trakced go below a
        // particular threshold
        if (prevFeatures.size() < MIN_NUM_FEAT) {
            // cout << "Number of tracked features reduced to " << prevFeatures.size() << endl;
            // cout << "trigerring redection" << endl;
            featureDetection(prevImage, prevFeatures);
            featureTracking(prevImage, currentImage, prevFeatures, currFeatures, status);
        }

        prevImage = currentImage.clone();
        prevFeatures = currFeatures;

        // int x = int(finalTranslationVector.at<double>(0)) + 300;
        // int y = int(finalTranslationVector.at<double>(2)) + 100;
        // circle(trajectoryMap, Point(x, y), 1, CV_RGB(255, 0, 0), 2);

        // rectangle(trajectoryMap, Point(10, 30), Point(550, 50), Scalar::all(0), cv::FILLED);
        // sprintf(text, "Coordinates: x = %02fm y = %02fm z = %02fm",
        //         finalTranslationVector.at<double>(0), finalTranslationVector.at<double>(1),
        //         finalTranslationVector.at<double>(2));
        // putText(trajectoryMap, text, textOrg, fontFace, fontScale, Scalar::all(255), thickness, 8);

        // imshow("Road facing camera", currentImage);
        // imshow("Trajectory Map", trajectoryMap);

        if (waitKey(1) == 27) {
            break;
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - begin;
        fps = 1 / elapsed.count();
        averageFps = (averageFps * (numFrame - 2) + fps) / (numFrame - 1);

        begin = std::chrono::high_resolution_clock::now();

        std::cout << "Frame: " << numFrame << " FPS: " << fps << " Average FPS: " << averageFps
                  << std::endl;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - begin;
    fps = 1 / elapsed.count();
    averageFps = (averageFps * (MAX_FRAME - 2) + fps) / (MAX_FRAME - 1);
    std::cout << "Frame: " << MAX_FRAME << " FPS: " << fps << " Average FPS: " << averageFps
              << std::endl;

    // cout << finalRotationVector << endl;
    // cout << finalTranslationVector << endl;

    return 0;
}