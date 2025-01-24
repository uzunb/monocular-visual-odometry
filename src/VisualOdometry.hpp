#ifndef SRC_VISUALODOMETRY
#define SRC_VISUALODOMETRY

#include <ctype.h>

#define FMT_HEADER_ONLY
#include <fmt/format.h>

#include <algorithm>  // for copy
#include <chrono>
#include <ctime>
#include <fstream>
#include <iostream>
#include <iterator>  // for ostream_iterator
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <sstream>
#include <string>
#include <vector>

#include "Constant.hpp"

#define MAX_FRAME 4000
#define MIN_NUM_FEAT 5000

class VisualOdometry {
   private:
    cv::Mat K;
    cv::Mat prevImage, currentImage;
    std::vector<cv::Point2f> prevFeatures, currFeatures;
    cv::Mat finalRotationVector, finalTranslationVector;
    cv::Mat E, R, t, mask;
    cv::Mat trajectoryMap;
    cv::VideoCapture* cap;

    // fx="767.855773858" fy="767.855773858" cx="318.240564418" cy="142.074238737"
    // k1="0.0583525153274" k2="-1.42298815528" p1="-0.0455457805793" p2="0.00114958025084"
    // k3="8.83677813991"
    const double FOCAL_LENGTH_X = 767.855773858;
    const double FOCAL_LENGTH_Y = 767.855773858;
    const double CX = 318.240564418;
    const double CY = 142.074238737;
    const double K1 = 0.0583525153274;
    const double K2 = -1.42298815528;
    const double P1 = -0.0455457805793;
    const double P2 = 0.00114958025084;
    const double K3 = 8.83677813991;
    const cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << FOCAL_LENGTH_X, 0, CX, 0, FOCAL_LENGTH_Y, CY, 0, 0, 1);
    const cv::Point2d cameraPrincipalPoint = cv::Point2d(CX, CY);
    const cv::Mat distCoeffs = (cv::Mat_<double>(5, 1) << K1, K2, P1, P2, K3);

   public:
    VisualOdometry();
    ~VisualOdometry();

    void featureTracking(cv::Mat img_1, cv::Mat img_2, std::vector<cv::Point2f>& points1,
                         std::vector<cv::Point2f>& points2, std::vector<uchar>& status);
    void featureDetection(cv::Mat img_1, std::vector<cv::Point2f>& points1);

    double getAbsoluteScale(int frame_id, int sequence_id, double z_cal);

    void readGroundTruthPoses();

    void run();

    void warmup();
};

#endif  // SRC_VISUALODOMETRY
