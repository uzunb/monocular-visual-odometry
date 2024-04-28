#ifndef SRC_VISUALODOMETRY
#define SRC_VISUALODOMETRY

#include <ctype.h>

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
    /* data */
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
