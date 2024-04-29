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

    // TODO: add a fucntion to load these values directly from KITTI's calib files
    //  WARNING: different sequences in the KITTI VO dataset have different intrinsic/extrinsic
    //  parameters
    const double FOCAL_LENGTH = 718.856;
    const double CX = 607.1928;
    const double CY = 185.2157;
    const cv::Point2d cameraPrincipalPoint = cv::Point2d(CX, CY);

   public:
    VisualOdometry();
    ~VisualOdometry();

    /**
     * @brief Feature tracking between two images using optical flow.
     * 
     * @param img_1 
     * @param img_2 
     * @param points1 
     * @param points2 
     * @param status 
     * 
     * @return void
     */
    void featureTracking(cv::Mat img_1, cv::Mat img_2, std::vector<cv::Point2f>& points1,
                         std::vector<cv::Point2f>& points2, std::vector<uchar>& status);

    /**
     * @brief Feature detection in an image.
     * 
     * @param img_1 
     * @param points1 
     * 
     * @return void
     */
    void featureDetection(cv::Mat img_1, std::vector<cv::Point2f>& points1);

    /**
     * @brief Get the Absolute Scale object
     * 
     * @param frame_id 
     * @param sequence_id 
     * @param z_cal 
     * 
     * @return double 
     */
    double getAbsoluteScale(int frame_id, int sequence_id, double z_cal);

    /**
     * @brief Read the ground truth poses from the KITTI dataset.
     * 
     * @return void
     */
    void readGroundTruthPoses();

    /**
     * @brief Run the visual odometry pipeline.
     * 
     * @return void
     */
    void run();

    /**
     * @brief Warm up the visual odometry pipeline.
     * 
     * @return void
     */
    void warmup();
};

#endif  // SRC_VISUALODOMETRY
