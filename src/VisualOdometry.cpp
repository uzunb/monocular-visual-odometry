#include "VisualOdometry.hpp"

VisualOdometry::VisualOdometry(/* args */) {
    this->cap = new cv::VideoCapture("/dev/video0");
    if (!this->cap->isOpened()) {
        std::cout << "Error opening video stream or file" << std::endl;
    }
}

VisualOdometry::~VisualOdometry() {}

void VisualOdometry::featureTracking(cv::Mat img_1, cv::Mat img_2,
                                     std::vector<cv::Point2f>& points1,
                                     std::vector<cv::Point2f>& points2,
                                     std::vector<uchar>& status) {
    // this function automatically gets rid of points for which tracking fails

    std::vector<float> err;
    cv::Size winSize = cv::Size(21, 21);
    cv::TermCriteria termcrit =
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01);

    if (img_1.type() != img_2.type()) {
        cv::cvtColor(img_2, img_2,
                     img_1.type() == CV_8UC1 ? cv::COLOR_BGR2GRAY : cv::COLOR_GRAY2BGR);
    }
    cv::calcOpticalFlowPyrLK(img_1, img_2, points1, points2, status, err, winSize, 3, termcrit, 0,
                             0.001);

    // getting rid of points for which the KLT tracking failed or those who have gone outside the
    // frame
    int indexCorrection = 0;
    for (int i = 0; i < status.size(); i++) {
        cv::Point2f pt = points2.at(i - indexCorrection);
        if ((status.at(i) == 0) || (pt.x < 0) || (pt.y < 0)) {
            if ((pt.x < 0) || (pt.y < 0)) {
                status.at(i) = 0;
            }
            points1.erase(points1.begin() + (i - indexCorrection));
            points2.erase(points2.begin() + (i - indexCorrection));
            indexCorrection++;
        }
    }
}

void VisualOdometry::featureDetection(
    cv::Mat img_1,
    std::vector<cv::Point2f>& points1) {  // uses FAST as of now, modify parameters as necessary
    std::vector<cv::KeyPoint> keypoints_1;
    int fast_threshold = 20;
    bool nonmaxSuppression = true;
    cv::FAST(img_1, keypoints_1, fast_threshold, nonmaxSuppression);
    cv::KeyPoint::convert(keypoints_1, points1, std::vector<int>());
}

double VisualOdometry::getAbsoluteScale(int frame_id, int sequence_id, double z_cal) {
    double x = 0, y = 0, z = 0;
    double x_prev, y_prev, z_prev;

    double scale = sqrt((x - x_prev) * (x - x_prev) + (y - y_prev) * (y - y_prev) +
                        (z - z_prev) * (z - z_prev));
    return scale;
}

void VisualOdometry::readGroundTruthPoses() {
    std::ifstream file(Constant::Kitti::GROUND_TRUTH_PATH);
    if (!file.is_open()) {
        std::cout << "Unable to open file" << std::endl;
        return;
    }

    std::string line;
    while (getline(file, line)) {
        std::cout << line << std::endl;
    }
}

void VisualOdometry::run() {
    this->warmup();

    std::string filename;
    std::filesystem::path imagePath = Constant::Kitti::IMAGE_PATH;
    double scale = 1.00;

    float fps = 0.0;
    float averageFps = 0.0;
    auto begin = std::chrono::high_resolution_clock::now();

    int numFrame = 0;
    char text[100];
    int fontFace = cv::FONT_HERSHEY_PLAIN;
    double fontScale = 1;
    int thickness = 1;
    cv::Point textOrg(10, 50);

    while (true) {
        this->cap->read(this->currentImage);
        if (!this->currentImage.data) {
            std::cout << " --(!) Error reading images " << std::endl;
            return;
        }

        // cvt to grayscale
        cv::cvtColor(this->currentImage, this->currentImage, cv::COLOR_RGB2GRAY);

        std::vector<uchar> status;
        featureTracking(this->prevImage, this->currentImage, this->prevFeatures, this->currFeatures,
                        status);

        this->E = cv::findEssentialMat(this->currFeatures, this->prevFeatures, this->cameraMatrix,
                                       cv::RANSAC, 0.999, 1.0, this->mask);
        cv::recoverPose(E, this->currFeatures, this->prevFeatures, this->cameraMatrix, this->R,
                        this->t, this->mask);

        cv::Mat prevPts(2, this->prevFeatures.size(), CV_64F),
            currPts(2, this->currFeatures.size(), CV_64F);

        for (int i = 0; i < this->prevFeatures.size(); i++) {
            // this (x,y) combination makes sense as observed from the source code of
            // triangulatePoints on GitHub
            prevPts.at<double>(0, i) = this->prevFeatures.at(i).x;
            prevPts.at<double>(1, i) = this->prevFeatures.at(i).y;

            currPts.at<double>(0, i) = this->currFeatures.at(i).x;
            currPts.at<double>(1, i) = this->currFeatures.at(i).y;
        }

        scale = getAbsoluteScale(numFrame, 0, this->t.at<double>(2));

        // std::cout << "Scale is " << scale << std::endl;

        if ((this->t.at<double>(2) > this->t.at<double>(0)) &&
            (this->t.at<double>(2) > this->t.at<double>(1))) {
            this->finalTranslationVector =
                this->finalTranslationVector + (this->finalRotationVector * this->t);
            this->finalRotationVector = this->R * this->finalRotationVector;

        } else {
            std::cout << "scale below 0.1, or incorrect translation" << std::endl;
        }

        // lines for printing results
        // myfile << this->finalTranslationVector.at<double>(0) << " " <<
        // this->finalTranslationVector.at<double>(1) << " " <<
        // this->finalTranslationVector.at<double>(2) << std::endl;

        // a redetection is triggered in case the number of feautres being trakced go below a
        // particular threshold
        if (this->prevFeatures.size() < MIN_NUM_FEAT) {
            // std::cout << "Number of tracked features reduced to " << this->prevFeatures.size() <<
            // std::endl; std::cout << "trigerring redection" << std::endl;
            featureDetection(this->prevImage, this->prevFeatures);
            featureTracking(this->prevImage, this->currentImage, this->prevFeatures,
                            this->currFeatures, status);
        }

        this->prevImage = this->currentImage.clone();
        this->prevFeatures = this->currFeatures;

        int x = int(this->finalTranslationVector.at<double>(0)) + 300;
        int y = int(this->finalTranslationVector.at<double>(2)) + 100;
        cv::circle(trajectoryMap, cv::Point(x, y), 1, CV_RGB(255, 0, 0), 2);

        char text[100];

        sprintf(text, "Coordinates: x = %02fm y = %02fm z = %02fm",
                this->finalTranslationVector.at<double>(0),
                this->finalTranslationVector.at<double>(1),
                this->finalTranslationVector.at<double>(2));

        // write the text with backgroud on the trajectory map
        cv::rectangle(trajectoryMap, cv::Point(10, 30), cv::Point(550, 50), CV_RGB(0, 0, 0), cv::FILLED);
        
        cv::putText(trajectoryMap, text, textOrg, fontFace, fontScale, cv::Scalar::all(255),
                    thickness, 8);

        imshow("Road facing camera", this->currentImage);
        imshow("Trajectory Map", trajectoryMap);

        if (cv::waitKey(1) == 27) {
            break;
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - begin;
        fps = 1 / elapsed.count();
        if (numFrame > 1) {
            averageFps = (averageFps * (numFrame - 2) + fps) / (numFrame - 1);
        } else {
            averageFps = fps;
        }

        begin = std::chrono::high_resolution_clock::now();

        std::cout << "Frame: " << numFrame << " FPS: " << fps << " Average FPS: " << averageFps
                  << std::endl;

        numFrame++;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - begin;
    fps = 1 / elapsed.count();
    averageFps = (averageFps * (MAX_FRAME - 2) + fps) / (MAX_FRAME - 1);
    std::cout << "Frame: " << MAX_FRAME << " FPS: " << fps << " Average FPS: " << averageFps
              << std::endl;

    // std::cout << this->finalRotationVector << std::endl;
    // std::cout << this->finalTranslationVector << std::endl;
}

void VisualOdometry::warmup() {
    cv::Mat img_1, img_2;

    std::ofstream myfile;
    myfile.open("results1_1.txt");

    char text[100];
    int fontFace = cv::FONT_HERSHEY_PLAIN;
    double fontScale = 1;
    int thickness = 1;
    cv::Point textOrg(10, 50);

    // read the first two frame from the camera
    cv::Mat img_1_c, img_2_c;
    this->cap->read(img_1_c);
    this->cap->read(img_2_c);
    if (!img_1_c.data || !img_2_c.data) {
        std::cout << " --(!) Error reading images " << std::endl;
        return;
    }

    // we work with grayscale images
    cvtColor(img_1_c, img_1, cv::COLOR_RGB2GRAY);
    cvtColor(img_2_c, img_2, cv::COLOR_RGB2GRAY);

    // feature detection, tracking
    // vectors to store the coordinates of the feature points
    std::vector<cv::Point2f> points1, points2;
    featureDetection(img_1, points1);  // detect features in img_1
    std::vector<uchar> status;
    featureTracking(img_1, img_2, points1, points2, status);  // track those features to img_2

    // recovering the pose and the essential matrix
    this->E = cv::findEssentialMat(points2, points1, this->cameraMatrix, cv::RANSAC, 0.999, 1.0,
                                   this->mask);
    cv::recoverPose(this->E, points2, points1, this->cameraMatrix, this->R, this->t, this->mask);

    this->prevImage = img_2;
    this->currentImage;
    this->prevFeatures = points2;
    this->currFeatures;

    this->finalRotationVector = this->R.clone();
    this->finalTranslationVector = this->t.clone();

    // namedWindow("Road facing camera", WINDOW_AUTOSIZE);
    // namedWindow("Trajectory Map", WINDOW_AUTOSIZE);

    this->trajectoryMap = cv::Mat::zeros(600, 600, CV_8UC3);
}