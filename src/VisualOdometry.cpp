#include "VisualOdometry.hpp"

VisualOdometry::VisualOdometry(/* args */) {
    cv::Mat img_1, img_2;
    cv::Mat finalRotationVector, finalTranslationVector;

    std::ofstream myfile;
    myfile.open("results1_1.txt");

    double scale = 1.00;
    char filename1[4000];
    char filename2[4000];
    std::string imagePath = Constant::Kitti::IMAGE_PATH / "%06d.png";
    sprintf(filename1, imagePath.c_str(), 0);
    sprintf(filename2, imagePath.c_str(), 1);

    char text[100];
    int fontFace = cv::FONT_HERSHEY_PLAIN;
    double fontScale = 1;
    int thickness = 1;
    cv::Point textOrg(10, 50);

    // read the first two frames from the dataset
    cv::Mat img_1_c = cv::imread(filename1);
    cv::Mat img_2_c = cv::imread(filename2);

    if (!img_1_c.data || !img_2_c.data) {
        std::cout << " --(!) Error reading images " << std::endl;
        return -1;
    }

    // we work with grayscale images
    cvtColor(img_1_c, img_1, cv::COLOR_BGR2GRAY);
    cvtColor(img_2_c, img_2, cv::COLOR_BGR2GRAY);

    // feature detection, tracking
    std::vector<cv::Point2f> points1,
        points2;                       // vectors to store the coordinates of the feature points
    featureDetection(img_1, points1);  // detect features in img_1
    std::vector<uchar> status;
    featureTracking(img_1, img_2, points1, points2, status);  // track those features to img_2

    // TODO: add a fucntion to load these values directly from KITTI's calib files
    //  WARNING: different sequences in the KITTI VO dataset have different intrinsic/extrinsic
    //  parameters
    double cameraFocalLength = 718.8560;
    cv::Point2d cameraPrincipalPoint(607.1928, 185.2157);

    // recovering the pose and the essential matrix
    cv::Mat E, R, t, mask;
    E = cv::findEssentialMat(points2, points1, cameraFocalLength, cameraPrincipalPoint, cv::RANSAC,
                             0.999, 1.0, mask);
    cv::recoverPose(E, points2, points1, R, t, cameraFocalLength, cameraPrincipalPoint, mask);

    cv::Mat prevImage = img_2;
    cv::Mat currentImage;
    std::vector<cv::Point2f> prevFeatures = points2;
    std::vector<cv::Point2f> currFeatures;

    char filename[100];

    finalRotationVector = R.clone();
    finalTranslationVector = t.clone();

    // namedWindow("Road facing camera", WINDOW_AUTOSIZE);
    // namedWindow("Trajectory Map", WINDOW_AUTOSIZE);

    cv::Mat trajectoryMap = cv::Mat::zeros(600, 600, CV_8UC3);
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
    std::string line;
    int i = 0;
    std::ifstream myfile(Constant::Kitti::GROUND_TRUTH_PATH);
    double x = 0, y = 0, z = 0;
    double x_prev, y_prev, z_prev;
    if (myfile.is_open()) {
        while ((getline(myfile, line)) && (i <= frame_id)) {
            z_prev = z;
            x_prev = x;
            y_prev = y;
            std::istringstream in(line);
            // std::cout << line << '\n';
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
        std::cout << "Unable to open file";
        return 0;
    }

    return sqrt((x - x_prev) * (x - x_prev) + (y - y_prev) * (y - y_prev) +
                (z - z_prev) * (z - z_prev));
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
    float fps = 0.0;
    float averageFps = 0.0;
    auto begin = std::chrono::high_resolution_clock::now();

    for (int numFrame = 2; numFrame < MAX_FRAME; numFrame++) {
        sprintf(filename, imagePath.c_str(), numFrame);

        currentImage = imread(filename, cv::IMREAD_GRAYSCALE);
        std::vector<uchar> status;
        featureTracking(prevImage, currentImage, prevFeatures, currFeatures, status);

        E = cv::findEssentialMat(currFeatures, prevFeatures, cameraFocalLength,
                                 cameraPrincipalPoint, cv::RANSAC, 0.999, 1.0, mask);
        cv::recoverPose(E, currFeatures, prevFeatures, R, t, cameraFocalLength,
                        cameraPrincipalPoint, mask);

        cv::Mat prevPts(2, prevFeatures.size(), CV_64F), currPts(2, currFeatures.size(), CV_64F);

        for (int i = 0; i < prevFeatures.size(); i++) {
            // this (x,y) combination makes sense as observed from the source code of
            // triangulatePoints on GitHub
            prevPts.at<double>(0, i) = prevFeatures.at(i).x;
            prevPts.at<double>(1, i) = prevFeatures.at(i).y;

            currPts.at<double>(0, i) = currFeatures.at(i).x;
            currPts.at<double>(1, i) = currFeatures.at(i).y;
        }

        scale = getAbsoluteScale(numFrame, 0, t.at<double>(2));

        // std::cout << "Scale is " << scale << std::endl;

        if ((scale > 0.1) && (t.at<double>(2) > t.at<double>(0)) &&
            (t.at<double>(2) > t.at<double>(1))) {
            finalTranslationVector = finalTranslationVector + scale * (finalRotationVector * t);
            finalRotationVector = R * finalRotationVector;

        } else {
            // std::cout << "scale below 0.1, or incorrect translation" << std::endl;
        }

        // lines for printing results
        // myfile << finalTranslationVector.at<double>(0) << " " <<
        // finalTranslationVector.at<double>(1) << " " << finalTranslationVector.at<double>(2) <<
        // std::endl;

        // a redetection is triggered in case the number of feautres being trakced go below a
        // particular threshold
        if (prevFeatures.size() < MIN_NUM_FEAT) {
            // std::cout << "Number of tracked features reduced to " << prevFeatures.size() <<
            // std::endl; std::cout << "trigerring redection" << std::endl;
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
        // putText(trajectoryMap, text, textOrg, fontFace, fontScale, Scalar::all(255), thickness,
        // 8);

        // imshow("Road facing camera", currentImage);
        // imshow("Trajectory Map", trajectoryMap);

        if (cv::waitKey(1) == 27) {
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

    // std::cout << finalRotationVector << std::endl;
    // std::cout << finalTranslationVector << std::endl;
}

void VisualOdometry::warmup() {
    
}