#ifndef CONSTANT
#define CONSTANT

#include <filesystem>

namespace Constant {
namespace Project {
const std::filesystem::path ROOT_PATH = std::filesystem::current_path().parent_path();
const std::filesystem::path DATASET_PATH = ROOT_PATH / "dataset" / "sequences" / "00";
}  // namespace Project

namespace Vision {
const int MAX_FRAME = 1000;
const int MIN_NUM_FEAT = 2000;
}  // namespace Vision

namespace Kitti {
const std::filesystem::path CALIB_PATH = Project::DATASET_PATH / "calib.txt";
const std::filesystem::path GROUND_TRUTH_PATH = Project::DATASET_PATH / "00.txt";
const std::filesystem::path TIMESTAMP_PATH = Project::DATASET_PATH / "times.txt";
const std::filesystem::path IMAGE_PATH = Project::DATASET_PATH / "image_0";


}  // namespace Kitti

}  // namespace Constant

#endif  // CONSTANT
