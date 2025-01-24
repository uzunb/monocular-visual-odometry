#include <iostream>
#include <memory>

#include "Constant.hpp"
#include "VisualOdometry.hpp"

int main(int argc, char** argv) {
    std::unique_ptr<VisualOdometry> vo = std::make_unique<VisualOdometry>();
    vo->run();

    return 0;
}

#include <stdio.h>
