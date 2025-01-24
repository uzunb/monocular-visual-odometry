cmake_minimum_required(VERSION 2.8.12)
project( mono-vo )
find_package( OpenCV REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS} )

file(GLOB viso
    "src/*.h"
    "src/*.cpp"
)

add_compile_options(-std=c++17 -O3)

add_executable( mono-vo ${viso} )

target_link_libraries( mono-vo ${OpenCV_LIBS} )
