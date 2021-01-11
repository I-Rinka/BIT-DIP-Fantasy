cmake_minimum_required(VERSION 2.8)
project( DisplayImage )

# debug setting
SET(CMAKE_BUILD_TYPE "Debug")

find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( DisplayImage DisplayImage.cpp )
target_link_libraries( DisplayImage ${OpenCV_LIBS} )

