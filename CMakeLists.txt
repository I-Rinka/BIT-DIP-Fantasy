cmake_minimum_required(VERSION 2.8)
project( DisplayImage )

SET(CMAKE_BUILD_TYPE "Debug")#设置debug

find_package( OpenCV REQUIRED )#找到OpenCV的库

include_directories( ./lib )#自己的库
aux_source_directory( ./lib My_LIBS )#


include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( DisplayImage DisplayImage.cpp ${My_LIBS} )

target_link_libraries( DisplayImage ${OpenCV_LIBS} )#链接OpenCV的库
