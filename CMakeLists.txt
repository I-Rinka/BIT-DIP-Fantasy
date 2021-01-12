cmake_minimum_required(VERSION 2.8)
project( DipFantasy )

SET(CMAKE_BUILD_TYPE "Debug")#设置debug

find_package( OpenCV REQUIRED )#找到OpenCV的库

include_directories( ./lib )#自己写的头文件
aux_source_directory( ./lib My_LIBS )#实现头文件的各种源文件


include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( DipFantasy Main.cpp ${My_LIBS} )

target_link_libraries( DipFantasy ${OpenCV_LIBS} )#链接OpenCV的库
