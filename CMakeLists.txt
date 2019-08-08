cmake_minimum_required(VERSION 3.8.1)
project(robonet)


set(CMAKE_CXX_STANDARD 14)

add_subdirectory(clBP)

option(BUILD_TRACKING "Build the robot tracking executable" OFF)

find_package(OpenCV REQUIRED)

find_package(Boost REQUIRED)

add_executable(robonet main.cpp serialib.cpp neural.cpp neural.h bandpass.cpp)
target_link_libraries(robonet clBP ${OpenCV_LIBS})

if(BUILD_TRACKING)
  add_executable(camtrack tracking/Source.cpp)
  target_link_libraries(camtrack ${OpenCV_LIBS})
endif()
