cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 11)

project (clBP)

add_library(clBP
        lib/Layer.cpp
        lib/Net.cpp
        lib/Neuron.cpp)

target_include_directories(clBP PUBLIC include)

