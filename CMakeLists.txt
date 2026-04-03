cmake_minimum_required(VERSION 3.10)

project(optimizer LANGUAGES CXX)

# Set C++ compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra")


#add executables
add_executable(optimizer
    implementation.cpp
    fobj.h
    multivariate.h
)
