cmake_minimum_required(VERSION 3.10)

project(Plum)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build by default in release mode
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Find the required packages
find_package(nanoflann REQUIRED)
find_package(TBB REQUIRED) 
find_package(Threads REQUIRED)
find_package(yaml-cpp REQUIRED)

# Define source files
set(SOURCES
    src/main.cpp
    src/ConfigParser.cpp
    src/ParticleFilter.cpp
    src/Plum.cpp
    src/PointCloud.cpp
    src/utils.cpp
)

# Add the executable
add_executable(plum ${SOURCES})

# Link required libraries
target_link_libraries(plum
    nanoflann::nanoflann
    TBB::tbb
    yaml-cpp::yaml-cpp
    Threads::Threads
)