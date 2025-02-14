cmake_minimum_required(VERSION 3.10)

project(GeometryPoseEstimator)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build by default in release mode.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(CMAKE_PREFIX_PATH "/usr/local/lib/cmake/embree-3.13.1" ${CMAKE_PREFIX_PATH})

# Find the required packages.
find_package(Assimp REQUIRED)
find_package(CGAL REQUIRED)
find_package(Embree 3.13.1 REQUIRED)
find_package(nanoflann REQUIRED)
find_package(TBB REQUIRED) 
find_package(Threads REQUIRED)
find_package(yaml-cpp REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/src/objectiveFunctions
                    ${PROJECT_SOURCE_DIR}/src/pointCloud
                    ${PROJECT_SOURCE_DIR}/src/searchAlgorithms
                    ${PROJECT_SOURCE_DIR}/src/utils)

# Define source files.
set(SOURCES
    src/main.cpp
    src/utils/ConfigParser.cpp
    src/searchAlgorithms/ParticleFilter.cpp
    src/objectiveFunctions/Plum.cpp
    src/objectiveFunctions/Msoe.cpp
    src/objectiveFunctions/Raycaster.cpp
    src/pointCloud/PointCloud.cpp
    src/utils/utils.cpp
)

# Add the executable.
add_executable(pose_estimator ${SOURCES})

# Link required libraries.
target_link_libraries(pose_estimator
    nanoflann::nanoflann
    TBB::tbb
    yaml-cpp::yaml-cpp
    Threads::Threads
    CGAL
    embree
    ${ASSIMP_LIBRARIES}
)