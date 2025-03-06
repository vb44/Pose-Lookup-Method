cmake_minimum_required(VERSION 3.18)

project(GenerateLookupTable)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build by default in release mode.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package(Open3D REQUIRED)
find_package(yaml-cpp REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/../common)

# Define source files.
set(SOURCES
    src/generateLookupTable.cpp
    src/ConfigParserLookup.cpp
    ../common/utils.cpp
)

# Add an executable called generateLookup.
add_executable(generateLookup ${SOURCES})
target_link_libraries(generateLookup
    Open3D::Open3D
    yaml-cpp::yaml-cpp
)
