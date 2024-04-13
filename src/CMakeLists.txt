cmake_minimum_required(VERSION 3.18)

project(PLuM LANGUAGES C CXX)

add_compile_options(-std=c++17 -O3)

find_package(Open3D REQUIRED)
find_package(TBB REQUIRED) 

# Add an executable called plum.
add_executable(plum plum.cpp utils.cpp)
target_link_libraries(plum TBB::tbb)

# Add an executable called generateLookup.
add_executable(generateLookup)
target_sources(generateLookup PRIVATE generateLookupTable.cpp utils.cpp)
target_link_libraries(generateLookup PRIVATE Open3D::Open3D)

# Add an executable called raycastModel.
add_executable(raycastModel)
target_sources(raycastModel PRIVATE raycastModel.cpp utils.cpp)
target_link_libraries(raycastModel PRIVATE Open3D::Open3D)