cmake_minimum_required(VERSION 3.18)

project(PLuM LANGUAGES C CXX)

add_compile_options(-std=c++17 -O3)

# add an executable called plum
add_executable(plum plum.cpp helper.cpp)

find_package(Open3D REQUIRED)

# add an executable called generateLookup
add_executable(generateLookup)
target_sources(generateLookup PRIVATE generateLookupTable.cpp helper.cpp)
target_link_libraries(generateLookup PRIVATE Open3D::Open3D)

# add an executable called raycastModel
add_executable(raycastModel)
target_sources(raycastModel PRIVATE raycastModel.cpp helper.cpp)
target_link_libraries(raycastModel PRIVATE Open3D::Open3D)