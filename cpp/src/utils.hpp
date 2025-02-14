#pragma once

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "eigen3/Eigen/Dense"
#include "tbb/parallel_for.h"

#include "ConfigParser.hpp"


namespace utils {

    /**
     * @brief Construct a homogeneous (4x4) matrix.
     * 
     * @param roll Roll angle in radians.
     * @param pitch Pitch angle in radians.
     * @param yaw Yaw angle in radians.
     * @param x X position in metres.
     * @param y Y position in metres.
     * @param z Z position in metres.
     * @return Eigen::Matrix4d The homogeneous (4x4) matrix constructed. 
     */
    Eigen::Matrix4d homogeneous(double roll, double pitch, double yaw, 
                                double x, double y, double z);

    /**
     * @brief Convert from a homogeneous (4x4) matrix to a homogeneous (6x1) vector.
     * 
     * @param T Input homogeneous (4x4) matrix.
     * @return std::vector<double> The constructed homogeneous (6x1) vector.
     */
    std::vector<double> hom2rpyxyz(const Eigen::Matrix4d &T);

    /**
     * @brief Read strings and convert to numbers to correctly order the input
     *        files. No two files should have the same name.
     * 
     * @param a The first string to compare.
     * @param b The second string to compare.
     * @return true If file a's name is smaller than file b's name.
     * @return false If file b's name is smaller than file a's name.
     */
    bool compareStrings(const std::string &a, const std::string &b);

    /**
     * @brief Print the progress to the terminal as a percentage out of 100%.
     * 
     * @param percentage The percentage of total scans registered.
     */
    void printProgress(double percentage);

} // namespace utils