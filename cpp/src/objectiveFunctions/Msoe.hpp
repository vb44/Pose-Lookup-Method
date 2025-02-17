#pragma once

#ifndef MSOE_HPP
#define MSOE_HPP

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "Eigen/Dense"
#include "tbb/parallel_for.h"

#include "ConfigParser.hpp"
#include "ObjectiveFunction.hpp"
#include "utils.hpp"
#include "Raycaster.hpp"

/**
 * @brief Handle the algorithm configuration parsing.
 * 
 */
class Msoe : public ObjectiveFunction
{
    public:
        Msoe(const ConfigParser &config);

        ~Msoe();

        std::vector<int> calculateEvidence(const Eigen::MatrixXd &hypotheses) override;

        void setPointCloud(std::vector<Eigen::Vector4d> &pointCloud) override;

    private:
        Eigen::MatrixXd pointCloud_; // nx4 matrix ready for homogeneous transforms
        Eigen::Matrix4d sensorToPlatform_;
        std::vector<int> evidences_;
        std::vector<double> platformToSensor_;
        std::string modelFilePath_;
        Raycaster* raycaster;
        double sigma_;
};

#endif // MSOE_HPP