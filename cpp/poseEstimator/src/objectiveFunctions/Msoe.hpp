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
 * @brief The MSoE objective function, described here,
 *        https://doi.org/10.3390/s21196473.
 * 
 */
class Msoe : public ObjectiveFunction
{
    public:
        /**
         * @brief Construct a new Msoe object.
         * 
         * @param config The algorithm configuration parameters.
         */
        Msoe(const ConfigParser &config);

        /**
         * @brief Destroy the Msoe object.
         * 
         */
        ~Msoe();

        /**
         * @brief Calculate the evidence for the hypotheses using the MSoE
         *        objective function. 
         * 
         * @param hypotheses The pose hypotheses to evaluate.
         * @return std::vector<int> The pose hypotheses rewards.
         */
        std::vector<int> calculateEvidence(const Eigen::MatrixXd &hypotheses) override;

        /**
         * @brief Set the point cloud used for calculating the objective
         *        function.
         * 
         * @param pointCloud 
         */
        void setPointCloud(std::vector<Eigen::Vector4d> &pointCloud) override;

    private:
        
        // The MSoE objective function configuration parameter.
        double sigma_;
 
        // The evidences for all hypotheses.
        std::vector<int> evidences_;

        // The fixed pose estimate from the platform to the sensor frame. 
        std::vector<double> platformToSensor_;

        // The path to the STL model used for raycasting.
        std::string modelFilePath_;
        
        // A (nx4) matrix ready for homogeneous transformation.
        Eigen::MatrixXd pointCloud_;

        // The fixed pose estimate from the sensor to the platform frame. 
        Eigen::Matrix4d sensorToPlatform_;

        // The raycasting application.
        Raycaster* raycaster;
};

#endif // MSOE_HPP