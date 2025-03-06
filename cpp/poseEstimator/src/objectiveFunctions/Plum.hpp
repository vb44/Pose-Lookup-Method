#pragma once

#ifndef PLUM_HPP
#define PLUM_HPP

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "Eigen/Dense"
#include "tbb/parallel_for.h"

#include "ConfigParser.hpp"
#include "ConfigParserLookup.hpp"
#include "ObjectiveFunction.hpp"
#include "utils.hpp"

/**
 * @brief The PLuM objective function, described here,
 *        https://doi.org/10.3390/s23063085.
 * 
 */
class Plum : public ObjectiveFunction
{
    public:
        /**
         * @brief Construct a new Plum object.
         * 
         * @param config The algorithm configuration parameters.
         */
        Plum(const ConfigParser &config);

        /**
         * @brief Destroy the Plum object.
         * 
         */
        ~Plum();

        /**
         * @brief Calculate the evidence for the hypotheses using the PLuM
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
        /**
         * @brief Handles the lookup table configuration and loading.
         * 
         */
        struct LookupTable
        {
            double stepSize = 0.0;
            double pointsPerMeter;
            std::string lookupTablePath;
            Eigen::Matrix4d lookupTableToModel;
            std::vector<double> maxXyz;
            std::vector<unsigned int> numXyz;
            uint8_t* lookupTable;
        
            void readLookupTable()
            {
                pointsPerMeter = 1.0/stepSize;
                numXyz[0] = round(maxXyz[0]* pointsPerMeter + 1);
                numXyz[1] = round(maxXyz[1]* pointsPerMeter + 1);
                numXyz[2] = round(maxXyz[2]* pointsPerMeter + 1);
                lookupTable = (uint8_t*) malloc(numXyz[0]*
                                                numXyz[1]*
                                                numXyz[2]*sizeof(uint8_t));
                if (!lookupTable)
                {
                    throw std::bad_alloc();
                }

                std::ifstream lookupTableFileStream(lookupTablePath,
                                        std::ios::binary);
                lookupTableFileStream.read(
                    reinterpret_cast<char*>(&lookupTable[0]), 
                                            numXyz[0]*numXyz[1]*numXyz[2]);
                lookupTableFileStream.close();
            }
        };

        // The lookup table used to compute per-point reward.
        LookupTable lookupTable_;

        // A (nx4) matrix ready for homogeneous transformation.
        Eigen::MatrixXd pointCloud_; 

        // The evidences for all hypotheses.
        std::vector<int> evidences_;
};

#endif // PLUM_HPP