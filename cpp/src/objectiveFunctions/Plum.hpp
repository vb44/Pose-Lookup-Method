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
#include "ObjectiveFunction.hpp"
#include "utils.hpp"

/**
 * @brief Handle the algorithm configuration parsing.
 * 
 */
class Plum : public ObjectiveFunction
{
    public:
        Plum(const ConfigParser &config);

        ~Plum();

        std::vector<int> calculateEvidence(const Eigen::MatrixXd &hypotheses) override;

        void setPointCloud(const std::vector<Eigen::Vector4d> &pointCloud) override;

    private:
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
                lookupTable = (uint8_t*) malloc(numXyz[0]*numXyz[1]*numXyz[2]*sizeof(uint8_t));
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

        LookupTable lookupTable_;
        Eigen::MatrixXd pointCloud_; // nx4 matrix ready for homogeneous transforms
        std::vector<int> evidences_;
};

#endif // PLUM_HPP