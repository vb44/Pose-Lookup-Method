#pragma once

#ifndef PARTICLE_FILTER_HPP
#define PARTICLE_FILTER_HPP

#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <boost/random.hpp>
#include <boost/math/distributions/normal.hpp>
#include "Eigen/Dense"
#include "tbb/parallel_for.h"

#include "ConfigParser.hpp"
#include "ObjectiveFunction.hpp"
#include "utils.hpp"

/**
 * @brief Handle the algorithm configuration parsing.
 * 
 */
class ParticleFilter
{
    public:
        ParticleFilter(const ConfigParser &config, std::shared_ptr<ObjectiveFunction> objFunc);

        ~ParticleFilter() = default;

        std::vector<double> findBestGeometryPose();

    private:
        unsigned int numberOfHypotheses_;
        Eigen::MatrixXd hypotheses_;
        Eigen::MatrixXd hypothesesSampled_;

        // Search heuristic parameters
        double rotSigma_;
        double transSigma_;
        double noIterations_;
        double resampleSize_;
        std::vector<double> seed_;
        std::vector<double> minDev_;
        std::vector<double> maxDev_;
        std::vector<double> stepSizes_;

        // Random number generation (noise in the particle filter)
        boost::variate_generator<boost::mt19937&, boost::normal_distribution<double> >* randRot_ = 0;
        boost::variate_generator<boost::mt19937&, boost::normal_distribution<double> >* randTrans_ = 0;

        // Objective function that is being maximised.
        std::shared_ptr<ObjectiveFunction> objFunc_;

        void generateHypotheses();
};

#endif // PARTICLE_FILTER_HPP