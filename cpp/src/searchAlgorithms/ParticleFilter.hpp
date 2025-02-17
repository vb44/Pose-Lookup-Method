#pragma once

#ifndef PARTICLE_FILTER_HPP
#define PARTICLE_FILTER_HPP

#include <functional>
#include <fstream>
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
 * @brief Particle filter pose search algorithm.
 *        The pose search algorithm is decribed in Algorithm 1 from
 *        https://doi.org/10.3390/s21196473.
 * 
 */
class ParticleFilter
{
    public:
        /**
         * @brief Construct the Particle Filter object.
         * 
         * @param config The algorithm configuration parameters.
         * @param objFunc The objective function used for ranking hypotheses.
         */
        ParticleFilter(const ConfigParser &config,
                        std::shared_ptr<ObjectiveFunction> objFunc);

        /**
         * @brief Destroy the Particle Filter object.
         * 
         */
        ~ParticleFilter() = default;

        /**
         * @brief Find the best pose hypothesis using the objective function.
         * 
         * @return std::vector<double> The pose hypothesis.
         *         (roll, pitch, yaw, x, y, z) (rad, m).
         */
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

        // Random number generation (noise in the particle filter).
        boost::variate_generator<boost::mt19937&, boost::normal_distribution<double> >* randRot_ = 0;
        boost::variate_generator<boost::mt19937&, boost::normal_distribution<double> >* randTrans_ = 0;

        // Objective function that is being maximised.
        std::shared_ptr<ObjectiveFunction> objFunc_;
        
        /**
         * @brief Generates uniformly sampled hypotheses for the first
         *        particle filter iteration. 
         * 
         */
        void generateHypotheses();
};

#endif // PARTICLE_FILTER_HPP