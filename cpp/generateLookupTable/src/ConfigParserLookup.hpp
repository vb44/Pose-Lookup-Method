#pragma once

#ifndef CONFIG_PARSER_LOOKUP_HPP
#define CONFIG_PARSER_LOOKUP_HPP

#include <ctime>
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>

#include <yaml-cpp/yaml.h>

/**
 * @brief Handle the algorithm configuration parsing.
 * 
 */
class ConfigParserLookup
{
    public:
        /**
         * @brief Construct a new ConfigParser object.
         * 
         * @param argc Number of commandline arguments.
         * @param argv Commandline arguments.
         */
        ConfigParserLookup(int argc, char** argv);

        /**
         * @brief Construct a new ConfigParser object.
         * 
         * @param yamlFilePath Path to the yaml file.
         */
        ConfigParserLookup(std::string yamlFilePath);

        /**
         * @brief Destroy the ConfigParser object.
         * 
         */
        ~ConfigParserLookup() = default;

        /**
         * @brief Parse the algorithm configuration path.
         * 
         * @return int Returns 0 if the algorithm configuration parsing was
         *             successful, 1 if there was an error.
         */
        int parseConfig();

        /**
         * @brief Get the lookup table step size.
         * 
         * @return const double Lookup table step size (m). 
         */
        const double getStepSize() const;

        /**
         * @brief Get the lookup table sigma.
         * 
         * @return const double Sigma (m). 
         */
        const double getSigma() const;

        /**
         * @brief Get the path to the STL model of the geometry used for
         *        generating the lookup table.
         * 
         * @return const std::string Path to the STL model of the geometry. 
         */
        const std::string getModelFilePath() const;
        
        /**
         * @brief Get the lookup table file name.
         * 
         * @return const std::string Lookup table file. 
         */
        const std::string getOutputFileName() const;
        
        /**
         * @brief Get the homogeneous transform from the lookup table to
         *        the model.
         * 
         * @return const std::vector<double> The 6-DOF homogeneous transform
         *         from the lookup table to the model.
         *         (roll, pitch, yaw, x, y, z) (rad, m).
         */
        const std::vector<double> getLookupTableToModel() const;

        /**
         * G@brief Get the maximum bounds of the lookup table.
         * 
         * @return const std::vector<double> Lookup table extents.
         *         (xMax, yMax, zMax) (m). 
         */
        const std::vector<double> getMaxBounds() const;  

        /**
         * @brief Write the configuration file for the lookup table.
         * 
         */
        void writeConfig();

    private:
        // The expected number of commandline arguments.
        static constexpr int EXPECTED_ARGUMENT_COUNT = 2;

        double stepSize_;
        double sigma_;
        std::string modelFilePath_;
        std::string outputFileName_;
        std::vector<double> lookupTableToModel_;
        std::vector<double> maxXyz_;

        std::string yamlFilePath_;
};

#endif // CONFIG_PARSER_LOOKUP_HPP