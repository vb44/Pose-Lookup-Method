#pragma once

#ifndef CONFIG_PARSER_HPP
#define CONFIG_PARSER_HPP

#include <iostream>
#include <string>

#include <yaml-cpp/yaml.h>

/**
 * @brief Handle the algorithm configuration parsing.
 * 
 */
class ConfigParser
{
    public:
        /**
         * @brief Construct a new ConfigParser object.
         * 
         * @param argc Number of commandline arguments.
         * @param argv Commandline arguments.
         */
        ConfigParser(int argc, char** argv);

        /**
         * @brief Destroy the ConfigParser object.
         * 
         */
        ~ConfigParser() = default;

        /**
         * @brief Parse the algorithm configuration path.
         * 
         * @return int Returns 0 if the algorithm configuration parsing was
         *             successful, 1 if there was an error.
         */
        int parseConfig();

        /**
         * @brief Get the path to the point cloud folder.
         * 
         * @return const std::string Path to the point cloud folder. 
         */
        const std::string getPcPath() const;
        
        const double getMaxSensorRange() const;
        const double getMinSensorRange() const;
        const double getPcSubsampleRadius() const;
        const std::vector<double> getPlatformToSensor() const;
        const std::vector<double> getPcRegionOfInterest() const;
        
        const double getLookupTableStepSize() const;
        const std::string getLookupTableFile() const;
        const std::vector<double> getLookupTableToModel() const;
        const std::vector<double> getLookupTableMaxXyz() const;

        const double getSigma() const;
        const std::string getModelFilePath() const;
        
        const double getSearchRotSigma() const;
        const double getSearchTransSigma() const;
        const double getSearchNoIterations() const;
        const double getResampleSize() const;
        const std::vector<double> getSearchSeed() const;
        const std::vector<double> getSearchMinDev() const;
        const std::vector<double> getSearchMaxDev() const;
        const std::vector<double> getSearchStepSizes() const;

        const std::string getPoseEstMethod() const;

    private:
        // The expected number of commandline arguments.
        static constexpr int EXPECTED_ARGUMENT_COUNT = 2;

        // Pose estimation method
        std::string poseEstMethod_;

        // Point cloud file
        std::string pointCloudPath_;
        double sensorMaxRange_;
        double sesnorMinRange_;
        double pcSubsampleRadius_;
        std::vector<double> platformToSensor_;
        std::vector<double> pcRegionOfInterest_;

        // PLuM: Lookup table parameters
        // TODO: Read these from the lookup details file.
        double lookupTableStepSize_;
        std::string lookupTableFile_;
        std::vector<double> lookupTableToModel_;
        std::vector<double> lookupTableMaxXyz_;

        // MSoE: Model file path and measurment uncertainty
        double sigma_;
        std::string modelFilePath_;

        // Search heuristic performance
        double searchRotSigma_;
        double searchTransSigma_;
        double searchNoIterations_;
        double searchResampleSize_;
        std::vector<double> searchSeed_;
        std::vector<double> searchMinDev_;
        std::vector<double> searchMaxDev_;
        std::vector<double> searchStepSizes_;

        // Path to the algorithm configuration file.
        std::string yamlFilePath_;
};

#endif // CONFIG_PARSER_HPP