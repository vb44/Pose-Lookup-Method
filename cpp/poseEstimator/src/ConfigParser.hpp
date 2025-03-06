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
         * @brief Get the pose estimation method to use.
         * 
         * @return const std::string Pose estimation method. 
         */
        const std::string getPoseEstMethod() const;

        /**
         * @brief Get the path to the point cloud folder.
         * 
         * @return const std::string Path to the point cloud folder. 
         */
        const std::string getPcPath() const;
        
        /**
         * @brief Get the maximum sensor range.
         * 
         * @return const double Maximum sensor range (m).
         */
        const double getMaxSensorRange() const;
        
        /**
         * @brief Get the minimum sensor range.
         * 
         * @return const double Minimum sensor range (m). 
         */
        const double getMinSensorRange() const;
        
        /**
         * @brief Get the point cloud subsampling radius.
         * 
         * @return const double Radial subsampling radius (m).  
         */
        const double getPcSubsampleRadius() const;
        
        /**
         * @brief Get the sensor's pose relative to the platform.
         *        Set to [0,0,0,0,0,0] to estimate geometry pose in the sensor
         *        frame.
         * 
         * @return const std::vector<double> 6-DOF homogeneous sensor pose.
         *         (roll, pitch, yaw, x, y, z) (rad, m) 
         */
        const std::vector<double> getPlatformToSensor() const;
        
        /**
         * @brief Get the point cloud region of interst, defined in the
         *        platform frame. 
         * 
         * @return const std::vector<double> Region of interest to keep.
         *         (xMin, xMax, yMin, yMax, zMin, zMax) (m). 
         */
        const std::vector<double> getPcRegionOfInterest() const;
         
        /**
         * @brief Get the lookup table file.
         * 
         * @return const std::string Lookup table file. 
         */
        const std::string getLookupTableFile() const;
        
        /**
         * @brief Get the MSoE sigma configuration.
         * 
         * @return const double Sigma (m). 
         */
        const double getSigma() const;
        
        /**
         * @brief Get the path to the STL model of the geometry used for
         *        raycasting in MSoE.
         * 
         * @return const std::string Path to the STL model of the geometry. 
         */
        const std::string getModelFilePath() const;
        
        /**
         * @brief Get the initial rotational sigma for the pose search
         *        particle filter.
         * 
         * @return const double Rotational sigma (m).
         */
        const double getSearchRotSigma() const;
        
        /**
         * @brief Get the initial translational sigma for the pose search
         *        particle filter.
         * 
         * @return const double Translation sigma (m).
         */
        const double getSearchTransSigma() const;

        /**
         * @brief Get the no. of search iterations for the pose search
         *        partilce filter.
         * 
         * @return const double No. of search iterations (#).
         */
        const double getSearchNoIterations() const;

        /**
         * @brief Get the no. of search iterations for the pose search
         *        partilce filter.
         * 
         * @return const double No. of search iterations (#).
         */
        const double getResampleSize() const;
        
        /**
         * @brief Get the search seed for the pose search
         *        partilce filter.
         * 
         * @return const double Search seed around which the pose hypothesis
         *         space is centred.
         *         (roll, pitch, yaw, x, y, z) (rad, m).
         */
        const std::vector<double> getSearchSeed() const;
        
        /**
         * @brief Get the minimum search deviation from the seed for the
         *        pose search partilce filter.
         * 
         * @return const double Minimum search bounds from the seed for which
         *         the pose hypothesis space is centred.
         *         (roll, pitch, yaw, x, y, z) (rad, m).
         */
        const std::vector<double> getSearchMinDev() const;
        
        /**
         * @brief Get the maximum search deviation from the seed for the
         *        pose search partilce filter.
         * 
         * @return const double Maximum search bounds from the seed for which
         *         the pose hypothesis space is centred.
         *         (roll, pitch, yaw, x, y, z) (rad, m).
         */
        const std::vector<double> getSearchMaxDev() const;

        /**
         * @brief Get the pose search step sizes for the particle filter.
         * 
         * @return const std::vector<double> Search step sizes.
         *         (roll, pitch, yaw, x, y, z) (rad, m).
         */
        const std::vector<double> getSearchStepSizes() const;

    private:
        // The expected number of commandline arguments.
        static constexpr int EXPECTED_ARGUMENT_COUNT = 2;

        // Pose estimation method: one of "plum" or "msoe".
        std::string poseEstMethod_;

        // Point cloud file.
        // Region of interest and subsampling for computational benefit.
        double sensorMaxRange_;
        double sesnorMinRange_;
        double pcSubsampleRadius_;
        std::string pointCloudPath_;
        std::vector<double> platformToSensor_;
        std::vector<double> pcRegionOfInterest_;

        // PLuM: Lookup table parameters.
        // TODO: Read these from the lookup details file.
        double lookupTableStepSize_;
        std::string lookupTableFile_;
        std::vector<double> lookupTableToModel_;
        std::vector<double> lookupTableMaxXyz_;

        // MSoE: Model file path and measurment uncertainty.
        double sigma_;
        std::string modelFilePath_;

        // Search heuristic configuration parameters.
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