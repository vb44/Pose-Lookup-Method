#pragma once

#ifndef POINTCLOUD_HPP
#define POINTCLOUD_HPP

#include <fstream>
#include <set>

#include <eigen3/Eigen/Dense>
#include <nanoflann.hpp>
#include <tbb/parallel_for.h>

#include "ConfigParser.hpp"
#include "nanoflannUtils.hpp"
#include "utils.hpp"

/**
 * @brief A generic class for performing operations on a pointcloud.
 * 
 */
class PointCloud
{
    public:
        /**
         * @brief Construct a new Point Cloud object.
         * 
         * @param config The algorithm configuration parameters.
         */
        PointCloud(const ConfigParser &config);
        
        /**
         * @brief Destroy the Point Cloud object.
         * 
         */
        ~PointCloud() = default;

        /**
         * @brief Read a new .bin point cloud file (kitti format).
         * 
         * @param fileName Name of the file to read.
         */
        void readScan(const std::string &fileName);

        /**
         * @brief Get the point cloud.
         * 
         * @return const std::vector<Eigen::Vector4d>& The point cloud.
         */
        std::vector<Eigen::Vector4d> &getPtCloud();

    private:
        static constexpr int NUM_COLUMNS_BIN = 4;

        // Point cloud subsample radius.
        double subsampleRadius_;
        
        // Maximum range of the points in the point cloud.
        double maxSensorRange_;

        // Minimum range of the points in the point cloud.
        double minSensorRange_;

        // A container used for radially subsampling the points.
        std::set<int> allPoints_;

        // Container to store the point cloud.
        std::vector<Eigen::Vector4d> ptCloud_;

        Eigen::Matrix4d platformToSensor_;
        
        std::vector<double> pcRegionOfInterest_;

        // Container for a nanoflann-friendly point cloud.
        NanoflannPointsContainer<double> pcForKdTree_;

        /**
         * @brief Convert the points to a nanoflann-friendly container.
         * 
         * @param pts The points to convert to a nanoflann-friendly container.
         */
        void convertToPointCloudKdTree(const std::vector<Eigen::Vector4d> &pts);

        /**
         * @brief Radially subsample the point cloud.
         * 
         * @param pts The points to subsample.
         * @param subsampleRadius The subsample radius in meters.
         */
        void subsample(std::vector<Eigen::Vector4d> &pts,
                        double subsampleRadius);

        /**
         * @brief Transform the point cloud from the sensor to platform frame.
         * 
         */
        void transformToPlatformFrame();
        
        /**
         * @brief Get the point cloud region of interest, defined in the
         *        platform frame.
         * 
         */
        void getRegionOfInterest();

        /**
         * @brief Print the point cloud to std::cout.
         * 
         */
        void printPtCloud();
};

#endif // POINTCLOUD_HPP