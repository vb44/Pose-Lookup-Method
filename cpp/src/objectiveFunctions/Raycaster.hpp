#pragma once

#ifndef RAYCASTER_HPP
#define RAYCASTER_HPP

#include <cfloat>
#include <cmath>
#include <chrono>
#include <iostream>
#include <fstream>
#include <vector>

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <Eigen/Dense>
#include <embree3/rtcore.h>
#include <tbb/parallel_for.h>

class Raycaster
{
    public:
        /**
         * @brief Construct a new Raycaster object.
         * 
         * @param geometryPath Path to the geometry.
         */
        Raycaster(const std::string &geometryPath);

        /**
         * @brief Destroy the Raycaster object.
         * 
         */
        ~Raycaster();

        /**
         * @brief Compute the ray directions from the point cloud.
         *        These are used in the raycasting operation. 
         * 
         * @param pointCloud Point cloud used to extract the ray directions.
         */
        void computeRays(const std::vector<Eigen::Vector4d> &pointCloud);


        std::pair<std::vector<Eigen::Vector3f>, std::vector<int>> getRaycastResults();

        /**
         * @brief Print the raycast results. 
         * 
         */
        void printHits();

        /**
         * @brief Perform a raycast operation. 
         * 
         */
        void raycast();

        std::vector<double> getMeasuredRanges();

        /**
         * @brief Set the pose of the geometry.
         * 
         * @param geometryPose The geometry's pose as a homogeneous (4x4)
         *                     transform. 
         */
        void setGeometryPose(const Eigen::Matrix4f &geometryPose);

    private:

        // Ray directions.
        std::vector<Eigen::Vector3f> directions_;
        std::vector<double> measuredRanges_;
        
        // The geometry's pose.
        Eigen::Matrix4f geometryPose_;

        // The path to the geometry.
        std::string geometryPath_;

        // Embree raycasting containers.
        RTCDevice device_;
        RTCScene embreeScene_;
        RTCGeometry geom_;
        
        // The geometry ID.
        unsigned int geomID_;

        // The original geometry indices and vertices of the geometry,
        // loaded once and applied to different poses.
        std::vector<Eigen::Vector3f> originalVertices_;
        std::vector<unsigned int> originalIndices_;

        // Containers to store the raycasting results.
        std::vector<Eigen::Vector3f> hits_;
        std::vector<int> hitsValid_;

        // Transform the geometry at the current pose.
        void transformGeometry();
};


#endif // RAYCASTER_HPP