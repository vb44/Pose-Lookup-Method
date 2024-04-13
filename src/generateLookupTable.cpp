/*
This file generates a lookup table given,
    - a geometry model,
    - the homogeneous transform from the sensor to the model,
    - the maximum bounds of the lookup table,
    - the stepsize of the points in the lookup table, and
    - the standard deviation (sigma) of the reward funtion.
TODO: 
    - Add error-checking for argument parsing.
    - Only reward points where a ray can intersect the geometry.
        - I.e., not on the inside of a geometry.

Author: Vedant Bhandari
Last modified: 16/07/2023
*/

#include "utils.hpp"

// Use the open3d library to perform the raycasting operations.
#include "open3d/Open3D.h"
#include "open3d/t/geometry/RaycastingScene.h"
#include "open3d/core/EigenConverter.h"

int main(int argc, char* argv[])
{
    // Argument parsing.
    lookupTableParams params;
    int err =  parseArgsLookupTable(&params, argc, argv);
    if (err) exit(1);

    // Generate the query points in the lookup table.
    unsigned int numX = round(params.maxXYZ[0]/params.stepSize) + 1;
    unsigned int numY = round(params.maxXYZ[1]/params.stepSize) + 1;
    unsigned int numZ = round(params.maxXYZ[2]/params.stepSize) + 1;
    Eigen::VectorXd xPts = Eigen::VectorXd::LinSpaced(numX,0,params.maxXYZ[0]);
    Eigen::VectorXd yPts = Eigen::VectorXd::LinSpaced(numY,0,params.maxXYZ[1]);
    Eigen::VectorXd zPts = Eigen::VectorXd::LinSpaced(numZ,0,params.maxXYZ[2]);

    // Read the geometry model.
    auto mesh = std::make_shared<open3d::geometry::TriangleMesh>();
    open3d::io::ReadTriangleMesh(params.modelFileName,*mesh);
    auto tmesh = open3d::t::geometry::TriangleMesh::FromLegacy(*mesh,open3d::core::Float32, open3d::core::Int64);

    // Transform the model to the lookup frame.
    open3d::core::Tensor tf = open3d::core::eigen_converter::EigenMatrixToTensor(params.lookupToModel);
    auto tmeshTf = tmesh.Transform(tf);
    
    // Create the scene.
    open3d::t::geometry::RaycastingScene scene;
    scene.AddTriangles(tmeshTf);

    // Write the lookup table.
    std::ofstream lookupFile;
    lookupFile.open(params.outputFileName, std::ios::binary);
    uint8_t reward;
    double queryPt[3];
    for (unsigned int i = 0; i < numX; i++)
    {
        for (unsigned int j = 0; j < numY; j++)
        {
            for (unsigned int k = 0; k < numZ; k++)
            {
                // Create the query point.
                queryPt[0] = xPts(i);
                queryPt[1] = yPts(j);
                queryPt[2] = zPts(k);

                // Find the closest distance to the model.
                auto queryPtTensor = open3d::core::Tensor::Zeros({1,3},open3d::core::Float32);
                queryPtTensor.SetItem({open3d::core::TensorKey::Index(0)},
                                       open3d::core::Tensor::Init<double>({queryPt[0],queryPt[1],queryPt[2]}));
                auto closestDistanceTensor = open3d::core::Tensor::Zeros({1,1},open3d::core::Float32);
                closestDistanceTensor.SetItem({open3d::core::TensorKey::Index(0)},scene.ComputeDistance(queryPtTensor));
                auto closestDistance = open3d::core::eigen_converter::TensorToEigenMatrixXd(closestDistanceTensor);

                // Calculate the reward (saved as a 8-bit number from 0-225).
                reward = 255 * exp(-0.5 * closestDistance(0,0) * closestDistance(0,0) / (params.sigma * params.sigma));
                
                // Write to lookup table.
                lookupFile << reward;
            }
        }
    }

    // Close the file.
    lookupFile.close();

    // Write the configuration file.
    writeLookupConfigFile(&params);

    return 0;
}