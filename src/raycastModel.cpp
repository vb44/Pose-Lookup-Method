/*
This file generates a point cloud given,
    - a geometry model,
    - the heading and elevation ranges of the rays from the sensor, and
    - the homogeneous transform from the sensor to the model.
TODO: 
    - Add the functionality to generate noisy sensor measurements.
    - Add error-checking.

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
    raycastParams params;
    int err = parseArgsRaycast(&params, argc, argv);
    if (err) exit(1);    
    
    // Setup the sensor rays.
    double radius = 1;
    unsigned int numPhi   = round(abs(params.phiRange[1]-params.phiRange[0])/params.phiRange[2]);
    unsigned int numTheta = round(abs(params.thetaRange[1]-params.thetaRange[0])/params.thetaRange[2]);
    Eigen::VectorXd phiPts   = Eigen::VectorXd::LinSpaced(numPhi,params.phiRange[0],params.phiRange[1]);
    Eigen::VectorXd thetaPts = Eigen::VectorXd::LinSpaced(numTheta,params.thetaRange[0],params.thetaRange[1]);

    // Read the geometry model mesh.
    auto mesh = std::make_shared<open3d::geometry::TriangleMesh>();
    open3d::io::ReadTriangleMesh(params.modelFileName,*mesh);
    auto tmesh = open3d::t::geometry::TriangleMesh::FromLegacy(*mesh,open3d::core::Float32, open3d::core::Int64);

    // Apply the sensor to model transform to the geometry.
    open3d::core::Tensor tf = open3d::core::eigen_converter::EigenMatrixToTensor(params.sensorToModel);
    auto tmeshTf = tmesh.Transform(tf);

    // Create a raycasting scene.
    open3d::t::geometry::RaycastingScene scene;
    scene.AddTriangles(tmeshTf);

    // Open the file to save the raycast results.
    std::ofstream outputResultsFile(params.outputFileName); 
    
    // Perform the ryacasting operation.
    for (unsigned int i = 0; i < numPhi; i++)
    {
        for (unsigned j = 0; j < numTheta; j++)
        {
            int numRays = 1;
            auto rays = open3d::core::Tensor::Zeros({numRays,6},open3d::core::Float32);

            // Calculate the ray's [x,y,z] point by using the spherical coordinates.
            // The ray always starts at the origin [0,0,0] (sensor frame) to begin with.
            // The ray needs to be normalised to have a length of 1, so use radius = 1.
            double rayEndX = radius*sin(thetaPts(j)*M_PI/180)*cos(phiPts(i)*M_PI/180);
            double rayEndY = radius*sin(thetaPts(j)*M_PI/180)*sin(phiPts(i)*M_PI/180);
            double rayEndZ = radius*cos(thetaPts(j)*M_PI/180);
            Eigen::Vector3f rayEnd = Eigen::Vector3f(rayEndX,rayEndY,rayEndZ).normalized(); 
            rays.SetItem({open3d::core::TensorKey::Index(0), open3d::core::TensorKey::Slice(3,6,1)},
                          open3d::core::Tensor::Init<float>({rayEnd(0),rayEnd(1),rayEnd(2)}));

            // Cast the ray.
            auto castResults = scene.CastRays(rays);
            auto &castDists = castResults["t_hit"];
            auto distTensor = open3d::core::Tensor::Zeros({1,1},open3d::core::Float32);
            distTensor.SetItem({open3d::core::TensorKey::Index(0)},castDists);

            // Calculate the range.
            auto dist = open3d::core::eigen_converter::TensorToEigenMatrixXd(distTensor)(0,0);

            // Save the range if it intersects the geometry, i.e., the range length is not infinite.
            if (!std::isinf(dist))
            {
                double intersectX = dist*sin(thetaPts(j)*M_PI/180)*cos(phiPts(i)*M_PI/180);
                double intersectY = dist*sin(thetaPts(j)*M_PI/180)*sin(phiPts(i)*M_PI/180);
                double intersectZ = dist*cos(thetaPts(j)*M_PI/180);

                // Write the result to file.
                outputResultsFile << intersectX << " " << intersectY << " " << intersectZ << std::endl;
            }
        }
    }

    // Close the file.
    outputResultsFile.close();

    // Write the configuration file.
    writeRaycastConfigFile(&params);

    return 0;
}