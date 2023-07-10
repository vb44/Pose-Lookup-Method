#include "helper.h"
#include <typeinfo>
#include <memory>
#include "open3d/Open3D.h"
#include "open3d/t/geometry/RaycastingScene.h"
#include "open3d/core/EigenConverter.h"
// #include <iostream>

int main(int argc, char* argv[])
{
    // USER CONFIGURATION -----------------------------------------------------
    std::string fileName = "../sample_data/StanfordBunny.stl";
    std::string outputFileName = "../outputs/bunny_pointcloud";

    // location of the model relative to the sensor
    Eigen::MatrixXd sensorToModel = homogeneous(0,0,30*M_PI/180,30,20,-5); 
    
    // LiDAR is at the origin - all measurements and results are in the LiDAR frame
    double origin[3] = {0,0,0}; 

    // Point cloud raycast setup
    double phiRange[3] = {-90,90,9};    // heading   [min,max,increment] (degrees)
    double thetaRange[3] = {0,90,4.5};  // elevation [min,max,increment] (degrees)
    
    // SETUP THE RAYS ---------------------------------------------------------
    double radius = 1;
    unsigned int numPhi = round(abs(phiRange[1]-phiRange[0])/phiRange[2]);
    unsigned int numTheta = round(abs(thetaRange[1]-thetaRange[0])/thetaRange[2]);
    Eigen::VectorXd phiPts = Eigen::VectorXd::LinSpaced(numPhi,phiRange[0],phiRange[1]);
    Eigen::VectorXd thetaPts = Eigen::VectorXd::LinSpaced(numTheta,thetaRange[0],thetaRange[1]);

    // READ THE MESH ----------------------------------------------------------
    auto mesh = std::make_shared<open3d::geometry::TriangleMesh>();
    open3d::io::ReadTriangleMesh(fileName,*mesh);
    auto tmesh = open3d::t::geometry::TriangleMesh::FromLegacy(*mesh,open3d::core::Float32, open3d::core::Int64);

    // APPLY THE SENSOR TO MODEL TRANSFORM ------------------------------------ 
    open3d::core::Tensor tf = open3d::core::eigen_converter::EigenMatrixToTensor(sensorToModel);
    auto tmeshTf = tmesh.Transform(tf);

    // CREATE A RAYCASTING SCENE ----------------------------------------------
    open3d::t::geometry::RaycastingScene scene;
    scene.AddTriangles(tmeshTf);

    // PERFORM THE RAYCASTING OPERATION ---------------------------------------
    // open the file to save the raycast results
    std::ofstream outputResultsFile(outputFileName);

    // calculate the range for each ray
    for (unsigned int i = 0; i < numPhi; i++)
    {
        for (unsigned j = 0; j < numTheta; j++)
        {
            int numRays = 1;
            auto rays = open3d::core::Tensor::Zeros({numRays,6},open3d::core::Float32);

            // calculate the ray's [x,y,z] point by using the spherical coordinates
            // the ray always starts at the origin [0,0,0] (sensor frame) to begin with
            // needs to be normalised to have a length of 1, so use radius = 1
            double rayEndX = radius*sin(thetaPts(j)*M_PI/180)*cos(phiPts(i)*M_PI/180);
            double rayEndY = radius*sin(thetaPts(j)*M_PI/180)*sin(phiPts(i)*M_PI/180);
            double rayEndZ = radius*cos(thetaPts(j)*M_PI/180);
            Eigen::Vector3f rayEnd = Eigen::Vector3f(rayEndX,rayEndY,rayEndZ).normalized(); 
            rays.SetItem({open3d::core::TensorKey::Index(0), open3d::core::TensorKey::Slice(3,6,1)},
                          open3d::core::Tensor::Init<float>({rayEnd(0),rayEnd(1),rayEnd(2)}));

            // cast the ray
            auto castResults = scene.CastRays(rays);
            auto &castDists = castResults["t_hit"];
            auto distTensor = open3d::core::Tensor::Zeros({1,1},open3d::core::Float32);
            distTensor.SetItem({open3d::core::TensorKey::Index(0)},castDists);

            // calculate the range
            auto dist = open3d::core::eigen_converter::TensorToEigenMatrixXd(distTensor)(0,0);

            // save the range if it intersects the geometry, i.e., the range length is not infinite
            if (!std::isinf(dist))
            {
                double intersectX = dist*sin(thetaPts(j)*M_PI/180)*cos(phiPts(i)*M_PI/180);
                double intersectY = dist*sin(thetaPts(j)*M_PI/180)*sin(phiPts(i)*M_PI/180);
                double intersectZ = dist*cos(thetaPts(j)*M_PI/180);

                // write the result to file
                outputResultsFile << intersectX << " " << intersectY << " " << intersectZ << std::endl;
            }
        }
    }

    // close the file
    outputResultsFile.close();

    return 0;
}