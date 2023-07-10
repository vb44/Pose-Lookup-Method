#include "helper.h"
#include <typeinfo>
#include <memory>
#include "open3d/Open3D.h"
#include "open3d/t/geometry/RaycastingScene.h"
#include "open3d/core/EigenConverter.h"
// #include <iostream>

int main(int argc, char* argv[])
{
    std::string fileName = "../sample_data/StanfordBunny.stl";
    std::string lookupFilename = "../outputs/bunny_lookup_table_sigma_1m_stepSize_1mm.lookup";
    Eigen::MatrixXd lookupToModel = homogeneous(0,0,0,50,40,10);
    double stepSize = 1;
    int maxXYZ[3] = {100,100,100}; // [x,y,z]
    double sigma = 1;

    // GENERATE THE QUERY POINTS
    unsigned int numX = round(maxXYZ[0]/stepSize) + 1;
    unsigned int numY = round(maxXYZ[1]/stepSize) + 1;
    unsigned int numZ = round(maxXYZ[2]/stepSize) + 1;
    Eigen::VectorXd xPts = Eigen::VectorXd::LinSpaced(numX,0,maxXYZ[0]);
    Eigen::VectorXd yPts = Eigen::VectorXd::LinSpaced(numY,0,maxXYZ[1]);
    Eigen::VectorXd zPts = Eigen::VectorXd::LinSpaced(numZ,0,maxXYZ[2]);

    // READ THE MESH ----------------------------------------------------------
    auto mesh = std::make_shared<open3d::geometry::TriangleMesh>();
    open3d::io::ReadTriangleMesh(fileName,*mesh);
    auto tmesh = open3d::t::geometry::TriangleMesh::FromLegacy(*mesh,open3d::core::Float32, open3d::core::Int64);

    // TRANSFORM THE MODELT O THE LOOKUP FRAME --------------------------------
    open3d::core::Tensor tf = open3d::core::eigen_converter::EigenMatrixToTensor(lookupToModel);
    auto tmeshTf = tmesh.Transform(tf);


    // CREATE THE SCENE -------------------------------------------------------
    open3d::t::geometry::RaycastingScene scene;
    scene.AddTriangles(tmeshTf);

    // WRITE THE LOOKUP TABLE
    std::ofstream lookupFile;
    lookupFile.open(lookupFilename, std::ios::binary);

    uint8_t reward;
    double queryPt[3];

    for (unsigned int i = 0; i < numX; i++)
    {
        for (unsigned int j = 0; j < numY; j++)
        {
            for (unsigned int k = 0; k < numZ; k++)
            {
                // query pt
                queryPt[0] = xPts(i);
                queryPt[1] = yPts(j);
                queryPt[2] = zPts(k);

                // find the closest distance to the model
                auto queryPtTensor = open3d::core::Tensor::Zeros({1,3},open3d::core::Float32);
                queryPtTensor.SetItem({open3d::core::TensorKey::Index(0)},
                                       open3d::core::Tensor::Init<double>({queryPt[0],queryPt[1],queryPt[2]}));
                auto closestDistanceTensor = open3d::core::Tensor::Zeros({1,1},open3d::core::Float32);
                closestDistanceTensor.SetItem({open3d::core::TensorKey::Index(0)},scene.ComputeDistance(queryPtTensor));
                auto closestDistance = open3d::core::eigen_converter::TensorToEigenMatrixXd(closestDistanceTensor);

                // calculate the reward
                reward = 255 * exp(-0.5 * closestDistance(0,0) * closestDistance(0,0) / (sigma * sigma));
                
                // write to a binary file here
                lookupFile << reward;
            }
        }
    }

    // close the file
    lookupFile.close();

    return 0;
}
