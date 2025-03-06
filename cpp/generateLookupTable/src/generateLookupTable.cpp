# include <fstream>

#include "open3d/Open3D.h"
#include "open3d/t/geometry/RaycastingScene.h"
#include "open3d/core/EigenConverter.h"

#include "ConfigParserLookup.hpp"
#include "utils.hpp"

int main(int argc, char* argv[])
{
    // Argument parsing.
    ConfigParserLookup config(argc, argv);
    int configStatus = config.parseConfig();
    if (configStatus) exit(1);

    // Generate the query points for the lookup table.
    std::vector<double> maxBounds = config.getMaxBounds();
    double stepSize = config.getStepSize();
    unsigned int numX = round(maxBounds[0]/stepSize) + 1;
    unsigned int numY = round(maxBounds[1]/stepSize) + 1;
    unsigned int numZ = round(maxBounds[2]/stepSize) + 1;
    Eigen::VectorXd xPts = Eigen::VectorXd::LinSpaced(numX,0,maxBounds[0]);
    Eigen::VectorXd yPts = Eigen::VectorXd::LinSpaced(numY,0,maxBounds[1]);
    Eigen::VectorXd zPts = Eigen::VectorXd::LinSpaced(numZ,0,maxBounds[2]);

    // Read the geometry model.
    auto mesh = std::make_shared<open3d::geometry::TriangleMesh>();
    open3d::io::ReadTriangleMesh(config.getModelFilePath(),*mesh);
    auto tmesh = open3d::t::geometry::TriangleMesh::FromLegacy(*mesh,open3d::core::Float32, open3d::core::Int64);

    // Transform the model to the lookup frame.
    std::vector<double> lookupToModel = config.getLookupTableToModel();
    open3d::core::Tensor tf = open3d::core::eigen_converter::EigenMatrixToTensor(
                                utils::homogeneous(lookupToModel[0], lookupToModel[1],
                                                   lookupToModel[2], lookupToModel[3],
                                                   lookupToModel[4], lookupToModel[5]));
    auto tmeshTf = tmesh.Transform(tf);
    
    // Create the scene.
    open3d::t::geometry::RaycastingScene scene;
    scene.AddTriangles(tmeshTf);

    // Write the lookup table.
    std::ofstream lookupFile;
    lookupFile.open(config.getOutputFileName()+".lookup", std::ios::binary);
    uint8_t reward;
    double queryPt[3];
    double sigma2 = config.getSigma() * config.getSigma();
    long numEntries = numX * numY * numZ;
    long counter = 0;
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
                reward = 255 * exp(-0.5 * closestDistance(0,0) * closestDistance(0,0) / (sigma2));
                lookupFile << reward;

                // Print the current progress to terminal.
                counter++;
                printf("\rProgress: %ld/%ld", counter, numEntries);
                fflush(stdout);
            }
        }
    }
    std::cout << std::endl;

    lookupFile.close();

    // Write the configuration file.
    config.writeConfig();

    return 0;
}