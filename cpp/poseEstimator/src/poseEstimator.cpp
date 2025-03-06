#include <chrono>
#include <filesystem>
#include <memory>

#include "ConfigParser.hpp"
#include "Msoe.hpp"
#include "ParticleFilter.hpp"
#include "Plum.hpp"
#include "PointCloud.hpp"
#include "utils.hpp"

int main(int argc, char* argv[])
{
    //  Argument parsing.
    ConfigParser config(argc, argv);
    int configStatus = config.parseConfig();
    if (configStatus) exit(1);

    // Load the scan paths.
    std::vector<std::string> scanFiles;
    for (auto const& dir_entry : 
            std::filesystem::directory_iterator(config.getPcPath()))
    {
        scanFiles.push_back(dir_entry.path());
    }

    // Sort the scans in order of the file name.
    std::sort(scanFiles.begin(), scanFiles.end(), utils::compareStrings);
    unsigned int numScans = scanFiles.size();

    // Store the pose estimates in (roll,pitch,yaw,x,y,z,registrationScore)
    // format.
    std::vector<std::vector<double> > poseEstimates(numScans, 
                                                    std::vector<double>(7));
    
    // Start the timer.
    auto startReg = std::chrono::high_resolution_clock::now();

    PointCloud pointCloud(config);
    std::shared_ptr<ObjectiveFunction> objFunc;

    // Instantiate the configured objective function.
    std::string poseEstMethod = config.getPoseEstMethod();
    if (!poseEstMethod.compare("plum"))
    {
        objFunc = std::make_shared<Plum>(config);
    } else if (!poseEstMethod.compare("msoe"))
    {
        objFunc = std::make_shared<Msoe>(config);
    }
    ParticleFilter partilceFilter(config, objFunc);

    // Loop over all input scans and estimate the registration results.
    for (unsigned int scanNum = 0; scanNum < numScans; scanNum++)
    {
        auto start = std::chrono::high_resolution_clock::now(); 
        pointCloud.readScan(scanFiles[scanNum]);

        objFunc->setPointCloud(pointCloud.getPtCloud());

        std::vector<double> poseEstimate = partilceFilter.findBestGeometryPose();

        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop-start);
    
        std::cout << poseEstimate[0] << "," << poseEstimate[1] << ","
                  << poseEstimate[2] << "," << poseEstimate[3] << ","
                  << poseEstimate[4] << "," << poseEstimate[5] << ","
                  << poseEstimate[6] << "," << duration.count() << std::endl;
    }
    
    return 0;
}