#include "Msoe.hpp"

Msoe::Msoe(const ConfigParser &config) :
    sigma_(config.getSigma())
{
    std::cout << config.getModelFilePath() << std::endl;
    raycaster = new Raycaster(config.getModelFilePath());
}

Msoe::~Msoe()
{
}

void Msoe::setPointCloud(const std::vector<Eigen::Vector4d> &pointCloud)
{
    pointCloud_.resize(pointCloud.size(), 4);
    tbb::parallel_for(
        tbb::blocked_range<int>(0, pointCloud.size()),
        [&](tbb::blocked_range<int> r)
        {
            for (unsigned int i = r.begin(); i < r.end(); i++)
            {    
                pointCloud_.row(i) << pointCloud[i](0),
                                      pointCloud[i](1),
                                      pointCloud[i](2),
                                      1;
            }
        }
    );
    raycaster->computeRays(pointCloud);
    std::cout << "Finished setting the point cloud in MSoE: " << pointCloud_.rows() << " " << std::endl;
}

std::vector<int> Msoe::calculateEvidence(const Eigen::MatrixXd &hypotheses)
{
    evidences_.resize(hypotheses.rows());
    std::vector<double> measuredRanges = raycaster->getMeasuredRanges();
    std::cout << "Number of hypotheses: " << hypotheses.rows() << std::endl;

    // Calculate the evidence for each hypothesis.
    for (unsigned int i = 0; i < hypotheses.rows(); i++)
    {    
        // Transform the pointcloud measurements to the lookup frame.
        Eigen::Matrix4f sensorToModel = (utils::homogeneous(
                                            hypotheses(i,0),
                                            hypotheses(i,1),
                                            hypotheses(i,2),
                                            hypotheses(i,3),
                                            hypotheses(i,4),
                                            hypotheses(i,5))).cast<float>();

        // Raycast the scene.
        raycaster->setGeometryPose(sensorToModel);
        raycaster->raycast();
        auto raycastResult = raycaster->getRaycastResults();
        
        if (measuredRanges.size() != raycastResult.second.size())
        {
            std::cerr << "The number of measured and raycasted results do not match! " << measuredRanges.size() << " " << raycastResult.second.size() << std::endl;
            exit(1);
        }

        // Initialise the evidence to zero.
        double evidence = 0;
        int counter = 0;
        for (size_t j = 0; j < raycastResult.first.size(); j++)
        {
            if (raycastResult.second[j])
            {
                // TODO: Can use norm here?
                counter++;
                double raycastNorm = std::sqrt(
                    raycastResult.first[j].x()*raycastResult.first[j].x() +
                    raycastResult.first[j].y()*raycastResult.first[j].y() +
                    raycastResult.first[j].z()*raycastResult.first[j].z());
                evidence += std::exp(-std::pow(raycastNorm-measuredRanges[j],2)/(2*std::pow(sigma_,2)));
            }
        }
        
        // Relative reward - scale by 100 to use integers.
        evidences_[i] = int(evidence*100);
    }
    return evidences_;   
}