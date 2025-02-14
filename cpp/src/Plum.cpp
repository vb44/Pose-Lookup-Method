#include "Plum.hpp"

Plum::Plum(const ConfigParser &config)
{
    lookupTable_.lookupTablePath = config.getLookupTableFile();
    std::vector<double> lookupToModel = config.getLookupTableToModel();
    lookupTable_.lookupTableToModel = utils::homogeneous(lookupToModel[0],
                                                         lookupToModel[1],
                                                         lookupToModel[2],
                                                         lookupToModel[3],
                                                         lookupToModel[4],
                                                         lookupToModel[5]);
    lookupTable_.maxXyz = config.getLookupTableMaxXyz();
    lookupTable_.numXyz.resize(3);
    lookupTable_.stepSize = config.getLookupTableStepSize();

    lookupTable_.readLookupTable();
}

Plum::~Plum()
{
    if (lookupTable_.lookupTable)
    {
        free(lookupTable_.lookupTable);
        lookupTable_.lookupTable = nullptr;
    }
}

void Plum::setPointCloud(const std::vector<Eigen::Vector4d> &pointCloud)
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
}

std::vector<double> Plum::calculateEvidence(const Eigen::MatrixXd &hypotheses)
{
    evidences_.resize(hypotheses.rows());

    // Calculate the evidence for each hypothesis.
    tbb::parallel_for(
    tbb::blocked_range<int>(0, hypotheses.rows()),
    [&](tbb::blocked_range<int> r)
    {
        for (unsigned int i = r.begin(); i < r.end(); i++)
        {    
            // Transform the pointcloud measurements to the lookup frame.
            Eigen::Matrix4d sensorToModel = utils::homogeneous(
                                                hypotheses(i,0),
                                                hypotheses(i,1),
                                                hypotheses(i,2),
                                                hypotheses(i,3),
                                                hypotheses(i,4),
                                                hypotheses(i,5));
            Eigen::MatrixXd pointcloudLookup = (
                lookupTable_.lookupTableToModel*sensorToModel.inverse())
                                                * pointCloud_.transpose(); 
            // Initialise the evidence to zero
            int evidence = 0;

            // Iterate through the sensor measurements and sum the evidence.
            double x, y, z;
            unsigned int xIndex, yIndex, zIndex, lookupIndex;
            for (size_t k = 0; k < pointCloud_.rows(); k++)
            {
                // Compute the lookup table indicies
                x = pointcloudLookup(0,k);
                y = pointcloudLookup(1,k);
                z = pointcloudLookup(2,k);

                if (x >= 0 && x <= lookupTable_.maxXyz[0] &&
                    y >= 0 && y <= lookupTable_.maxXyz[1] &&
                    z >= 0 && z <= lookupTable_.maxXyz[2])
                {
                    xIndex = round(x*lookupTable_.pointsPerMeter);
                    yIndex = round(y*lookupTable_.pointsPerMeter);
                    zIndex = round(z*lookupTable_.pointsPerMeter);
                    lookupIndex = zIndex +
                                  yIndex*lookupTable_.numXyz[2] +
                                  xIndex*lookupTable_.numXyz[1]*
                                         lookupTable_.numXyz[2]; 
                    evidence += lookupTable_.lookupTable[lookupIndex];
                }
            }
            evidences_[i] = evidence;
        }
    });

    return evidences_;   
}