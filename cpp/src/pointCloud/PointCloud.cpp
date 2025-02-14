#include "PointCloud.hpp"

PointCloud::PointCloud(const ConfigParser &config)
    : subsampleRadius_(config.getPcSubsampleRadius()),
      maxSensorRange_(config.getMaxSensorRange()),
      minSensorRange_(config.getMinSensorRange())
{
}

void PointCloud::readScan(const std::string &fileName)
{
    ptCloud_.clear();
    allPoints_.clear();
    std::ifstream file(fileName, std::ios::in | std::ios::binary);
    if (!file)
    {
        throw std::runtime_error("Failed to open file: " + fileName);
    }

    float item;
    std::vector<double> ptsFromFile;
    int counter = 0;
    
    while (file.read((char*)&item, sizeof(item)))
    {
        ptsFromFile.push_back(item);
    }
    file.close();

    // .bin format
    unsigned int numPts = ptsFromFile.size() / NUM_COLUMNS_BIN; 
 
    for (unsigned int i = 0; i < ptsFromFile.size(); i+=NUM_COLUMNS_BIN)
    {
        // Save the pt if it is within the maximum and mininmum sensor ranges.
        double normSquared = pow(ptsFromFile[i], 2) + 
                             pow(ptsFromFile[i+1], 2) + 
                             pow(ptsFromFile[i+2], 2); 
        if ((normSquared > pow(minSensorRange_, 2)) &&
            (normSquared < pow(maxSensorRange_, 2)))
        {
            ptCloud_.push_back({ptsFromFile[i],
                                ptsFromFile[i+1],
                                ptsFromFile[i+2],
                                1});
            
            // Save the pt index for subsampling.
            allPoints_.insert(counter); 
            counter++;
        }
    }

    // Subsample the point cloud.
    subsample_(ptCloud_, subsampleRadius_);
}

const std::vector<Eigen::Vector4d>& PointCloud::getPtCloud() const
{
    return ptCloud_;
}

void PointCloud::subsample_(std::vector<Eigen::Vector4d> &pts,
                            double subsampleRadius)
{
    std::vector<Eigen::Vector4d> ptsSubsampled;
    
    // Nanoflann uses the squared radius.
    subsampleRadius = pow(subsampleRadius, 2);
    convertToPointCloudKdTree_(pts);

    // Create a Kd tree (dimension, scan, max leaf).
    my_kd_tree_t *scanKdTree = new my_kd_tree_t(3, pcForKdTree_,{10});
    unsigned int counter = 0;

    // Subsample radially.
    for (unsigned int i : allPoints_)
    {
        std::vector<nanoflann::ResultItem<uint32_t, double>> ret_matches;
        const double query_pt[3] = {pts[i][0], pts[i][1], pts[i][2]};
        const size_t nMatches = scanKdTree->radiusSearch(&query_pt[0],
                                subsampleRadius, ret_matches);
        for (unsigned int j = 0; j < nMatches; j++)
        {
            if (i != ret_matches[j].first)
            {
                allPoints_.erase(ret_matches[j].first);
            }
        }
        ptsSubsampled.push_back({pts[i][0], pts[i][1], pts[i][2], 1});
    }
    delete scanKdTree;

    // Overwrite the scan with the subsampled points.
    pts = ptsSubsampled;
}

void PointCloud::convertToPointCloudKdTree_(
                 const std::vector<Eigen::Vector4d> &pts)
{
    size_t pcLength = pts.size();
    pcForKdTree_.pts.clear();
    pcForKdTree_.pts.resize(pcLength);

    tbb::parallel_for(
    tbb::blocked_range<int>(0, pcLength),
    [&](tbb::blocked_range<int> r)
    { 
        for (size_t i = r.begin(); i < r.end(); i++)
        {
            pcForKdTree_.pts[i].x = pts[i](0);
            pcForKdTree_.pts[i].y = pts[i](1);
            pcForKdTree_.pts[i].z = pts[i](2);        
        } 
    });
}