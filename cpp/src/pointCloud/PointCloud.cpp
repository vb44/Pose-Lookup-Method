#include "PointCloud.hpp"

PointCloud::PointCloud(const ConfigParser &config)
    : subsampleRadius_(config.getPcSubsampleRadius()),
      maxSensorRange_(config.getMaxSensorRange()),
      minSensorRange_(config.getMinSensorRange()),
      pcRegionOfInterest_(config.getPcRegionOfInterest())
{
    std::vector<double> platformToSensor = config.getPlatformToSensor();
    platformToSensor_ = utils::homogeneous(platformToSensor[0],
                                           platformToSensor[1],
                                           platformToSensor[2],
                                           platformToSensor[3],
                                           platformToSensor[4],
                                           platformToSensor[5]);
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
    subsample(ptCloud_, subsampleRadius_);

    // Transform the point cloud to the platform frame.
    // If the platform frame is not specified, this operation will not change
    // the point cloud.
    transformToPlatformFrame();

    // Segment points in the region of interest if specified.
    if (pcRegionOfInterest_.size() == 6)
    {
        getRegionOfInterest();
    }
}

std::vector<Eigen::Vector4d>& PointCloud::getPtCloud()
{
    return ptCloud_;
}

void PointCloud::subsample(std::vector<Eigen::Vector4d> &pts,
                            double subsampleRadius)
{
    std::vector<Eigen::Vector4d> ptsSubsampled;
    
    // Nanoflann uses the squared radius.
    subsampleRadius = pow(subsampleRadius, 2);
    convertToPointCloudKdTree(pts);

    // Create a Kd tree (dimension, point cloud, max leaf).
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

    // Overwrite the point cloud with the subsampled points.
    pts = ptsSubsampled;
}

void PointCloud::convertToPointCloudKdTree(
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
        }
    );
}

void PointCloud::transformToPlatformFrame()
{
    tbb::parallel_for(
        tbb::blocked_range<int>(0, ptCloud_.size()),
        [&](tbb::blocked_range<int> r)
        { 
            for (size_t i = r.begin(); i < r.end(); i++)
            {
                ptCloud_[i] = platformToSensor_ * ptCloud_[i];  
            } 
        }
    );
}

void PointCloud::getRegionOfInterest()
{
    std::vector<Eigen::Vector4d> ptCloudFiltered;

    // Save the point cloud measurements in the region of interest.
    for (int i = 0; i < ptCloud_.size(); i++)
    {
        if (!((ptCloud_[i][0] < pcRegionOfInterest_[0]) ||
              (ptCloud_[i][0] > pcRegionOfInterest_[1]) || 
              (ptCloud_[i][1] < pcRegionOfInterest_[2]) ||
              (ptCloud_[i][1] > pcRegionOfInterest_[3]) || 
              (ptCloud_[i][2] < pcRegionOfInterest_[4]) ||
              (ptCloud_[i][2] > pcRegionOfInterest_[5])))
        {
            ptCloudFiltered.emplace_back(ptCloud_[i]);
        }
    }
    ptCloud_ = ptCloudFiltered;
}

void PointCloud::printPtCloud()
{
    for (const auto &pt : ptCloud_)
    {
        std::cout << pt(0) << " " << pt(1) << " " << pt(2) << std::endl;
    }
}