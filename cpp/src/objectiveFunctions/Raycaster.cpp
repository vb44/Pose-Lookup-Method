#include "Raycaster.hpp"

Raycaster::Raycaster(const std::string &geometryPath)
    : geometryPath_(geometryPath)
{
    device_ = rtcNewDevice(nullptr);
    embreeScene_ = rtcNewScene(device_);
    geom_ = rtcNewGeometry(device_, RTC_GEOMETRY_TYPE_TRIANGLE);

    // Import the geometry.
    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(geometryPath_, aiProcess_Triangulate);
    if (!scene || !scene->HasMeshes())
    {
        throw std::runtime_error("pose_estimator: Failed to load STL file");
    }

    aiMesh* mesh = scene->mMeshes[0];
    originalVertices_.resize(mesh->mNumVertices);
    originalIndices_.resize(mesh->mNumFaces * 3);

    for (unsigned int i = 0; i < mesh->mNumVertices; i++)
    {
        originalVertices_[i] = Eigen::Vector3f(mesh->mVertices[i].x,
                                               mesh->mVertices[i].y,
                                               mesh->mVertices[i].z);
    }

    for (unsigned int i = 0; i < mesh->mNumFaces; i++)
    {
        originalIndices_[i * 3] = mesh->mFaces[i].mIndices[0];
        originalIndices_[i * 3 + 1] = mesh->mFaces[i].mIndices[1];
        originalIndices_[i * 3 + 2] = mesh->mFaces[i].mIndices[2];
    }

    geometryPose_ = Eigen::Matrix4f::Identity();
    transformGeometry();
    
    // Avoid optimizations that may reduce algorithmic accuracy
    rtcSetSceneFlags(embreeScene_, RTC_SCENE_FLAG_DYNAMIC | RTC_SCENE_FLAG_ROBUST);
    rtcCommitScene(embreeScene_);

}

Raycaster::~Raycaster()
{
    rtcReleaseScene(embreeScene_);
    rtcReleaseDevice(device_);
}

void Raycaster::computeRays(const std::vector<Eigen::Vector4d> &pointCloud)
{
    directions_.resize(pointCloud.size());
    measuredRanges_.resize(pointCloud.size());

    tbb::parallel_for(
        tbb::blocked_range<int>(0, pointCloud.size()),
        [&](tbb::blocked_range<int> r)
        {
            for (size_t i = r.begin(); i < r.end(); i++)
            {
                // Estimate the ray directions
                float theta = std::atan2(pointCloud[i](1), pointCloud[i](0));
                double ptNorm = std::sqrt(pointCloud[i](0)*pointCloud[i](0) +
                                          pointCloud[i](1)*pointCloud[i](1) +
                                          pointCloud[i](2)*pointCloud[i](2));
                float phi = std::acos(pointCloud[i](2) / ptNorm);
                directions_[i] = {std::sin(phi) * std::cos(theta),
                                  std::sin(phi) * std::sin(theta),
                                  std::cos(phi)};
                measuredRanges_[i] = ptNorm;
            }
        }
    );
    std::cout << "Finished setting the rays in MSoE: " << directions_.size() << std::endl;
}

std::pair<std::vector<Eigen::Vector3f>, std::vector<int>> Raycaster::getRaycastResults()
{
    return std::make_pair(hits_, hitsValid_);
}

void Raycaster::setGeometryPose(const Eigen::Matrix4f &geometryPose)
{
    geometryPose_ = geometryPose;
    transformGeometry();
}

void Raycaster::transformGeometry()
{
    if (geom_ != nullptr)
    {
        rtcDetachGeometry(embreeScene_, geomID_);
        rtcReleaseGeometry(geom_);
    }

    geom_ = rtcNewGeometry(device_, RTC_GEOMETRY_TYPE_TRIANGLE);

    Eigen::Vector3f* vertices = reinterpret_cast<Eigen::Vector3f*>(
        rtcSetNewGeometryBuffer(geom_, RTC_BUFFER_TYPE_VERTEX, 0,
                                RTC_FORMAT_FLOAT3, sizeof(Eigen::Vector3f),
                                originalVertices_.size()));
    unsigned int* indices = reinterpret_cast<unsigned int*>(
        rtcSetNewGeometryBuffer(geom_, RTC_BUFFER_TYPE_INDEX, 0,
                                RTC_FORMAT_UINT3, sizeof(unsigned int) * 3,
                                originalIndices_.size()));

    tbb::parallel_for(
        tbb::blocked_range<int>(0, originalVertices_.size()),
        [&](tbb::blocked_range<int> r)
        {
            for (size_t i = r.begin(); i < r.end(); i++)
            {
                Eigen::Vector4f transformed = geometryPose_ *
                                    Eigen::Vector4f(originalVertices_[i].x(),
                                                    originalVertices_[i].y(),
                                                    originalVertices_[i].z(), 1.0f);
                vertices[i] = transformed.head<3>();
            }
        }
    );

    std::copy(originalIndices_.begin(), originalIndices_.end(), indices);
    rtcCommitGeometry(geom_);
    geomID_ = rtcAttachGeometry(embreeScene_, geom_);
    rtcCommitScene(embreeScene_);
}

void Raycaster::raycast()
{
    hits_.clear();
    hitsValid_.clear();
    hits_.resize(directions_.size());
    hitsValid_.resize(directions_.size(), 0);
    RTCIntersectContext context;
    rtcInitIntersectContext(&context);

    tbb::parallel_for(
        tbb::blocked_range<int>(0, directions_.size()),
        [&](tbb::blocked_range<int> r)
        {
            for (size_t i = r.begin(); i < r.end(); i++)
            {
                RTCRayHit rayhit;
                rayhit.ray.org_x = 0.0f;
                rayhit.ray.org_y = 0.0f;
                rayhit.ray.org_z = 0.0f;
                rayhit.ray.dir_x = directions_[i].x();
                rayhit.ray.dir_y = directions_[i].y();
                rayhit.ray.dir_z = directions_[i].z();
                rayhit.ray.tnear = 0.0f;
                rayhit.ray.tfar = FLT_MAX;
                rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;

                rtcIntersect1(embreeScene_, &context, &rayhit);
                if (rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID)
                {
                    float t = rayhit.ray.tfar;
                    hits_[i] = {rayhit.ray.org_x + rayhit.ray.dir_x * t,
                                rayhit.ray.org_y + rayhit.ray.dir_y * t,
                                rayhit.ray.org_z + rayhit.ray.dir_z * t};
                    hitsValid_[i] = 1;
                }
            }
        }
    );
}

void Raycaster::printHits()
{
    for (size_t i = 0; i < hits_.size(); i++)
    {
        if (hitsValid_[i])
        {
            std::cout << hits_[i].x() << " "
                      << hits_[i].y() << " "
                      << hits_[i].z() << std::endl;
        }
    }
    std::cout << hits_.size() << std::endl;
}

std::vector<double> Raycaster::getMeasuredRanges()
{
    return measuredRanges_;
}