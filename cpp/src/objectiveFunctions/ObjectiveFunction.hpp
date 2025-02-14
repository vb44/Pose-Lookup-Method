#ifndef OBJECTIVE_FUNCTION_HPP
#define OBJECTIVE_FUNCTION_HPP

#include <vector>

#include <Eigen/Dense>

class ObjectiveFunction
{
    public:
        virtual void setPointCloud(const std::vector<Eigen::Vector4d> &pointCloud) = 0;
        virtual std::vector<int> calculateEvidence(const Eigen::MatrixXd &hypotheses) = 0;
        virtual ~ObjectiveFunction() = default;
};

#endif // OBJECTIVE_FUNCTION_HPP