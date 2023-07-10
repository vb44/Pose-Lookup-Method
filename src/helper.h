#include <math.h>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <iterator>
#include <utility>
#include <chrono>
#include <random>
#include <cstdlib>
#include <assert.h>
#include <sstream>
#include <boost/algorithm/string.hpp>

#include <Eigen/Dense>

Eigen::Matrix4d homogeneous(double roll, double pitch, double yaw, 
                            double x, double y, double z);

std::vector<double> hom2rpyxyz(Eigen::Matrix4d T);

Eigen::MatrixXd readPointCloud(std::string fileName);

Eigen::MatrixXd generateHypotheses(Eigen::RowVectorXd seed,
                                   Eigen::RowVectorXd minDeviation,
                                   Eigen::RowVectorXd maxDeviation,
                                   Eigen::RowVectorXd stepSizes);

void readLookupTable(uint8_t* lookupTable, 
                     unsigned int* numXYZ,
                     std::string lookupTablePath);

std::vector<double> calculateEvidence(uint8_t* lookupTable,
                                      Eigen::Matrix4d lookupToModel,
                                      Eigen::MatrixXd hypotheses,
                                      Eigen::MatrixXd pointCloud,
                                      double* maxXYZ, unsigned int* numXYZ,
                                      double pointsPerMeter);