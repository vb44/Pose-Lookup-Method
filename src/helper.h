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
#include <boost/random.hpp>
#include <boost/math/distributions/normal.hpp>
#include <Eigen/Dense>

/**
 * @brief A container to store all the raycast settings.
 */
struct raycastParams {
    std::string modelFileName;      // path of the known geometry 
    std::string outputFileName;     // path of the output file name
    Eigen::MatrixXd sensorToModel;  // homogeneous transform from the senor to the model
    std::vector<double> phiRange;   // heading   [min,max,increment] (degrees)
    std::vector<double> thetaRange; // elevation [min,max,increment] (degrees)
    bool verbose = false;           // print the parameters to the termninal
};

/**
 * @brief A container to store all the lookup table generation settings.
 */
struct lookupTableParams {
    std::string modelFileName;      // path of the known geometry 
    std::string outputFileName;     // path of the output file name
    Eigen::MatrixXd lookupToModel;  // homogeneous transform from the lookup table to the model
    std::vector<int> maxXYZ;        // maximum bounds of the lookup table [m]
    double stepSize;                // lookup table step size [m]
    double sigma;                   // the reward sigma [m]
};

/**
 * @brief A container to store all the plum settings.
 */
struct plumParams {
    std::string pointCloudFile;         // path to the point cloud file 
    std::string lookupTableFile;        // LOOKUP_TABLE: path to the lookup table
    Eigen::MatrixXd lookupToModel;      // LOOKUP_TABLE: homogeneous transform from the lookup table to the model
    std::vector<double> maxXYZ;            // LOOKUP_TABLE: maximum bounds of the lookup table [m]
    double stepSize;                    // LOOKUP_TABLE: lookup table step size [m]
    Eigen::RowVectorXd seed;           // SEARCH_HEURISTIC: seed for the pose hypotheses
    Eigen::RowVectorXd minDeviation;   // SEARCH_HEURSITIC: minDeviation for the pose hypotheses
    Eigen::RowVectorXd maxDeviation;   // SEARCH_HEURISTIC: maxDeviation for the pose hypotheses
    Eigen::RowVectorXd stepSizes;      // SEARCH_HEURISTIC: stepSizes for the pose hypotheses
    double rotSigma;                    // SEARCH_HEURISTIC: pose hypotheses resampling rotational sigma
    double transSigma;                  // SEARCH_HEURISTIC: pose hypotheses resampling translational sigma
    int noIterations;                // SEARCH_HEURISTIC: no. of iterations for the particle filter
    int resampleSize;                // SEARCH_HEURISTIC: resample size for the particle filter
};


/**
 * @brief Parse the raycast utility commandline arguments are store the
 *        configuration parameters and the test settings. 
 * 
 * @param parameters    An params type to store the configuration parameters. 
 * @param argc          The number of arguments entered at the commandline.
 * @param argv          The commandline input.
 * @return int          Returns 0 if arguments are successfully passed, or 1
 *                      if there is an error in parsing the arguments.
 */
int parseArgsRaycast(raycastParams* config, int argc, char* argv[]);

/**
 * @brief Parse the lookup table utility commandline arguments are store the
 *        configuration parameters and the test settings. 
 * 
 * @param parameters    An params type to store the configuration parameters. 
 * @param argc          The number of arguments entered at the commandline.
 * @param argv          The commandline input.
 * @return int          Returns 0 if arguments are successfully passed, or 1
 *                      if there is an error in parsing the arguments.
 */
int parseArgsLookupTable(lookupTableParams* config, int argc, char* argv[]);

/**
 * @brief Parse the plum utility commandline arguments are store the
 *        configuration parameters and the test settings. 
 * 
 * @param parameters    An params type to store the configuration parameters. 
 * @param argc          The number of arguments entered at the commandline.
 * @param argv          The commandline input.
 * @return int          Returns 0 if arguments are successfully passed, or 1
 *                      if there is an error in parsing the arguments.
 */
int parseArgsPlum(plumParams* config, int argc, char* argv[]);

/**
 * @brief Write the configuration settings for the raycasting operation.
 * 
 * @param config            Configuration parameters used to print the MATLAB
 *                          readable config file.
 */
void writeRaycastConfigFile(raycastParams* config);

/**
 * @brief Write the configuration settings for the lookup generation.
 * 
 * @param config            Configuration parameters used to print the MATLAB
 *                          readable config file.
 */
void writeLookupConfigFile(lookupTableParams* config);

/**
 * @brief Construct a homogeneous (4x4) matrix.
 * 
 * @param roll              Roll angle in radians.
 * @param pitch             Pitch angle in radians.
 * @param yaw               Yaw angle in radians.
 * @param x                 X position in metres.
 * @param y                 Y position in metres.
 * @param z                 Z position in metres.
 * @return Eigen::Matrix4d  The homogeneous (4x4) matrix constructed. 
 */
Eigen::Matrix4d homogeneous(double roll, double pitch, double yaw, 
                            double x, double y, double z);

/**
 * @brief Convert from a homogeneous (4x4) matrix to a homogeneous (6x1) vector.
 * 
 * @param T                     Input homogeneous (4x4) matrix.
 * @return std::vector<double>  The constructed homogeneous (6x1) vector.
 */
std::vector<double> hom2rpyxyz(Eigen::Matrix4d T);

/**
* @brief Read the point cloud file.
*
* @param fileName         The filename of the point cloud file to read.
* @return Eigen::MatrixXd The point cloud read from filename.
*/
Eigen::MatrixXd readPointCloud(std::string fileName);

/**
* @brief Generate the initial list of hypothesis for the particle filter.
*
* @param params             The test parameters.
* @return Eigen::MatrixXd   Intitial list of pose hypotheses.
*/
Eigen::MatrixXd generateHypotheses(plumParams* params);

/**
* @brief Read the lookup table.
*
* @param lookupTable        Container to store the lookup table.
* @param numXYZ             The number of elements in each dimension.
* @param lookupTablePath    Path to the lookup table file.
*/
void readLookupTable(uint8_t* lookupTable, 
                     unsigned int* numXYZ,
                     std::string lookupTablePath);

/**
* @brief Calculate the evidenvce for a set of pose hypotheses.
*
* @param lookupTable            The lookup table.
* @param lookupToModel          Homogeneous transform from the lookup frame to
*                               the lookup model.
* @param hypotheses             The hypotheses set to score.
* @param pointCloud             The point cloud from the sensor.
* @param maxXYZ                 The maximum bounds of the lookup table.
* @param numXYZ                 The number of elements in the lookup table in
*                               each dimension.
* @param pointsPerMeter         The number of points per meter.
* @return std::vector<double>   The calculated evidence for each pose
*                               hypothesis in hypotheses.
*/
std::vector<double> calculateEvidence(uint8_t* lookupTable,
                                      Eigen::Matrix4d lookupToModel,
                                      Eigen::MatrixXd hypotheses,
                                      Eigen::MatrixXd pointCloud,
                                      std::vector<double> maxXYZ, unsigned int* numXYZ,
                                      double pointsPerMeter);