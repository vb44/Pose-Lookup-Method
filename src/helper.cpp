#include "helper.h"

Eigen::Matrix4d homogeneous(double roll, double pitch, double yaw, 
                            double x, double y, double z)
{
    Eigen::Matrix4d T;
    T.setZero();

    T(0,0) = cos(yaw)*cos(pitch);
    T(0,1) = cos(yaw)*sin(pitch)*sin(roll)-sin(yaw)*cos(roll);
    T(0,2) = cos(yaw)*sin(pitch)*cos(roll)+sin(yaw)*sin(roll);
    T(0,3) = x;
    T(1,0) = sin(yaw)*cos(pitch);
    T(1,1) = sin(yaw)*sin(pitch)*sin(roll)+cos(yaw)*cos(roll);
    T(1,2) = sin(yaw)*sin(pitch)*cos(roll)-cos(yaw)*sin(roll);
    T(1,3) = y;
    T(2,0) = -sin(pitch);
    T(2,1) = cos(pitch)*sin(roll);
    T(2,2) = cos(pitch)*cos(roll);
    T(2,3) = z;
    T(3,3) = 1;

    return T;
}

std::vector<double> hom2rpyxyz(Eigen::Matrix4d T)
{
    double ROLL = atan2(T(2,1),T(2,2));
    double PITCH = asin(-T(2,0));
    double YAW = atan2(T(1,0),T(0,0));
    double X = T(0,3);
    double Y = T(1,3);
    double Z = T(2,3);
    std::vector<double> result = {ROLL,PITCH,YAW,X,Y,Z};
    return result;
}

Eigen::MatrixXd readPointCloud(std::string fileName)
{
    // file format: each row is a Cartesian pt in [x y z] where
    // the delimiter is a space
    std::fstream file(fileName, std::ios_base::in);
    // if (!file) return EXIT_FAILURE;

    float item;
    Eigen::MatrixXd ptsRead;
    Eigen::MatrixXd ptsCorrected;
    std::vector<double> ptsFromFile;    // pts read from file
    unsigned int counter = 0;
    float pt;
    
    while (file >> pt)
        ptsFromFile.push_back(pt);

    unsigned int numPts = ptsFromFile.size()/3;
    ptsRead.resize(numPts,4);
    
    for (unsigned int i = 0; i < ptsFromFile.size(); i+=3)
    {
        ptsRead.row(counter) << ptsFromFile[i],
                                ptsFromFile[i+1],
                                ptsFromFile[i+2],1;
        counter++;
    }
    
    return ptsRead;
}

Eigen::MatrixXd generateHypotheses(Eigen::RowVectorXd seed,
                                   Eigen::RowVectorXd minDeviation,
                                   Eigen::RowVectorXd maxDeviation, 
                                   Eigen::RowVectorXd stepSizes)
{
    // generate a list of initial hypothesis
    Eigen::MatrixXd hypotheses;
    double hyp[6];
    unsigned int hypIndex = 0;
    unsigned int numRoll, numPitch, numYaw, numX, numY, numZ;
    numRoll  = round((maxDeviation[0] - minDeviation[0]) / stepSizes[0] + 1);
    numPitch = round((maxDeviation[1] - minDeviation[1]) / stepSizes[1] + 1);
    numYaw   = round((maxDeviation[2] - minDeviation[2]) / stepSizes[2] + 1);
    numX     = round((maxDeviation[3] - minDeviation[3]) / stepSizes[3] + 1);
    numY     = round((maxDeviation[4] - minDeviation[4]) / stepSizes[4] + 1);
    numZ     = round((maxDeviation[5] - minDeviation[5]) / stepSizes[5] + 1);

    hypotheses.resize(numRoll*numPitch*numYaw*numX*numY*numZ,6);

    for (unsigned int roll = 0; roll < numRoll; roll++) {
        hyp[0] = seed[0] + minDeviation[0] + stepSizes[0]*roll;
        for (unsigned int pitch = 0; pitch < numPitch; pitch++) {
            hyp[1] = seed[1] + minDeviation[1] + stepSizes[1]*pitch;
            for (unsigned int yaw = 0; yaw < numYaw; yaw++) {
                hyp[2] = seed[2] + minDeviation[2] + stepSizes[2]*yaw;
                for (unsigned int x = 0; x < numX; x++) {
                    hyp[3] = seed[3] + minDeviation[3] + stepSizes[3]*x;
                    for (unsigned int y = 0; y < numY; y++) {
                        hyp[4] = seed[4] + minDeviation[4] + stepSizes[4]*y;
                        for (unsigned int z = 0; z < numZ; z++) {
                            hyp[5] = seed[5] + minDeviation[5] + stepSizes[5]*z;
                            
                            // save each hypothesis to the hypotheses list
                            hypotheses.row(hypIndex) << hyp[0],hyp[1],hyp[2],
                                                        hyp[3],hyp[4],hyp[5];
                            hypIndex++;
                        }
                    }
                }
            }
        }
    }

    return hypotheses;
}

void readLookupTable(uint8_t* lookupTable, unsigned int* numXYZ,
                     std::string lookupTablePath)
{
    std::ifstream lookupTableFileStream(lookupTablePath, std::ios::binary);
    lookupTableFileStream.read(reinterpret_cast<char*>(&lookupTable[0]), 
                               numXYZ[0]*numXYZ[1]*numXYZ[2]);
    lookupTableFileStream.close();
}

std::vector<double> calculateEvidence(uint8_t* lookupTable,
                                      Eigen::Matrix4d lookupToModel,
                                      Eigen::MatrixXd hypotheses,
                                      Eigen::MatrixXd pointCloud,
                                      double* maxXYZ, unsigned int* numXYZ,
                                      double pointsPerMeter)
{
    std::vector<double> hypEvidence;
    int evidence;

    // calculate the evidence for each hypothesis
    for (unsigned int i = 0; i < hypotheses.rows(); i++)
    {    
        // transform the pointcloud measurements to the lookup frame
        Eigen::Matrix4d sensorToModel = homogeneous(hypotheses(i,0),
                                                    hypotheses(i,1),
                                                    hypotheses(i,2),
                                                    hypotheses(i,3),
                                                    hypotheses(i,4),
                                                    hypotheses(i,5));
        Eigen::MatrixXd pointcloudLookup = (lookupToModel*sensorToModel.inverse())
                                            * pointCloud.transpose(); 
        
        // initialise the evidence to zero
        evidence = 0;

        // iterate through the sensor measurements and sum the evidence
        for (unsigned int k = 0; k < pointCloud.rows(); k++)
        {
            // compute the lookup table indicies
            double x = pointcloudLookup(0,k);
            double y = pointcloudLookup(1,k);
            double z = pointcloudLookup(2,k);

            if (x >= 0 && x <= maxXYZ[0] &&
                y >= 0 && y <= maxXYZ[1] &&
                z >= 0 && z <= maxXYZ[2])
            {
                int xIndex = round(x*pointsPerMeter);
                int yIndex = round(y*pointsPerMeter);
                int zIndex = round(z*pointsPerMeter);
                unsigned int index = zIndex + yIndex*numXYZ[2] +
                                     xIndex*numXYZ[1]*numXYZ[2]; 
                evidence += lookupTable[index];
            }
        }
        hypEvidence.push_back(evidence);
    }
    return hypEvidence;
}