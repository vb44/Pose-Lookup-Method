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

Eigen::MatrixXd generateHypotheses(plumParams* params)
{
    Eigen::RowVectorXd seed = params->seed;
    Eigen::RowVectorXd minDeviation = params->minDeviation;
    Eigen::RowVectorXd maxDeviation = params->maxDeviation;
    Eigen::RowVectorXd stepSizes = params->stepSizes;
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
                                      std::vector<double> maxXYZ, unsigned int* numXYZ,
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

int parseArgsRaycast(raycastParams* config, int argc, char* argv[])
{
    // check the number of arguments
    if (argc != 11)
    {
        std::cerr << "raycastModel usage:"                                                          << std::endl <<
                     "./raycastModel "                                                              << std::endl <<
                     "--modelFileName        \"path_to_geometry\" "                                 << std::endl <<
                     "--outputFileName       \"path_to_file\" "                                     << std::endl <<
                     "--sensorToModel        \"roll(deg),pitch(deg),yaw(deg),x(m),y(m),z(m)\" "     << std::endl <<
                     "--headingRange         \"min(deg),max(deg),increment(deg)\" "                 << std::endl <<
                     "--elevationRange       \"min(deg),max(deg),increment(deg)\" "                 << std::endl;
        return 1;
    }
    
    // parse the arguments
    for (unsigned int i = 0; i < argc; i++)
    {
        if (strcmp(argv[i],"--modelFileName") == 0)
        {
            config->modelFileName = argv[i+1];        
        }
        if (strcmp(argv[i],"--outputFileName") == 0)
        {
            config->outputFileName = argv[i+1];        
        }
        if (strcmp(argv[i],"--sensorToModel") == 0)
        {
            std::stringstream str(argv[i+1]);
            std::vector<double> input;
            while (str.good())
            {
                std::string substr;
                getline(str,substr,',');
                input.push_back(std::stod(substr));
            }
            if (input.size() != 6) // need all six parameters
                return 1;
            config->sensorToModel = homogeneous(input[0]*M_PI/180,
                                                input[1]*M_PI/180,
                                                input[2]*M_PI/180,
                                                input[3],input[4],input[5]);
        }
        if (strcmp(argv[i],"--headingRange") == 0)
        {
            std::stringstream str(argv[i+1]);
            while (str.good())
            {
                std::string substr;
                getline(str,substr,',');
                config->phiRange.push_back(std::stod(substr));
            }
            if (config->phiRange.size() != 3) // need all three parameters
                return 1;
        }
        if (strcmp(argv[i],"--elevationRange") == 0)
        {
            std::stringstream str(argv[i+1]);
            while (str.good())
            {
                std::string substr;
                getline(str,substr,',');
                config->thetaRange.push_back(std::stod(substr));
            }
            if (config->thetaRange.size() != 3) // need all three parameters
                return 1;
        }
    }
    return 0;
}

int parseArgsLookupTable(lookupTableParams* config, int argc, char* argv[])
{
    // check the number of arguments
    if (argc != 13)
    {
        std::cerr << "raycastModel usage:"                                                          << std::endl <<
                     "./generateLookup "                                                            << std::endl <<
                     "--modelFileName        \"path_to_geometry\" "                                 << std::endl <<
                     "--outputFileName       \"path_to_file\" "                                     << std::endl <<
                     "--lookupToModel        \"roll(deg),pitch(deg),yaw(deg),x(m),y(m),z(m)\" "     << std::endl <<
                     "--maxXYZ               \"xMax(m),yMax(m),zMax(m)\" "                          << std::endl <<
                     "--stepSize             \"stepSize(m)\" "                                      << std::endl <<
                     "--sigma                \"standardDeviation(m)\" "                             << std::endl;
        return 1;
    }
    
    // parse the arguments
    for (unsigned int i = 0; i < argc; i++)
    {
        if (strcmp(argv[i],"--modelFileName") == 0)
        {
            config->modelFileName = argv[i+1];
        }
        if (strcmp(argv[i],"--outputFileName") == 0)
        {
            config->outputFileName = argv[i+1];        
        }
        if (strcmp(argv[i],"--lookupToModel") == 0)
        {
            std::stringstream str(argv[i+1]);
            std::vector<double> input;
            while (str.good())
            {
                std::string substr;
                getline(str,substr,',');
                input.push_back(std::stod(substr));
            }
            if (input.size() != 6) // need all six parameters
                return 1;
            config->lookupToModel = homogeneous(input[0]*M_PI/180,
                                                input[1]*M_PI/180,
                                                input[2]*M_PI/180,
                                                input[3],input[4],input[5]);
        }
        if (strcmp(argv[i],"--maxXYZ") == 0)
        {
            std::stringstream str(argv[i+1]);
            while (str.good())
            {
                std::string substr;
                getline(str,substr,',');
                config->maxXYZ.push_back(std::stod(substr));
            }
            if (config->maxXYZ.size() != 3) // need all six parameters
                return 1;  
        }
        if (strcmp(argv[i],"--stepSize") == 0)
        {
            config->stepSize = std::stod(argv[i+1]);
        }
        if (strcmp(argv[i],"--sigma") == 0)
        {
            config->sigma = std::stod(argv[i+1]);        
        }
    }
    return 0;
}

int parseArgsPlum(plumParams* config, int argc, char* argv[])
{
    if (argc != 3)
    {
        std::cerr << "plum usage:"                            << std::endl <<
                     "./plum --configFile \"configFilePath\"" << std::endl;
        return 1;
    }
    std::string configFileName = argv[2];
    std::ifstream configFile(configFileName);
    std::string newLine;
    std::vector<std::vector<std::string> > paramsAll;
    while (std::getline(configFile,newLine))
    {

    std::stringstream str(newLine);

    std::vector<std::string> paramsFromFile;
    while (str.good())
    {
        std::string substr;
        getline(str,substr,':');
        auto noSpaces = std::remove(substr.begin(), substr.end(), ' ');
        substr.erase(noSpaces,substr.end());

        // std::cout << substr << std::endl;
        paramsFromFile.push_back(substr);
    }
    paramsAll.push_back(paramsFromFile);
    }

    if (paramsAll.size() != 13) exit(1);

    // store the config values
    for (unsigned int i = 0; i < paramsAll.size(); i++)
    {
        std::string paramsName = paramsAll[i][0];

        if (paramsName.compare("pointCloudFile") == 0)
            config->pointCloudFile = paramsAll[i][1]; 

        if (paramsName.compare("LOOKUP_TABLE_lookupTableFile") == 0)
            config->lookupTableFile = paramsAll[i][1];

        if (paramsName.compare("LOOKUP_TABLE_lookupToModel") == 0)
        {
            std::stringstream str(paramsAll[i][1]);
            std::vector<double> input;
            while (str.good())
            {
                std::string substr;
                getline(str,substr,',');
                input.push_back(std::stod(substr));
            }
            if (input.size() != 6) // need all six parameters
                return 1;
            config->lookupToModel = homogeneous(input[0]*M_PI/180,
                                                input[1]*M_PI/180,
                                                input[2]*M_PI/180,
                                                input[3],input[4],input[5]);
        }

        if (paramsName.compare("LOOKUP_TABLE_maxXYZ") == 0)
        {
            std::stringstream str(paramsAll[i][1]);
            while (str.good())
            {
                std::string substr;
                getline(str,substr,',');
                config->maxXYZ.push_back(std::stod(substr));
            }
            if (config->maxXYZ.size() != 3) // need all six parameters
                return 1; 
        }

        if (paramsName.compare("LOOKUP_TABLE_stepSize") == 0)
            config->stepSize = std::stod(paramsAll[i][1]);

        if (paramsName.compare("SEARCH_HEURISTIC_seed") == 0)
        {
            std::stringstream str(paramsAll[i][1]);
            config->seed.resize(6);
            int counter = 0;
            while (str.good())
            {
                std::string substr;
                getline(str,substr,',');
                config->seed(counter) = std::stod(substr);
                counter++;
            }
            if (counter != 6) // need all six parameters
                return 1;

            config->seed(0) = config->seed(0) * M_PI/180;  
            config->seed(1) = config->seed(1) * M_PI/180;  
            config->seed(2) = config->seed(2) * M_PI/180;  
        }
        
        if (paramsName.compare("SEARCH_HEURISTIC_minDeviation") == 0)
        {
            std::stringstream str(paramsAll[i][1]);
            config->minDeviation.resize(6);
            int counter = 0;
            while (str.good())
            {
                std::string substr;
                getline(str,substr,',');
                config->minDeviation(counter) = std::stod(substr);
                counter++;
            }
            if (counter != 6) // need all six parameters
                return 1;  
            
            config->minDeviation(0) = config->minDeviation(0) * M_PI/180;  
            config->minDeviation(1) = config->minDeviation(1) * M_PI/180;  
            config->minDeviation(2) = config->minDeviation(2) * M_PI/180;
        }

        if (paramsName.compare("SEARCH_HEURISTIC_maxDeviation") == 0)
        {
            std::stringstream str(paramsAll[i][1]);
            config->maxDeviation.resize(6);
            int counter = 0; 
            while (str.good())
            {
                std::string substr;
                getline(str,substr,',');
                config->maxDeviation(counter) = std::stod(substr);
                counter++;
            }
            if (counter != 6) // need all six parameters
                return 1;

            config->maxDeviation(0) = config->maxDeviation(0) * M_PI/180;  
            config->maxDeviation(1) = config->maxDeviation(1) * M_PI/180;  
            config->maxDeviation(2) = config->maxDeviation(2) * M_PI/180; 
            
        }

        if (paramsName.compare("SEARCH_HEURISTIC_stepSizes") == 0)
        {
            std::stringstream str(paramsAll[i][1]);
            config->stepSizes.resize(6);
            int counter = 0; 
            while (str.good())
            {
                std::string substr;
                getline(str,substr,',');
                config->stepSizes(counter) = std::stod(substr);
                counter++;
            }
            if (counter != 6) // need all six parameters
                return 1;
            config->stepSizes(0) = config->stepSizes(0) * M_PI/180;  
            config->stepSizes(1) = config->stepSizes(1) * M_PI/180;  
            config->stepSizes(2) = config->stepSizes(2) * M_PI/180; 
        }

        if (paramsName.compare("SEARCH_HEURISTIC_rotSigma") == 0)
            config->rotSigma = std::stod(paramsAll[i][1]);

        if (paramsName.compare("SEARCH_HEURISTIC_transSigma") == 0)
            config->transSigma = std::stod(paramsAll[i][1]);
        
        if (paramsName.compare("SEARCH_HEURISTIC_noIterations") == 0)
            config->noIterations = std::stoi(paramsAll[i][1]);
        
        if (paramsName.compare("SEARCH_HEURISTIC_resampleSize") == 0)
            config->resampleSize = std::stoi(paramsAll[i][1]);
    }
    return 0;
}

void writeRaycastConfigFile(raycastParams* config)
{
    // write the config to file
    // get the time now
    auto end = std::chrono::system_clock::now();
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    std::vector<double> sensorToModelVals = hom2rpyxyz(config->sensorToModel);

    std::string outputConfigFileName = config->outputFileName+"_config.m";
    std::ofstream outputConfigFile(outputConfigFileName);
    outputConfigFile << "% computation finised at : " << std::ctime(&end_time)
                     << std::endl
                     << "modelFileName  = " << "\"" <<  config->modelFileName  << "\"" << ";" << std::endl
                     << "outputFileName = " << "\"" <<  config->outputFileName << "\"" << ";" << std::endl
                     << "sensorToModel  = " << "[" << sensorToModelVals[0]*180/M_PI    << ","
                                                   << sensorToModelVals[1]*180/M_PI    << ","
                                                   << sensorToModelVals[2]*180/M_PI    << ","
                                                   << sensorToModelVals[3]             << ","
                                                   << sensorToModelVals[4]             << ","
                                                   << sensorToModelVals[5]     << "];" << std::endl
                     << "headingRange   = " << "[" << config->phiRange[0]      << ","
                                                   << config->phiRange[1]      << ","
                                                   << config->phiRange[2]      << "];" << std::endl
                     << "elevationRange = " << "[" << config->thetaRange[0]    << ","
                                                   << config->thetaRange[1]    << ","
                                                   << config->thetaRange[2]    << "];" << std::endl;
    outputConfigFile.close();
}

void writeLookupConfigFile(lookupTableParams* config)
{
    // write the config to file
    // get the time now
    auto end = std::chrono::system_clock::now();
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    std::vector<double> lookupToModelVals = hom2rpyxyz(config->lookupToModel);

    std::string outputConfigFileName = config->outputFileName+"_config.m";
    std::ofstream outputConfigFile(outputConfigFileName);
    outputConfigFile << "% computation finised at : " << std::ctime(&end_time)
                     << std::endl
                     << "modelFileName  = " << "\"" <<  config->modelFileName  << "\"" << ";" << std::endl
                     << "outputFileName = " << "\"" <<  config->outputFileName << "\"" << ";" << std::endl
                     << "lookupToModel  = " << "[" << lookupToModelVals[0]*180/M_PI    << ","
                                                   << lookupToModelVals[1]*180/M_PI    << ","
                                                   << lookupToModelVals[2]*180/M_PI    << ","
                                                   << lookupToModelVals[3]             << ","
                                                   << lookupToModelVals[4]             << ","
                                                   << lookupToModelVals[5]     << "];" << std::endl
                     << "maxXYZ         = " << "[" << config->maxXYZ[0]        << ","
                                                   << config->maxXYZ[1]        << ","
                                                   << config->maxXYZ[2]        << "];" << std::endl
                     << "stepSize       = "        << config->stepSize         << ";"  << std::endl
                     << "sigma          = "        << config->sigma            << ";"  << std::endl;
    outputConfigFile.close();
}