#include "ConfigParser.hpp"

ConfigParser::ConfigParser(int argc, char** argv)
{
    if (argc != EXPECTED_ARGUMENT_COUNT)
    {
        std::cerr << "Usage: ./plum config_file.yaml" << std::endl;
        exit(EXIT_FAILURE);
    }
    yamlFilePath_ = argv[1];
}

int ConfigParser::parseConfig()
{
    try {
        YAML::Node configFromYaml = YAML::LoadFile(yamlFilePath_);

        // Check the pose estimation method being used.
        poseEstMethod_ = configFromYaml["method"].as<std::string>();

        if (!poseEstMethod_.compare("plum"))
        {
            // Lookup table
            lookupTableStepSize_ = configFromYaml["lookupTableStepSize"].as<double>();
            lookupTableFile_ = configFromYaml["lookupTableFile"].as<std::string>();
            lookupTableToModel_ = configFromYaml["lookupTableToModel"].as<std::vector<double>>();
            lookupTableMaxXyz_ = configFromYaml["lookupTableMaxXyz"].as<std::vector<double>>();
        } else if (!poseEstMethod_.compare("msoe"))
        {
            sigma_ = configFromYaml["sigma"].as<double>();
            modelFilePath_ = configFromYaml["modelFilePath"].as<std::string>();
        } else
        {
            std::cerr << "./plum: The parmater \"method\" must be \"plum\" or \"msoe\"" << std::endl;
            exit(EXIT_FAILURE);
        }
        // Point cloud
        pointCloudPath_= configFromYaml["pointCloudFolder"].as<std::string>();
        sesnorMinRange_ = configFromYaml["sensorMinRange"].as<double>();
        sensorMaxRange_ = configFromYaml["sensorMaxRange"].as<double>();
        pcSubsampleRadius_ = configFromYaml["subsampleRadius"].as<double>();
        platformToSensor_ = configFromYaml["platformToSensor"].as<std::vector<double>>();
        pcRegionOfInterest_ = configFromYaml["pcRegionOfInterest"].as<std::vector<double>>();

        // Search heuristic performance
        searchRotSigma_ = configFromYaml["searchRotSigma"].as<double>();
        searchTransSigma_ = configFromYaml["searchTransSigma"].as<double>();
        searchNoIterations_ = configFromYaml["searchNoIterations"].as<double>();
        searchResampleSize_ = configFromYaml["searchResampleSize"].as<double>();
        searchSeed_ = configFromYaml["searchSeed"].as<std::vector<double>>();
        searchMinDev_ = configFromYaml["searchMinDev"].as<std::vector<double>>();
        searchMaxDev_ = configFromYaml["searchMaxDev"].as<std::vector<double>>();
        searchStepSizes_ = configFromYaml["searchStepSizes"].as<std::vector<double>>();

    } catch(const YAML::BadFile& e)
    {
        std::cerr << e.what() << std::endl;
        return 1;
    } catch(const YAML::ParserException& e)
    {
        std::cerr << e.msg << std::endl;
        return 1;
    }
    return 0;
}

const std::string ConfigParser::getPcPath() const
{
    return pointCloudPath_;
}

const double ConfigParser::getMaxSensorRange() const
{
    return sensorMaxRange_;
}

const double ConfigParser::getMinSensorRange() const
{
    return sesnorMinRange_;
}

const double ConfigParser::getPcSubsampleRadius() const
{
    return pcSubsampleRadius_;
}

const double ConfigParser::getLookupTableStepSize() const
{
    return lookupTableStepSize_;
}

const std::string ConfigParser::getLookupTableFile() const
{
    return lookupTableFile_;   
}

const std::vector<double> ConfigParser::getLookupTableToModel() const
{
    return lookupTableToModel_;
}

const std::vector<double> ConfigParser::getLookupTableMaxXyz() const
{
    return lookupTableMaxXyz_;
}

const double ConfigParser::getSearchRotSigma() const
{
    return searchRotSigma_;
}

const double ConfigParser::getSearchTransSigma() const
{
    return searchTransSigma_;
}

const double ConfigParser::getSearchNoIterations() const
{
    return searchNoIterations_;
}

const double ConfigParser::getResampleSize() const
{
    return searchResampleSize_;
}

const std::vector<double> ConfigParser::getSearchSeed() const
{
    return searchSeed_;
}

const std::vector<double> ConfigParser::getSearchMinDev() const
{
    return searchMinDev_; 
}

const std::vector<double> ConfigParser::getSearchMaxDev() const
{
    return searchMaxDev_;
}

const std::vector<double> ConfigParser::getSearchStepSizes() const
{
    return searchStepSizes_;
}

const double ConfigParser::getSigma() const
{
    return sigma_;
}

const std::string ConfigParser::getModelFilePath() const
{
    return modelFilePath_;
}

const std::string ConfigParser::getPoseEstMethod() const
{
    return poseEstMethod_;
}

const std::vector<double> ConfigParser::getPlatformToSensor() const
{
    return platformToSensor_;
}

const std::vector<double> ConfigParser::getPcRegionOfInterest() const
{
    return pcRegionOfInterest_;
}