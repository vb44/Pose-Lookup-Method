#include "ConfigParserLookup.hpp"

ConfigParserLookup::ConfigParserLookup(int argc, char** argv)
{
    if (argc != EXPECTED_ARGUMENT_COUNT)
    {
        std::cerr << "Usage: ./generateLookupTable config_file.yaml" << std::endl;
        exit(EXIT_FAILURE);
    }
    yamlFilePath_ = argv[1];
}

ConfigParserLookup::ConfigParserLookup(std::string yamlFilePath)
    : yamlFilePath_(yamlFilePath)
{
}

int ConfigParserLookup::parseConfig()
{
    try {
        YAML::Node configFromYaml = YAML::LoadFile(yamlFilePath_);

        // Lookup table
        stepSize_ = configFromYaml["stepSize"].as<double>();
        sigma_ = configFromYaml["sigma"].as<double>();
        modelFilePath_ = configFromYaml["modelFileName"].as<std::string>();
        outputFileName_ = configFromYaml["outputFileName"].as<std::string>();
        lookupTableToModel_ = configFromYaml["lookupTableToModel"].as<std::vector<double>>();
        maxXyz_ = configFromYaml["maxBoundsXyz"].as<std::vector<double>>();

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

const double ConfigParserLookup::getStepSize() const
{
    return stepSize_;
}

const double ConfigParserLookup::getSigma() const
{
    return sigma_;
}

const std::string ConfigParserLookup::getModelFilePath() const
{
    return modelFilePath_;   
}

const std::string ConfigParserLookup::getOutputFileName() const
{
    return outputFileName_;   
}

const std::vector<double> ConfigParserLookup::getLookupTableToModel() const
{
    return lookupTableToModel_;
}

const std::vector<double> ConfigParserLookup::getMaxBounds() const
{
    return maxXyz_;
}

void ConfigParserLookup::writeConfig()
{
    YAML::Node config;
    
    config["stepSize"] = stepSize_;
    config["sigma"] = sigma_;
    config["modelFileName"] = modelFilePath_;
    config["outputFileName"] = outputFileName_;
    config["lookupTableToModel"] = lookupTableToModel_;
    config["lookupTableToModel"].SetStyle(YAML::EmitterStyle::Flow);
    config["maxBoundsXyz"] = maxXyz_;
    config["maxBoundsXyz"].SetStyle(YAML::EmitterStyle::Flow);

    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::string timestamp = std::ctime(&now_c);
    timestamp.pop_back();
    config["fileGeneratedAt"] = timestamp;

    std::ofstream fout(outputFileName_+".yaml");
    fout << config;
    fout.close();
}