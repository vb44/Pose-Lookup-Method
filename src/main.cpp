#include "helper.h"
#include <boost/random.hpp>
#include <boost/math/distributions/normal.hpp>

int main()
{
  // --------------------------------------------------------------------------
  // USER CONFIGURATION
  // --------------------------------------------------------------------------
  // point cloud file
  std::string pointCloudFileName = "../outputs/bunny_pointcloud";

  // search heuristic setup
  Eigen::RowVectorXd seed(6);
  Eigen::RowVectorXd minDeviation(6);
  Eigen::RowVectorXd maxDeviation(6);
  Eigen::RowVectorXd stepSizes(6);
  seed << 0,0,20*M_PI/180,25,15,-10; // SENSOR TO MODEL HYPOTHESIS
  minDeviation  << -M_PI, -M_PI, -M_PI, -10, -10, -10; 
  maxDeviation  << M_PI,  M_PI,  M_PI, 10, 10, 10; 
  stepSizes  << 1.0472, 1.0472, 1.0472, 2, 2, 2; 
  double rotationalSigma = 0.05;
  double translationalSigma = 3;
  unsigned int noOfIterations = 40;
  unsigned int resampleSize = 1000;

  // lookup to model
  // std::string lookupTableFileName = "inputs/bunny_lookup_table_sigma_1.lookup";
  std::string lookupTableFileName = "../outputs/bunny_lookup_table_sigma_1m_stepSize_1mm.lookup";
  Eigen::Matrix4d lookupToModel = homogeneous(0,0,0,50,40,10);
  double stepSize = 1;
  double maxXYZ[] = {100, 100, 100}; 
  
  // ------------------------------------------------------------------------
  // READ THE POINT CLOUD
  // ------------------------------------------------------------------------
  Eigen::MatrixXd ptCloud;
  ptCloud = readPointCloud(pointCloudFileName);

  // ------------------------------------------------------------------------
  // GENERATE THE INITIAL HYPOTHESIS SET
  // ------------------------------------------------------------------------
  unsigned int numberOfHypotheses  = round((maxDeviation[0] - minDeviation[0]) / stepSizes[0] + 1)*
                                     round((maxDeviation[1] - minDeviation[1]) / stepSizes[1] + 1)* 
                                     round((maxDeviation[2] - minDeviation[2]) / stepSizes[2] + 1)*
                                     round((maxDeviation[3] - minDeviation[3]) / stepSizes[3] + 1)*
                                     round((maxDeviation[4] - minDeviation[4]) / stepSizes[4] + 1)*
                                     round((maxDeviation[5] - minDeviation[5]) / stepSizes[5] + 1);
  Eigen::MatrixXd hypotheses = generateHypotheses(seed, minDeviation, maxDeviation,stepSizes);

  // ------------------------------------------------------------------------
  // READ THE LOOKUP TABLE
  // ------------------------------------------------------------------------
  uint8_t* lookupTable;
  double pointsPerMeter = 1.0/stepSize;
  unsigned int numXYZ[3];
  numXYZ[0] = round(maxXYZ[0]* pointsPerMeter + 1);
  numXYZ[1] = round(maxXYZ[1]* pointsPerMeter + 1);
  numXYZ[2] = round(maxXYZ[2]* pointsPerMeter + 1);
  lookupTable = (uint8_t*) malloc(numXYZ[0]*numXYZ[1]*numXYZ[2]*sizeof(uint8_t));
  readLookupTable(lookupTable, numXYZ, lookupTableFileName);

  // ------------------------------------------------------------------------
  // SEARCH HEURISTIC SETUP
  // ------------------------------------------------------------------------
  Eigen::MatrixXd hypothesesSampled(resampleSize+1,6);
  boost::variate_generator<boost::mt19937&, boost::normal_distribution<double> >* randRotation = 0;
  boost::variate_generator<boost::mt19937&, boost::normal_distribution<double> >* randTranslation = 0;
  std::vector<double> evidence;

  // ------------------------------------------------------------------------
  // START TIMING FROM HERE
  // ------------------------------------------------------------------------
  auto start = std::chrono::high_resolution_clock::now(); 

  // evaluate hypotheses
  for (unsigned int iteration = 1; iteration <= noOfIterations; iteration++)
  // for (unsigned int iteration = 1; iteration <= 10; iteration++)
  {
    // calculate the evidence for the hypotheses
    evidence = calculateEvidence(lookupTable, lookupToModel,
                                 hypotheses, ptCloud,
                                 maxXYZ, numXYZ, pointsPerMeter);
    
    // search heuristic
    double normalisingConstant = 0.0;
    for (unsigned int i = 0; i < numberOfHypotheses; i++) {
          normalisingConstant += pow(evidence[i], iteration);
    }

    if (isinf(normalisingConstant)) {
          std::cout << "Normalising constant has reached infinity!" << std::endl;
          exit(1);
    }

    double total = 0;
    std::vector<double> exageratedHypothesisProb(numberOfHypotheses); 
    std::vector<double> cumProb(numberOfHypotheses); 
    for (unsigned int i = 0; i < numberOfHypotheses; i++)
    {
        exageratedHypothesisProb[i] = (pow(evidence[i], iteration) / normalisingConstant);
        total += exageratedHypothesisProb[i];
        cumProb[i] = total;
    }

    // random number generation--------------------------------
    if (randRotation) { delete randRotation; }
    if (randTranslation) { delete randTranslation; }

    // set the seed for reproducible results
    static boost::mt19937 seed(0);

    // scale the noise down per iteration.
    double scale = 1.0 / ( (double)iteration);

    boost::normal_distribution<double> rotationalDistribution(0.0, rotationalSigma * scale);
    boost::normal_distribution<double> translationalDistribution(0.0, translationalSigma * scale);

    randRotation = new boost::variate_generator<boost::mt19937&, boost::normal_distribution<double> >(seed, rotationalDistribution);
    randTranslation = new boost::variate_generator<boost::mt19937&, boost::normal_distribution<double> >(seed, translationalDistribution);

    // hard reset the random number generators
    randRotation->engine().seed(0);
    randRotation->distribution().reset();
    randTranslation->engine().seed(0);
    randTranslation->distribution().reset();
    
    std::vector<double> noiseVector(6, 0.0);

    // overwrite the best element index
    int maxElement = *std::max_element(&evidence[0], &evidence[0]+numberOfHypotheses);
    int maxElementIndex = std::find(&evidence[0], &evidence[0]+numberOfHypotheses, maxElement) - &evidence[0]; 

    // i (hypothesis index), j (where in the previous set we are perturbing from), k (used to add noise)
    unsigned int i, j, k;
    for (i = 0; i < resampleSize; i++) {

        // uniformly sample
        double pxSampleLevel = (i+0.5) * (1.0 / (double)(resampleSize));

        // find the first hypothesis where the cumProb (cumulative probability) is above the sample level
        for (j = 0; j < (numberOfHypotheses-1); j++) {
            if (cumProb[j] >= pxSampleLevel) {
                break;
            }
        }   

        // add some noise to hypothesis j and put it in the list
        noiseVector[0] = (*randRotation)();
        noiseVector[1] = (*randRotation)();
        noiseVector[2] = (*randRotation)();
        noiseVector[3] = (*randTranslation)();
        noiseVector[4] = (*randTranslation)();
        noiseVector[5] = (*randTranslation)();
        for (k = 0; k < 6; k++)
        {
            hypothesesSampled(i,k) = hypotheses(j,k) + noiseVector[k];
        }
    }

    if (iteration == 1)
    {
      resampleSize++;
      numberOfHypotheses = resampleSize;
    }

    // save the best evidence
      hypothesesSampled.row(resampleSize-1) = hypotheses.row(maxElementIndex);

    if (iteration != noOfIterations)
    {
      hypotheses.resize(resampleSize,6);
      hypotheses = hypothesesSampled;
      numberOfHypotheses = resampleSize;
    } 
  }
  auto stop = std::chrono::high_resolution_clock::now();
                  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop-start);
                  std::cout << "Time taken by entire function: " << duration.count() << " milliseconds" << std::endl;


  // print the best hypothesis ----------------------------------------------
  int maxElement = *std::max_element(&evidence[0], &evidence[0]+numberOfHypotheses);
  int maxElementIndex = std::find(&evidence[0], &evidence[0]+numberOfHypotheses, maxElement) - &evidence[0];

  std::cout << "roll(rad), pitch(rad), yaw(rad), x(m), y(m), z(m), evidence" << std::endl;
  std::cout << hypotheses(maxElementIndex, 0) << "," 
            << hypotheses(maxElementIndex, 1) << "," 
            << hypotheses(maxElementIndex, 2) << "," 
            << hypotheses(maxElementIndex, 3) << "," 
            << hypotheses(maxElementIndex, 4) << "," 
            << hypotheses(maxElementIndex, 5) << "," 
            << evidence[maxElementIndex] <<std::endl;

  return 0;
}