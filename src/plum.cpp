/*
 * PLUM.CPP
 * This file implements the PLuM algorithm published at: https://doi.org/10.3390/s23063085
 * The purpose of this file is to provide a simple to understand implementation
 * of the algorithm - the aim is to keep it readable!
 * The GPU results reported in the paper are performed with a slightly
 * different algorithm using different variable types etc. to suit GPU
 * operations. Please contact v.bhandari@uq.edu.au for more details
 * about the GPU version.
*/

#include "helper.h"

int main(int argc, char* argv[])
{
  // --------------------------------------------------------------------------
  // USER CONFIGURATION
  // --------------------------------------------------------------------------
  // Argument parsing.
  plumParams params;
  int err = parseArgsPlum(&params, argc, argv);
  if (err) exit(1);
  
  // ------------------------------------------------------------------------
  // READ THE POINT CLOUD
  // ------------------------------------------------------------------------
  Eigen::MatrixXd ptCloud = readPointCloud(params.pointCloudFile);
  
  // ------------------------------------------------------------------------
  // GENERATE THE INITIAL HYPOTHESIS SET
  // ------------------------------------------------------------------------
  unsigned int numberOfHypotheses  = round((params.maxDeviation[0] - params.minDeviation[0]) / params.stepSizes[0] + 1) *
                                     round((params.maxDeviation[1] - params.minDeviation[1]) / params.stepSizes[1] + 1) * 
                                     round((params.maxDeviation[2] - params.minDeviation[2]) / params.stepSizes[2] + 1) *
                                     round((params.maxDeviation[3] - params.minDeviation[3]) / params.stepSizes[3] + 1) *
                                     round((params.maxDeviation[4] - params.minDeviation[4]) / params.stepSizes[4] + 1) *
                                     round((params.maxDeviation[5] - params.minDeviation[5]) / params.stepSizes[5] + 1);
  Eigen::MatrixXd hypotheses = generateHypotheses(&params); 
  
  // ------------------------------------------------------------------------
  // READ THE LOOKUP TABLE
  // ------------------------------------------------------------------------
  uint8_t* lookupTable;
  double pointsPerMeter = 1.0/params.stepSize;
  unsigned int numXYZ[3];
  numXYZ[0] = round(params.maxXYZ[0]* pointsPerMeter + 1);
  numXYZ[1] = round(params.maxXYZ[1]* pointsPerMeter + 1);
  numXYZ[2] = round(params.maxXYZ[2]* pointsPerMeter + 1);
  lookupTable = (uint8_t*) malloc(numXYZ[0]*numXYZ[1]*numXYZ[2]*sizeof(uint8_t));
  readLookupTable(lookupTable, numXYZ, params.lookupTableFile);
  
  // ------------------------------------------------------------------------
  // SEARCH HEURISTIC SETUP: see Algorithm 1 from https://doi.org/10.3390/s21196473
  // ------------------------------------------------------------------------
  Eigen::MatrixXd hypothesesSampled(params.resampleSize+1,6);
  boost::variate_generator<boost::mt19937&, boost::normal_distribution<double> >* randRot = 0;
  boost::variate_generator<boost::mt19937&, boost::normal_distribution<double> >* randTrans = 0;
  std::vector<double> evidence;
  
  // ------------------------------------------------------------------------
  // START TIMING FROM HERE
  // ------------------------------------------------------------------------
  auto start = std::chrono::high_resolution_clock::now(); 

  // Evaluate hypotheses.
  for (unsigned int iteration = 1; iteration <= params.noIterations; iteration++)
  {
    // Calculate the evidence for the hypotheses set.
    evidence = calculateEvidence(lookupTable, params.lookupToModel,
                                 hypotheses, ptCloud,
                                 params.maxXYZ, numXYZ, pointsPerMeter);
    
    // Begin the search heuristic.
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

    // Random number generation.
    if (randRot) { delete randRot; }
    if (randTrans) { delete randTrans; }

    // Set the seed for reproducible results.
    static boost::mt19937 seed(0);

    // Scale the noise down per iteration.
    double scale = 1.0 / ((double)iteration);

    boost::normal_distribution<double> rotationalDistribution(0.0, params.rotSigma * scale);
    boost::normal_distribution<double> translationalDistribution(0.0, params.transSigma * scale);

    randRot = new boost::variate_generator<boost::mt19937&, boost::normal_distribution<double> >(seed, rotationalDistribution);
    randTrans = new boost::variate_generator<boost::mt19937&, boost::normal_distribution<double> >(seed, translationalDistribution);

    // Hard reset the random number generators.
    randRot->engine().seed(0);
    randRot->distribution().reset();
    randTrans->engine().seed(0);
    randTrans->distribution().reset();
    
    std::vector<double> noiseVector(6, 0.0);

    // Overwrite the best element index.
    int maxElement = *std::max_element(&evidence[0], &evidence[0]+numberOfHypotheses);
    int maxElementIndex = std::find(&evidence[0], &evidence[0]+numberOfHypotheses, maxElement) - &evidence[0]; 

    // i (hypothesis index), j (where in the previous set we are perturbing from), k (used to add noise)
    unsigned int i, j, k;
    for (i = 0; i < params.resampleSize; i++) {

        // Uniformly sample.
        double pxSampleLevel = (i+0.5) * (1.0 / (double)(params.resampleSize));

        // Find the first hypothesis where the cumProb (cumulative probability) is above the sample level.
        for (j = 0; j < (numberOfHypotheses-1); j++)
            if (cumProb[j] >= pxSampleLevel) break;

        // Add some noise to hypothesis j and put it in the list.
        noiseVector[0] = (*randRot)();
        noiseVector[1] = (*randRot)();
        noiseVector[2] = (*randRot)();
        noiseVector[3] = (*randTrans)();
        noiseVector[4] = (*randTrans)();
        noiseVector[5] = (*randTrans)();
        for (k = 0; k < 6; k++)
            hypothesesSampled(i,k) = hypotheses(j,k) + noiseVector[k];
    }

    if (iteration == 1)
    {
      params.resampleSize++;
      numberOfHypotheses = params.resampleSize;
    }

    // Save the best evidence.
      hypothesesSampled.row(params.resampleSize-1) = hypotheses.row(maxElementIndex);

    if (iteration != params.noIterations)
    {
      hypotheses.resize(params.resampleSize,6);
      hypotheses = hypothesesSampled;
      numberOfHypotheses = params.resampleSize;
    } 
  }
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop-start);
  std::cout << "Registration time: " << duration.count() << " milliseconds" << std::endl;

  // --------------------------------------------------------------------------
  // PRINT THE BEST HYPOTHESIS
  // --------------------------------------------------------------------------
  int maxElement = *std::max_element(&evidence[0], &evidence[0]+numberOfHypotheses);
  int maxElementIndex = std::find(&evidence[0], &evidence[0]+numberOfHypotheses, maxElement) - &evidence[0];

  std::cout << "roll(deg), pitch(deg), yaw(deg), x(m), y(m), z(m), evidence" << std::endl;
  std::cout << hypotheses(maxElementIndex, 0)*180/M_PI << "," 
            << hypotheses(maxElementIndex, 1)*180/M_PI << "," 
            << hypotheses(maxElementIndex, 2)*180/M_PI << "," 
            << hypotheses(maxElementIndex, 3)          << "," 
            << hypotheses(maxElementIndex, 4)          << "," 
            << hypotheses(maxElementIndex, 5)          << "," 
            << evidence[maxElementIndex]               << std::endl;
  
  return 0;
}