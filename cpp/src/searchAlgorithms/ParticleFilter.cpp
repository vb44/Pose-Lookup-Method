#include "ParticleFilter.hpp"

ParticleFilter::ParticleFilter(const ConfigParser &config, std::shared_ptr<ObjectiveFunction> objFunc) :
    rotSigma_(config.getSearchRotSigma()),
    transSigma_(config.getSearchTransSigma()),
    noIterations_(config.getSearchNoIterations()),
    resampleSize_(config.getResampleSize()),
    seed_(config.getSearchSeed()),
    minDev_(config.getSearchMinDev()),
    maxDev_(config.getSearchMaxDev()),
    stepSizes_(config.getSearchStepSizes())
{
    objFunc_= std::move(objFunc);
    hypothesesSampled_.resize(resampleSize_+1, 6);
}

void ParticleFilter::generateHypotheses()
{
    double hyp[6];
    unsigned int hypIndex = 0;
    unsigned int numRoll, numPitch, numYaw, numX, numY, numZ;
    numRoll  = round((maxDev_[0] - minDev_[0]) / stepSizes_[0] + 1);
    numPitch = round((maxDev_[1] - minDev_[1]) / stepSizes_[1] + 1);
    numYaw   = round((maxDev_[2] - minDev_[2]) / stepSizes_[2] + 1);
    numX     = round((maxDev_[3] - minDev_[3]) / stepSizes_[3] + 1);
    numY     = round((maxDev_[4] - minDev_[4]) / stepSizes_[4] + 1);
    numZ     = round((maxDev_[5] - minDev_[5]) / stepSizes_[5] + 1);

    numberOfHypotheses_ = numRoll*numPitch*numYaw*numX*numY*numZ;
    hypotheses_.resize(numberOfHypotheses_, 6);

    for (unsigned int roll = 0; roll < numRoll; roll++) 
    {
        hyp[0] = seed_[0] + minDev_[0] + stepSizes_[0]*roll;
        for (unsigned int pitch = 0; pitch < numPitch; pitch++)
        {
            hyp[1] = seed_[1] + minDev_[1] + stepSizes_[1]*pitch;
            for (unsigned int yaw = 0; yaw < numYaw; yaw++)
            {
                hyp[2] = seed_[2] + minDev_[2] + stepSizes_[2]*yaw;
                for (unsigned int x = 0; x < numX; x++)
                {
                    hyp[3] = seed_[3] + minDev_[3] + stepSizes_[3]*x;
                    for (unsigned int y = 0; y < numY; y++)
                    {
                        hyp[4] = seed_[4] + minDev_[4] + stepSizes_[4]*y;
                        for (unsigned int z = 0; z < numZ; z++)
                        {
                            hyp[5] = seed_[5] + minDev_[5] + stepSizes_[5]*z;
                            
                            // save each hypothesis to the hypotheses list
                            hypotheses_.row(hypIndex) << hyp[0],hyp[1],hyp[2],
                                                         hyp[3],hyp[4],hyp[5];
                            hypIndex++;
                        }
                    }
                }
            }
        }
    }
}

std::vector<double> ParticleFilter::findBestGeometryPose()
{

    std::vector<int> evidences;
    generateHypotheses();
    int resampleSize = resampleSize_;
    
    for (unsigned int iteration = 1; iteration <= noIterations_; iteration++)
    {
        // Calculate the evidence for the hypotheses set.
        evidences = objFunc_->calculateEvidence(hypotheses_);

        // if (iteration == noIterations_)
        // {
        //     continue;
        // }

        // Set the evidence for hypotheses out of the search range to 0. TODO: Check this?
        tbb::parallel_for(
            tbb::blocked_range<int>(0, hypotheses_.rows()),
            [&](tbb::blocked_range<int> r)
            {
                for (int kk = r.begin(); kk < r.end(); kk++)
                {
                    hypotheses_(kk,0) = std::min(std::max(seed_[0]+minDev_[0], hypotheses_(kk,0)), seed_[0]+maxDev_[0]);
                    hypotheses_(kk,1) = std::min(std::max(seed_[1]+minDev_[1], hypotheses_(kk,1)), seed_[1]+maxDev_[1]);
                    hypotheses_(kk,2) = std::min(std::max(seed_[2]+minDev_[2], hypotheses_(kk,2)), seed_[2]+maxDev_[2]);
                    hypotheses_(kk,3) = std::min(std::max(seed_[3]+minDev_[3], hypotheses_(kk,3)), seed_[3]+maxDev_[3]);
                    hypotheses_(kk,4) = std::min(std::max(seed_[4]+minDev_[4], hypotheses_(kk,4)), seed_[4]+maxDev_[4]);
                    hypotheses_(kk,5) = std::min(std::max(seed_[5]+minDev_[5], hypotheses_(kk,5)), seed_[5]+maxDev_[5]);
                }
            }
        );
        
        // Begin the search heuristic.
        double normConstant = 0.0;
        for (unsigned int i = 0; i < numberOfHypotheses_; i++)
        {
                normConstant += pow(evidences[i], iteration);
        }

        if (isinf(normConstant))
        {
            std::cout << "pose_estimator: Normalising constant is infinite!" << std::endl;
            exit(1);
        }

        double total = 0;
        std::vector<double> hypProb(numberOfHypotheses_); // hypothesis probability
        std::vector<double> cumProb(numberOfHypotheses_); // cumulative probability
        for (unsigned int i = 0; i < numberOfHypotheses_; i++)
        {
            hypProb[i] = (pow(evidences[i], iteration) /
                            normConstant);
            total += hypProb[i];
            cumProb[i] = total;
        }

        // Random number generation.
        // Set the seed for reproducible results.
        if (randRot_) { delete randRot_; }
        if (randTrans_) { delete randTrans_; }
        static boost::mt19937 seed(0);

        // Scale the particle noise down per iteration to allow for a
        // focused search.
        double scale = 1.0 / ((double)iteration);

        boost::normal_distribution<double> rotationalDistribution(
                                            0.0, rotSigma_ * scale);
        boost::normal_distribution<double> translationalDistribution(
                                            0.0, transSigma_ * scale);
        randRot_ = new boost::variate_generator<boost::mt19937&, boost::normal_distribution<double> >(seed, rotationalDistribution);
        randTrans_ = new boost::variate_generator<boost::mt19937&, boost::normal_distribution<double> >(seed, translationalDistribution);

        // Hard reset the random number generators.
        randRot_->engine().seed(0);
        randRot_->distribution().reset();
        randTrans_->engine().seed(0);
        randTrans_->distribution().reset();

        std::vector<double> noiseVector(6, 0.0);

        // Overwrite the highest rewarding element.
        int maxElement = *std::max_element(&evidences[0],
                                           &evidences[0]+numberOfHypotheses_);
        int maxElementIndex = std::find(&evidences[0],
                                        &evidences[0]+numberOfHypotheses_,
                                        maxElement) - &evidences[0];

        // Resample the hypotheses using the cumulative distribution
        // and the noise generators.
        int i, j, k;
        for (i = 0; i < resampleSize; i++)
        {
            double pxSampleLevel = (i+0.5) * (1.0 / (double)(resampleSize));

            for (j = 0; j < (numberOfHypotheses_-1); j++)
            {
                if (cumProb[j] >= pxSampleLevel) break;
            }

            // Add noise to the hypothesis.
            noiseVector[0] = (*randRot_)();
            noiseVector[1] = (*randRot_)();
            noiseVector[2] = (*randRot_)();
            noiseVector[3] = (*randTrans_)();
            noiseVector[4] = (*randTrans_)();
            noiseVector[5] = (*randTrans_)();
            for (k = 0; k < 6; k++)
            {
                hypothesesSampled_(i,k) = hypotheses_(j,k) + noiseVector[k];
            }
        }

        if (iteration == 1)
        {
            resampleSize++;
            numberOfHypotheses_ = resampleSize;
        }

        // Save the best evidence.
        hypothesesSampled_.row(resampleSize-1) = hypotheses_.row(maxElementIndex);

        if (iteration != noIterations_)
        {
            hypotheses_.resize(resampleSize,6);
            hypotheses_ = hypothesesSampled_;
            numberOfHypotheses_ = resampleSize;
        }
    }

    int maxElement = *std::max_element(&evidences[0],
                                       &evidences[0]+numberOfHypotheses_);
    int maxElementIndex = std::find(&evidences[0],
                                    &evidences[0]+numberOfHypotheses_,
                                    maxElement) - &evidences[0];

    return {hypotheses_(maxElementIndex, 0),
            hypotheses_(maxElementIndex, 1),
            hypotheses_(maxElementIndex, 2),
            hypotheses_(maxElementIndex, 3),
            hypotheses_(maxElementIndex, 4),
            hypotheses_(maxElementIndex, 5),
            double(evidences[maxElementIndex])};
}