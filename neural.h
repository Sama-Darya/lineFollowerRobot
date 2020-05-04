#pragma once
#include <vector>


namespace cv {
class Mat;
}
/**
 * This function initialised the neural network, it sets up the filter bank (FB) for optimal correlation of the
 * closed-loop error and the predictors internally.
 * @param numInputLayers number of hidden layers in the network
 * @param sampleRate sampling frequency fo the program, it is used for setting up filters
 */
void initialize_samanet(int numInputLayers, double sampleRate = 30.f);

/**
 * This function performs one iteration of learning and plots relevant information on the Stat Frame.
 * @param statFrame The frame where the data is plotted
 * @param in The input data, this is a pointer to the array of predictors obtained form the camera view frame
 * @param error This is the closed-loop error
 * @return returns the overall output of the network, this is the differential speed that is later sent to the motors
 */
double run_samanet(cv::Mat &statFrame, std::vector<double> &in, double error);
