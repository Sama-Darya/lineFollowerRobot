#include "neural.h"
#include "cldl/Net.h"
#include "cvui.h"
#include "bandpass.h"

#include <chrono>
#include <fstream>
#include <initializer_list>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <numeric>
#include <boost/circular_buffer.hpp>
#include <math.h>
#include <ctgmath>




using namespace std;

std::vector<std::array<Bandpass, 5>> bandpassFilters;
const int numPred = 48;
boost::circular_buffer<double> predVector1[numPred];
boost::circular_buffer<double> predVector2[numPred];
boost::circular_buffer<double> predVector3[numPred];
boost::circular_buffer<double> predVector4[numPred];
boost::circular_buffer<double> predVector5[numPred];

static void initialize_filters(int numInputs, double sampleRate) {

  int nPred = numInputs / 5;
  int delayFactor[8] = {17,16,15,14,13,12,11,10};
  for (int i = 0; i < nPred; i++){
    int j= (int)(i / 6);
    predVector1[i].rresize(1); //delayFactor[j]-5);
    predVector2[i].rresize(2); //delayFactor[j]-3);
    predVector3[i].rresize(3); //delayFactor[j]+0);
    predVector4[i].rresize(4); //delayFactor[j]+3);
    predVector5[i].rresize(5); //delayFactor[j]+5);
  }

  bandpassFilters.resize(numInputs);
  double fs = 1;
  int minT = 100;
  int maxT = 150;
  double fmin = fs / maxT;
  double fmax = fs / minT;
  double df = (fmax - fmin) / 4.0; // 4 is number of filters minus 1
  for (auto &bank : bandpassFilters) {
    double f = fmin;
    for (auto &filt : bank) {
      filt.setParameters(f, 0.51);
      f += df;

      for(int k=0;k<maxT;k++){
        double a = 0;
        if (k==minT){
          a = 1;
        }
        double b = filt.filter(a);
        assert(b != NAN);
        assert(b != INFINITY);
      }
      filt.reset();

    }
  }
}

std::unique_ptr<Net> samanet;
const int numLayers = 11;

void initialize_samanet(int numInputLayers, double sampleRate) {
  numInputLayers *= 5; // 5 is the number of filters
  int numNeurons[numLayers]= {};
  int firstLayer = 11;
  int lastHiddenLayer = 4;
  int incrementLayer = 1;
  int decrementLayer = 1;
  //int fibonacci1 = 1;
  //int fibonacci2 = 2;
  int totalNeurons = 0;
  numNeurons[numLayers - 1] = 3;
  for (int i = numLayers - 2; i >= 0; i--){
    //int addition = fibonacci2 + fibonacci1;
    numNeurons[i] = lastHiddenLayer + (numLayers - 2 - i)  * incrementLayer;
    totalNeurons += numNeurons[i];
    //fibonacci1 = fibonacci2; // 3 , 5
    //fibonacci2 = addition; // 5, 8
    assert(numNeurons[i] > 0);
  }
  //cout << numNeurons[0] << " neurons in first layer, total of: " << totalNeurons << endl;
  //numNeurons[numLayers - 1] = 3; //output layer

  samanet = std::make_unique<Net>(numLayers, numNeurons, numInputLayers);
  samanet->initNetwork(Neuron::W_RANDOM, Neuron::B_NONE, Neuron::Act_Sigmoid);
  double myLearningRate = exp(-1);
  cout << "myLearningRate: " << myLearningRate << endl;
  samanet->setLearningRate(myLearningRate);
  initialize_filters(numInputLayers, sampleRate);
}

std::ofstream weightDistancesfs("weight_distances.csv");
std::ofstream predictor("predictor.csv");

bool firstInputs = 1;

double run_samanet(cv::Mat &statFrame, std::vector<double> &predictorDeltas, double error) {

  using namespace std::chrono;
  milliseconds ms = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
  std::vector<double> networkInputs;

  predictor << ms.count();
  networkInputs.reserve(predictorDeltas.size() * 5);

  for (int i =0; i < predictorDeltas.size(); i++){
    predictor << " " << error;
    double sampleValue = predictorDeltas[i];
    predictor << " " << sampleValue;
    predVector1[i].push_back(sampleValue);
    predVector2[i].push_back(sampleValue);
    predVector3[i].push_back(sampleValue);
    predVector4[i].push_back(sampleValue);
    predVector5[i].push_back(sampleValue);

    networkInputs.push_back(predVector1[i][0]);
    networkInputs.push_back(predVector2[i][0]);
    networkInputs.push_back(predVector3[i][0]);
    networkInputs.push_back(predVector4[i][0]);
    networkInputs.push_back(predVector5[i][0]);

    predictor << " " << predVector1[i][0];
    predictor << " " << predVector2[i][0];
    predictor << " " << predVector3[i][0];
    predictor << " " << predVector4[i][0];
    predictor << " " << predVector5[i][0];
  }

/*
  for (int j = 0; j < predictorDeltas.size(); ++j) {
    predictor << " " << error;
    double sample = predictorDeltas[j];
    predictor << " " << sample;
    for (auto &filt : bandpassFilters[j]) {
      auto filtered = filt.filter(sample);
      networkInputs.push_back(filtered);
      predictor << " " << filtered;
    }
  } */

  predictor << "\n" ;
  if (firstInputs == 1){
    samanet->setInputs(networkInputs.data());
    samanet->propInputs();
    firstInputs = 0;
    //cout << "DONE THIS" << endl;
  }
  // cout << "neural: error: " << error << endl;
  assert(std::isfinite(error));
  samanet->setErrorCoeff(0,0,0,0,1,0);
  samanet->setGlobalError(error);
  samanet->setLocalError(error);
  samanet->propGlobalErrorBackwardLocally();
  samanet->updateWeights(); // Learn from previous action
  samanet->setInputs(networkInputs.data()); //then take a new action
  samanet->propInputs();
  samanet->snapWeights();
  double compensationScale = 1;
  for (int i = 0; i <numLayers; i++){
    if (i == 0){
      compensationScale = 0.01;
    }
    weightDistancesfs << compensationScale * samanet->getLayerWeightDistance(i) << " ";
    compensationScale = 1;
  }
  weightDistancesfs << 0.01 * samanet->getWeightDistance() << "\n";

  double coeff[4] = {1,3,5};
  double outSmall = samanet->getOutput(0);
  double outMedium = samanet->getOutput(1);
  double outLarge = samanet->getOutput(2);

  double resultNN = (coeff[0] * outSmall) + (coeff[1] * outMedium) + (coeff[2] * outLarge);
  return resultNN;
}
