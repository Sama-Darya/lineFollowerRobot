#include "neural.h"
#include "clbp/Net.h"
#include "cvui.h"
#include <chrono>
#include <fstream>
#include <initializer_list>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include "bandpass.h"
#include <boost/circular_buffer.hpp>

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
const int numLayers = 12;

void initialize_samanet(int numInputLayers, double sampleRate) {
  numInputLayers *= 5; // 5 is the number of filters
  int numNeurons[numLayers]= {};
  int firstLayer = 11;
  int decrementLayer = 0;
  for (int i=0; i < numLayers - 1; i++){
    numNeurons[i] = firstLayer - i * decrementLayer;
    assert(numNeurons[i] > 3);
  }
  numNeurons[numLayers - 1] = 3; //output layer

  samanet = std::make_unique<Net>(numLayers, numNeurons, numInputLayers);
  samanet->initNetwork(Neuron::W_RANDOM, Neuron::B_NONE, Neuron::Act_Sigmoid);
  samanet->setLearningRate(0.1);
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
  samanet->setGlobalError(error);
  samanet->setError(error);
  samanet->propError();
  samanet->updateWeights(); // Learn from previous action
  samanet->setInputs(networkInputs.data()); //then take a new action
  samanet->propInputs();
  samanet->saveWeights();

  for (int i = 0; i <numLayers; i++){
    weightDistancesfs << samanet->getLayerWeightDistance(i) << " ";
  }
  weightDistancesfs << samanet->getWeightDistance() << "\n";

  double coeff[4] = {1,3,5};
  double outSmall = samanet->getOutput(0);
  double outMedium = samanet->getOutput(1);
  double outLarge = samanet->getOutput(2);
  //double outExtraLarge = samanet->getOutput(3);
  //cout << "Final Errors " << outSmall << " " << outMedium << " " << outLarge << " " << outExtraLarge << endl;

  double resultNN = (coeff[0] * outSmall) + (coeff[1] * outMedium) + (coeff[2] * outLarge); // + (coeff[3] * outExtraLarge);
  return resultNN;
}
