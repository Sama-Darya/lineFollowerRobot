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

boost::circular_buffer<double> predVector1[48];
boost::circular_buffer<double> predVector2[48];
boost::circular_buffer<double> predVector3[48];
boost::circular_buffer<double> predVector4[48];
boost::circular_buffer<double> predVector5[48];

static void initialize_filters(int numInputs, float sampleRate) {

  int nPred = numInputs / 5;
  int delayFactor[8] = {19,18,17,16,15,13,12,10};
  for (int i = 0; i < nPred; i++){
    int j= (int)(i / 6);
    predVector1[i].rresize(delayFactor[j]-3);
    predVector2[i].rresize(delayFactor[j]-1);
    predVector3[i].rresize(delayFactor[j]+0);
    predVector4[i].rresize(delayFactor[j]+1);
    predVector5[i].rresize(delayFactor[j]+3);
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

void initialize_samanet(int numInputLayers, float sampleRate) {
  numInputLayers *= 5; // 5 is the number of filters

  int nNeurons[] = {18, 9, 4};
  samanet = std::make_unique<Net>(3, nNeurons, numInputLayers);
  samanet->initNetwork(Neuron::W_RANDOM, Neuron::B_NONE, Neuron::Act_Sigmoid);
  samanet->setLearningRate(0.01);
  initialize_filters(numInputLayers, sampleRate);
}

std::ofstream weightDistancesfs("weight_distances.csv");
std::ofstream predictor("predictor.csv");

bool firstInputs = 1;

float run_samanet(cv::Mat &statFrame, std::vector<float> &predictorDeltas, float error) {

  using namespace std::chrono;
  milliseconds ms = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
  std::vector<float> networkInputs;

  predictor << ms.count();
  networkInputs.reserve(predictorDeltas.size() * 5);
  
  for (int i =0; i < predictorDeltas.size(); i++){
    predictor << " " << error;
    float sampleValue = predictorDeltas[i];
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
    float sample = predictorDeltas[j];
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
    cout << "DONE THIS" << endl;
  }
  samanet->setError(error);
  samanet->propError();
  samanet->updateWeights(); // Learn from previous action
  samanet->setInputs(networkInputs.data()); //then take a new action
  samanet->propInputs();

  samanet->saveWeights();

  weightDistancesfs << samanet->getLayerWeightDistance(0) << " " << samanet->getLayerWeightDistance(1) << " " << samanet->getLayerWeightDistance(2) << " " << samanet->getWeightDistance() << "\n";
  float coeff[4] = {1,2,3,4};
  float outSmall = samanet->getOutput(0);
  float outMedium = samanet->getOutput(1);
  float outLarge = samanet->getOutput(2);
  float outExtraLarge = samanet->getOutput(3);
  float resultNN = (coeff[0]*outSmall) + (coeff[1]*outMedium) + (coeff[2]*outLarge) + (coeff[3]*outExtraLarge);
  return resultNN;
}



