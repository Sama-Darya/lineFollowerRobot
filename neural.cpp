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

std::vector<std::array<Bandpass, 5>> bandpassFilters;

static void initialize_filters(int numInputs, float sampleRate) {
  bandpassFilters.resize(numInputs);
  double fs = 1;
  double fmin = fs / 100;
  double fmax = fs / 10;
  double df = (fmax - fmin) / 4.0; // 4 is number of filters minus 1
  for (auto &bank : bandpassFilters) {
    double f = fmin;
    for (auto &filt : bank) {
      filt.setParameters(f, 0.51);
      f += df;
    }
  }
}

std::unique_ptr<Net> samanet;

void initialize_samanet(int numInputLayers, float sampleRate) {
  numInputLayers *= 5; // 5 is the number of filters

  int nNeurons[] = {5, 3, 1};
  samanet = std::make_unique<Net>(3, nNeurons, numInputLayers);
  samanet->initNetwork(Neuron::W_RANDOM, Neuron::B_NONE, Neuron::Act_Sigmoid);
  samanet->setLearningRate(0.001);
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
  predictor << " " << error;
  for (int j = 0; j < predictorDeltas.size(); ++j) {
    float sample = predictorDeltas[j];
    predictor << " " << sample;
    for (auto &filt : bandpassFilters[j]) {
      auto filtered = filt.filter(sample);
      networkInputs.push_back(filtered);
      predictor << " " << filtered;
    }
  }
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
  float outGentle = samanet->getOutput(0);
  //float outMedium = samanet->getOutput(1);
  //float outSharp = samanet->getOutput(2);
  float resultNN = 1 * outGentle;
  return resultNN;
}



