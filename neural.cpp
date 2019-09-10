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

  int nNeurons[] = {24, 12, 3};
  samanet = std::make_unique<Net>(3, nNeurons, numInputLayers);
  samanet->initNetwork(Neuron::W_RANDOM, Neuron::B_NONE, Neuron::Act_Sigmoid);
  samanet->setLearningRate(0.001);

  initialize_filters(numInputLayers, sampleRate);
}

std::ofstream weightDistancesfs("weight_distances.csv");
std::ofstream predictor("predictor.csv");

double run_samanet(cv::Mat &statFrame, std::vector<float> &predictorDeltas,
                   double error) {

  using namespace std::chrono;
  milliseconds ms =
      duration_cast<milliseconds>(system_clock::now().time_since_epoch());

  std::vector<double> networkInputs;

  //predictor << ms.count();
  networkInputs.reserve(predictorDeltas.size() * 5);
  for (int j = 0; j < predictorDeltas.size(); ++j) {
    predictor << " " << error;
    float sample = predictorDeltas[j];
      //if (j == 0) {
        predictor << " " << sample;
      //}
    //cout << "predictor value is: " << sample << endl;
    for (auto &filt : bandpassFilters[j]) {
      auto filtered = filt.filter(sample);
      //cout << "predictor value is: " << filtered << endl;
      networkInputs.push_back(filtered);
      //if (j == 0) {
        predictor << " " << filtered;
      //}
    }
  }
  predictor << "\n" ;

  samanet->setInputs(networkInputs.data()); //then take a new action
  samanet->propInputs();

  samanet->setError(error);
  samanet->propError();
  samanet->updateWeights(); // Learn from previous action
  
  save_samanet();

  weightDistancesfs << samanet->getLayerWeightDistance(0) << " " << samanet->getLayerWeightDistance(1) << " " << samanet->getLayerWeightDistance(2) << " " << samanet->getWeightDistance() << "\n";
  //cout << "weight distance is: " << samanet->getWeightDistance() << endl;

  double outGentle = samanet->getOutput(0);
  double outMedium = samanet->getOutput(1);
  double outSharp = samanet->getOutput(2);
  
  double resultNN = 1 * outGentle + 2 * outMedium + 3 * outSharp;
  
  return resultNN;
}

void save_samanet() { samanet->saveWeights(); }


