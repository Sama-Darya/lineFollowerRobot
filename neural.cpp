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
  double fmin = fs / 39;
  double fmax = fs / 9;
  double df = (fmax - fmin) / 4.0;
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
  numInputLayers *= 5;

  int nNeurons[] = {16, 8, 1};
  samanet = std::make_unique<Net>(3, nNeurons, numInputLayers);
  samanet->initNetwork(Neuron::W_RANDOM, Neuron::B_NONE, Neuron::Act_Sigmoid);
  samanet->setLearningRate(0.05);

  initialize_filters(numInputLayers, sampleRate);
}

std::ofstream weightDistancesfs("weight_distances.csv");
std::ofstream filterout("filterouts.csv");
std::ofstream unfilteredout("unfilteredouts.csv");

double run_samanet(cv::Mat &statFrame, std::vector<float> &predictorDeltas,
                   double error) {

  using namespace std::chrono;
  milliseconds ms =
      duration_cast<milliseconds>(system_clock::now().time_since_epoch());

  std::vector<double> networkInputs;

  filterout << "\n" << ms.count();
  unfilteredout << "\n" << ms.count();
  networkInputs.reserve(predictorDeltas.size() * 5);
  for (int j = 0; j < predictorDeltas.size(); ++j) {
    float sample = predictorDeltas[j];
    for (auto &filt : bandpassFilters[j]) {
      auto filtered = filt.filter(sample);
      networkInputs.push_back(filtered);
      if (j == 0) {
        unfilteredout << "," << sample;
        filterout << "," << filtered;
      }
    }
  }

  samanet->setError(error);
  samanet->propError();
  samanet->updateWeights(); // Learn from previous action

  samanet->setInputs(networkInputs.data());
  samanet->propInputs();

  //weightDistancesfs << ms.count() << "," << samanet->getWeightDistanceLayer(0) << "," << samanet->getWeightDistanceLayer(1) << "," << samanet->getWeightDistanceLayer(2) << "\n";
  return samanet->getOutput(0); //then take a new action
}

void dump_samanet() { samanet->saveWeights(); }
