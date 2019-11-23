#pragma once

#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <ctgmath>
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <fstream>
#include <iostream>
#include <math.h>
#include <fstream>
#include <iostream>
#include <string>
#include <numeric>
#include <vector>

#include "Layer.h"

class Net {
public:
    Net(int _nLayers, int *_nNeurons, int _nInputs);

    ~Net();
    Layer *getLayer(int _layerIndex);
    void initNetwork(Neuron::weightInitMethod _wim, Neuron::biasInitMethod _bim, Neuron::actMethod _am);

    void setLearningRate(double _learningRate);
    void setInputs(const double *_inputs);
    void propInputs();
    void setGlobalError(double _globalError);
    void setError(double _leadError);
    void propError();
    void updateWeights();

    double getOutput(int _neuronIndex);
    double getSumOutput(int _neuronIndex);
    int getnLayers();
    int getnInputs();
    double getWeightDistance();
    double getLayerWeightDistance(int _layerIndex);
    double getWeights(int _layerIndex, int _neuronIndex, int _weightIndex);
    int getnNeurons();
    void saveWeights();
    void printNetwork();

private:
    int nLayers = 0;
    int nInputs = 0;
    int nOutputs = 0;
    const double *inputs = 0;
    Layer **layers = 0;
    double learningRate = 0;
    int nNeurons = 0;
    int nWeights = 0;
    double theLeadError = 0;
    double globalError = 0;

};
