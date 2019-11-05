#pragma once

#include "Layer.h"

class Net {
public:
    Net(int _nLayers, int *_nNeurons, int _nInputs);
    ~Net();
    Layer *getLayer(int _layerIndex);
    void initNetwork(Neuron::weightInitMethod _wim, Neuron::biasInitMethod _bim, Neuron::actMethod _am);

    void setLearningRate(float _learningRate);
    void setInputs(const float *_inputs);
    void propInputs();
    void setError(float _leadError);
    void propError();
    void updateWeights();

    float getOutput(int _neuronIndex);
    float getSumOutput(int _neuronIndex);
    int getnLayers();
    int getnInputs();
    float getWeightDistance();
    float getLayerWeightDistance(int _layerIndex);
    float getWeights(int _layerIndex, int _neuronIndex, int _weightIndex);
    int getnNeurons();
    void saveWeights();
    void printNetwork();

private:
    int nLayers = 0;
    int nInputs = 0;
    int nOutputs = 0;
    const float *inputs = 0;
    Layer **layers = 0;
    float learningRate = 0;
    int nNeurons;
};
