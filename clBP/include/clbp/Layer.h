#pragma once
#include "Neuron.h"

class Layer {
public:
    Layer(int _nNeurons, int _nInputs);
    ~Layer();

    void setInputs(const float *_inputs); // only for the first layer
    void initLayer(Neuron::weightInitMethod _wim, Neuron::biasInitMethod _bim, Neuron::actMethod _am);
    void calcOutputs();
    float getOutput(int _neuronIndex);
    float getSumOutput(int _neuronIndex);
    void propInputs(int _index, float _value);
    /*this is for hidden and output layers (not input)*/
    void printLayer();
    void propError(int _neuronIndex, float _nextSum);
    int getnNeurons();
    void setlearningRate(float _learningRate);
    float getError(int _neuronIndex);
    float getWeights(int _neuronIndex, int _weightIndex);
    float getInitWeight(int _neuronIndex, int _weightIndex);
    float getWeightChange();
    float getWeightDistance();
    void setError(float _leadError);
    void updateWeights();
    int saveWeights(int _layerIndex, int _neuronCount);
    void snapWeights(int _layerIndex); // This one just saves the final weights
    // i.e. overwrites them

    Neuron *getNeuron(int _neuronIndex);

private:
    int nNeurons = 0;
    int nInputs = 0;
    const float *inputs = 0;
    Neuron **neurons = 0;
    float learningRate = 0;
    float weightChange=0;
};
