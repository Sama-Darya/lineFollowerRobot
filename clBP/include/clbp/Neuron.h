#pragma once

#include <assert.h>
#include <iostream>
#include <math.h>
#include <stdio.h>


using namespace std;

class Neuron {
public:
    Neuron(int _nInputs);
    ~Neuron();
    enum biasInitMethod { B_NONE = 0, B_RANDOM = 1 };
    enum weightInitMethod { W_ZEROS = 0, W_ONES = 1, W_RANDOM = 2 };
    enum actMethod {Act_Sigmoid = 0, Act_Tanh = 1, Act_NONE = 2};

    void initNeuron(weightInitMethod _wim, biasInitMethod _bim, Neuron::actMethod _am);
    void setLearningRate(float _learningRate);

    void setInput(int _index, float _value);
    void propInputs(int _index, float _value);
    void calcOutput();
    void updateWeights();
    float doActivation(float _sum);
    float doActivationPrime(float _input);
    void setError(float _nextSum);  // for the output layer only
    void propError(float _nextSum); // used for all layers except the output

    float getOutput();
    float getSumOutput();
    float getWeights(int _inputIndex);
    float getInitWeights(int _inputIndex);
    float getError();
    float getWeightChange();
    float getWeightDistance();
    int getnInputs();
    void saveWeights(string _fileName);

    inline void setWeight(int _index, float _weight) {
        assert((_index >= 0) && (_index < nInputs));
        weights[_index] = _weight;
    }
    void printNeuron();

private:
    int nInputs = 0;
    float *inputs = 0;
    float *weights = 0;
    float *initialWeights = 0;
    float bias = 0;
    float error = 0;
    float output = 0;
    float learningRate = 0;
    float sum = 0;
    float weightSum = 0;
    float maxWeight = 1;
    float minWeight = 1;
    float weightChange=0;
    float weightsDifference = 0;
    int actMet = 0;
};
