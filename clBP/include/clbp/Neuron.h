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


using namespace std;

class Neuron {
public:
    Neuron(int _nInputs);
    ~Neuron();
    enum biasInitMethod { B_NONE = 0, B_RANDOM = 1 };
    enum weightInitMethod { W_ZEROS = 0, W_ONES = 1, W_RANDOM = 2 };
    enum actMethod {Act_Sigmoid = 0, Act_Tanh = 1, Act_NONE = 2};

    void initNeuron(weightInitMethod _wim, biasInitMethod _bim, Neuron::actMethod _am);
    void setLearningRate(double _learningRate);

    void setInput(int _index, double _value);
    void propInputs(int _index, double _value);
    void calcOutput();
    void updateWeights();
    double doActivation(double _sum);
    double doActivationPrime(double _input);
    void setGlobalError(double _globalError);
    void setError(double _nextSum);  // for the output layer only
    void propError(double _nextSum); // used for all layers except the output

    double getOutput();
    double getSumOutput();
    double getWeights(int _inputIndex);
    double getInitWeights(int _inputIndex);
    double getError();
    double getMaxWeight();
    double getMinWeight();
    double getSumWeight();
    double getGlobalError();
    double getWeightChange();
    double getWeightDistance();
    int getnInputs();
    void saveWeights(string _fileName);

    inline void setWeight(int _index, double _weight) {
        assert((_index >= 0) && (_index < nInputs));
        weights[_index] = _weight;
    }
    void printNeuron();

private:
    int nInputs = 0;
    double *inputs = 0;
    double *weights = 0;
    double *initialWeights = 0;
    double bias = 0;
    double error = 0;
    double globalError = 0;
    double output = 0;
    double learningRate = 0;
    double sum = 0;
    double weightSum = 0;
    double maxWeight = 1;
    double minWeight = 1;
    double weightChange=0;
    double weightsDifference = 0;
    int actMet = 0;
};
