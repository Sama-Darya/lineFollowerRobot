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

#include "Neuron.h"

class Layer {
public:
    // constructor de-constructor
    Layer(int _nNeurons, int _nInputs);
    ~Layer();

    // initialisation:
    enum whichGradient {exploding = 0, average = 1, vanishing = 2};

    void initLayer(int _layerIndex, Neuron::weightInitMethod _wim, Neuron::biasInitMethod _bim, Neuron::actMethod _am);
    void setlearningRate(double _learningRate);

    //forward propagation of inputs:
    void setInputs(const double *_inputs); // only for the first layer
    void propInputs(int _index, double _value);
    void calcOutputs();

    // ->->forward->-> propagation of error:
    void setForwardError(double _leadForwardError); //only for the first layer
    void propErrorForward(int _index, double _value);
    void calcForwardError();
    double getForwardError(int _neuronIndex);

    //back propagation of error:
    void setBackwardError(double _leadError);
    void propErrorBackward(int _neuronIndex, double _nextSum);
    double getBackwardError(int _neuronIndex);

    //Mid propagation of error:
    void setMidError(double _leadMidError);
    void calcMidError();
    double getMidError(int _neuronIndex);
    void propMidErrorForward(int _index, double _value);
    void propMidErrorBackward(int _neuronIndex, double _nextSum);

    //exploding/vanishing gradient:
    double getGradient(Neuron::whichError _whichError, whichGradient _whichGradient);

    //learning:
    void setErrorCoeff(double _globalCoeff, double _backwardsCoeff,
                        double _midCoeff, double _forwardCoeff,
                        double _localCoeff, double  _echoCoeff);
    void updateWeights();

    //global settings
    void setGlobalError(double _globalError);

    //local backpropagation of error
    void setLocalError(double _leadLocalError);
    void propGlobalErrorBackwardLocally(int _neuronIndex, double _nextSum);
    double getLocalError(int _neuronIndex);

    void setEchoError(double _clError);
    void echoErrorBackward(int _neuronIndex, double _nextSum);
    double getEchoError(int _neuronIndex);
    void echoErrorForward(int _index, double _value);
    void calcEchoError();


        //getters:
    Neuron *getNeuron(int _neuronIndex);
    int getnNeurons();
    double getOutput(int _neuronIndex);
    double getSumOutput(int _neuronIndex);
    double getWeights(int _neuronIndex, int _weightIndex);
    double getWeightChange();
    double getWeightDistance();
    double getGlobalError(int _neuronIndex);
    double getInitWeight(int _neuronIndex, int _weightIndex);

    //saving and inspecting
    void saveWeights();
    void snapWeights(); // This one just saves the final weight i.e. overwrites them
    void printLayer();

private:
    // initialisation:
    int nNeurons = 0;
    int nInputs = 0;
    double learningRate = 0;
    int myLayerIndex = 0;
    Neuron **neurons = 0;
    
    int layerHasReported = 0;

    //forward propagation of inputs:
    const double *inputs = 0;

    //forward propagation of error:
    double leadForwardError = 0;

    //back propagation of error:
    double leadBackwardError = 0;

    //mid propagation of error:
    double leadMidError = 0;

    //global settings
    double globalError = 0;

    double leadLocalError =0;

    //exploding vasnishing gradient:
    double averageError = 0;
    double maxError = 0;
    double minError = 0;


    //learning:
    double weightChange=0;
};
