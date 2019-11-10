#include "clbp/Neuron.h"

#include <iostream>
#include <ctgmath>
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <fstream>

using namespace std;

Neuron::Neuron(int _nInputs)
{
    nInputs=_nInputs;
    weights = new float[nInputs];
    initialWeights = new float[nInputs];
    inputs = new float[nInputs];
}

Neuron::~Neuron(){
    delete [] weights;
    delete [] initialWeights;
    delete [] inputs;
}

void Neuron::setInput(int _index,  float _value) {
    /* the seInput function sets one input value at the given index,
     * it has to be implemented in a loop inside the layer class to set
     * all the inputs associated with all the neurons in that layer*/
    assert((_index>=0)&&(_index<nInputs));
    /*checking _index is a valid int, non-negative and within boundary*/
    inputs[_index] = _value;
    //cout << "Neuron the input is: " << _value << endl;
}

void Neuron::propInputs(int _index,  float _value){
    /*works like setInput function expect it only applies
     * to the neurons in the hidden and output layers
     * and not the input layer*/
    assert((_index>=0)&&(_index<nInputs));
    inputs[_index] = _value;
}

void Neuron::initNeuron(weightInitMethod _wim, biasInitMethod _bim, Neuron::actMethod _am){
    for (int i=0; i<nInputs; i++){
        switch (_wim){
            case W_ZEROS:
                weights[i]=0;
                break;
            case W_ONES:
                weights[i]=1;
                break;
            case W_RANDOM:
                weights[i]=((float)rand()/(RAND_MAX));
                break;
                //cout << " Neuron: weight is: " << weights[i] << endl;
                /* rand function generates a random function between
                 * 0 and RAND_MAX, after the devision the weights are
                 * set to a value between 0 and 1 */
        }
        initialWeights[i] = weights[i];
        weightSum = 0;
          for (int i=0; i<nInputs; i++){
              weightSum += weights[i];
              maxWeight = max(maxWeight, weights[i]);
              minWeight = min (minWeight, weights[i]);
          }
    }
    switch (_bim){
        case B_NONE:
            bias=0;
            break;
        case B_RANDOM:
            bias=((float)rand()/RAND_MAX);
            break;
    }
    switch(_am){
        case Act_Sigmoid:
            actMet = 0;
            break;
        case Act_Tanh:
            actMet = 1;
            break;
        case Act_NONE:
            actMet = 2;
            break;
    }
}


float Neuron::getOutput(){
    return (output);
}

float Neuron::getSumOutput(){
    return (sum);
}

float Neuron::doActivation(float _sum){
    switch(actMet){
        case 0:
            output= (1/(1+(exp(-_sum)))) - 0.5;
            break;
        case 1:
            output = tanh(_sum) * 2;
            break;
        case 2:
            output = _sum;
            break;
    }
    return (output);
}

float Neuron::doActivationPrime(float _input){
    float result = 0;
    switch(actMet){
        case 0:
            result = ((doActivation(_input) + 0.5) * (1.5 - doActivation(_input))); //exp(-_input) / pow((exp(-_input) + 1),2);
            break;
        case 1:
            result = 1 - pow (tanh(_input), 2);
            break;
        case 2:
            result = 1;
            break;
    }
    return (result);
}

void Neuron::setLearningRate(float _learningRate){
    learningRate=_learningRate;
}

void Neuron::calcOutput(){
    sum=0;
    for (int i=0; i<nInputs; i++){
        sum += inputs[i] * weights[i];
    }
    sum += bias;
    sum = sum;
    //cout << "Neuron: " << maxWeight <<  "  ............   " << minWeight << endl;
    assert(std::isfinite(sum));
    output = doActivation(sum);
    assert(std::isfinite(output));
}

void Neuron::setError(float _leadError){
    error = _leadError * doActivationPrime(sum);
    assert(std::isfinite(error));
    /*might take a different format to propError*/
}

void Neuron::propError(float _nextSum){
    error = _nextSum * doActivationPrime(sum);
    assert(std::isfinite(_nextSum));

}

void Neuron::updateWeights(){
  weightSum = 0;
  maxWeight = 0;
  minWeight = 0;
    for (int i=0; i<nInputs; i++){
        weights[i] += learningRate * (error * inputs[i]);
        weightSum += (weights[i]);
        maxWeight = max (maxWeight,weights[i]);
        minWeight = min (maxWeight,weights[i]);
    }
    // for (int i=0; i<nInputs; i++){
    //   weights[i] = weights[i] / (maxWeight + minWeight);
    // }
}

float Neuron::getWeightChange(){
    weightsDifference = 0;
    weightChange = 0;
    for (int i=0; i<nInputs; i++){
        weightsDifference = weights[i] - initialWeights[i];
        weightChange += pow(weightsDifference,2);
    }
    return (weightChange);
}

float Neuron::getWeightDistance(){
    float weightDistance=sqrt(weightChange);
    return (weightDistance);
}

float Neuron::getError(){
    return (error);
}

int Neuron::getnInputs(){
    return (nInputs);
}

float Neuron::getWeights(int _inputIndex){
    assert((_inputIndex>=0)&&(_inputIndex<nInputs));
    return (weights[_inputIndex]);
}

float Neuron::getInitWeights(int _inputIndex){
    assert((_inputIndex>=0)&&(_inputIndex<nInputs));
    return (initialWeights[_inputIndex]);
}

void Neuron::saveWeights(string _fileName){
    std::ofstream Icofile;
    Icofile.open(_fileName, fstream::app);
    for (int i=0; i<nInputs; i++){
        Icofile << weights[i] << " " ;
    }
    Icofile << "\n";
    Icofile.close();
}

void Neuron::printNeuron(){
    cout<< "\t \t This neuron has " << nInputs << " inputs:";
    for (int i=0; i<nInputs; i++){
        cout<< " " << inputs[i];
    }
    cout<<endl;
    cout<< "\t \t The weights for those inputs are:";
    for (int i=0; i<nInputs; i++){
        cout<< " " << weights[i];
    }
    cout<<endl;
    cout<< "\t \t The bias of the neuron is: " << bias << endl;
    cout<< "\t \t The sum and output of this neuron are: " << sum << ", " << output << endl;
}
