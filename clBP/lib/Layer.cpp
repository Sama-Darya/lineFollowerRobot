#include "clbp/Layer.h"
#include "clbp/Neuron.h"

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


#include <fstream>

Layer::Layer(int _nNeurons, int _nInputs)

{
    nNeurons = _nNeurons; // number of neurons in this layer
    nInputs = _nInputs; // number of inputs to each neuron
    neurons = new Neuron*[nNeurons];
    /* dynamic allocation of memory to n number of
     * neuron-pointers and returning a pointer, "neurons",
     * to the first element */
    for (int i=0;i<nNeurons;i++){
        neurons[i]=new Neuron(nInputs);
    }
    /* each element of "neurons" pointer is itself a pointer
     * to a neuron object with specific no. of inputs*/
}

Layer::~Layer(){
    for(int i=0;i<nNeurons;i++) {
        delete neurons[i];
    }
    delete[] neurons;
    //delete[] inputs;
    /* it is important to delete any dynamic
     * memory allocation created by "new" */
}

void Layer::setInputs(const double* _inputs){
    /*this is only for the first layer*/
    inputs=_inputs;
    for (int j=0; j<nInputs; j++){
        Neuron** neuronsp = neurons;//point to the 1st neuron
        /* sets a temporarily pointer to neuron-pointers
         * within the scope of this function. this is inside
         * the loop, so that it is set to the first neuron
         * everytime a new value is distributed to neurons */
        double input= *inputs; //take this input value
        for (int i=0; i<nNeurons; i++){
            (*neuronsp)->setInput(j,input);
            //set this input value for this neuron
            neuronsp++; //point to the next neuron
        }
        inputs++; //point to the next input value
    }
}

void Layer::propInputs(int _index, double _value){
    for (int i=0; i<nNeurons; i++){
        neurons[i]->propInputs(_index, _value);
    }
}

void Layer::calcOutputs(){
    for (int i=0; i<nNeurons; i++){
        neurons[i]->calcOutput();
    }
}

void Layer::setGlobalError(double _globalError){
  globalError = _globalError;
  for (int i=0; i<nNeurons; i++){
      neurons[i]->setGlobalError(globalError);
  }
}

void Layer::setError(double _leadError){
    /* this is only for the final layer */
    for (int i=0; i<nNeurons; i++){
        neurons[i]->setError(_leadError);
    }
}

void Layer::propError(int _nLayers, int _layerIndex, int _neuronIndex, double _nextSum){
    neurons[_neuronIndex]->propError(_nextSum);
    // if (_neuronIndex == 0){
    //   cout << " BP>> acc2=Sum(W*E): " << _nextSum;
    //   cout << " e=acc*sigmoid'(acc1): " << neurons[_neuronIndex]->getError();
    //   cout << " FP>> acc1=Sum(w.in): " << neurons[_neuronIndex]->getSumOutput();
    //   cout << " sigmoid(sum): " << neurons[_neuronIndex]->getOutput();
    //   cout << " " << endl;
    // }
    
    /*
    if (_neuronIndex == 0 && _layerIndex == 0){
      std::ofstream vanishErrfile;
      string name1 = "vanishingErrorL0N0.csv";
      vanishErrfile.open(name1, fstream::app);
      vanishErrfile << _nextSum << " " << neurons[_neuronIndex]->getError() << " "
                    << neurons[_neuronIndex]->getSumOutput() << " " << neurons[_neuronIndex]->getOutput();
      vanishErrfile << "\n";
      vanishErrfile.close();
    }

    if (_neuronIndex == 0 && _layerIndex == _nLayers){
      std::ofstream vanishActfile;
      string name2 = "vanishingActLnN0.csv";
      vanishActfile.open(name2, fstream::app);
      vanishActfile << _nextSum << " " << neurons[_neuronIndex]->getError() << " "
                    << neurons[_neuronIndex]->getSumOutput() << " " << neurons[_neuronIndex]->getOutput();
      vanishActfile << "\n";
      vanishActfile.close();
    }
    */

}

double Layer::getError(int _neuronIndex){
    return (neurons[_neuronIndex]->getError());
}

double Layer::getGlobalError(int _neuronIndex){
    return (neurons[_neuronIndex]->getGlobalError());
}

double Layer::getSumOutput(int _neuronIndex){
    return (neurons[_neuronIndex]->getSumOutput());
}

double Layer::getWeights(int _neuronIndex, int _weightIndex){
    return (neurons[_neuronIndex]->getWeights(_weightIndex));
}

double Layer::getInitWeight(int _neuronIndex, int _weightIndex){
    return (neurons[_neuronIndex]->getInitWeights(_weightIndex));
}

double Layer::getWeightChange(){
    weightChange=0;
    for (int i=0; i<nNeurons; i++){
        weightChange += neurons[i]->getWeightChange();
    }
    //cout<< "Layer: WeightChange is: " << weightChange << endl;
    return (weightChange);
}

double Layer::getWeightDistance(){
    return sqrt(weightChange);
}

double Layer::getOutput(int _neuronIndex){
    return (neurons[_neuronIndex]->getOutput());
}


void Layer::initLayer(Neuron::weightInitMethod _wim, Neuron::biasInitMethod _bim, Neuron::actMethod _am){
    for (int i=0; i<nNeurons; i++){
        neurons[i]->initNeuron(_wim, _bim, _am);
    }
}

void Layer::setlearningRate(double _learningRate){
    learningRate=_learningRate;
    for (int i=0; i<nNeurons; i++){
        neurons[i]->setLearningRate(learningRate);
    }
}

void Layer::updateWeights(){
    for (int i=0; i<nNeurons; i++){
        neurons[i]->updateWeights();
    }
}

int Layer::getnNeurons(){
    return (nNeurons);
}

int Layer::saveWeights(int _layerIndex, int _neuronCount){
    char l = '0';
    char n = '0';
    l += _layerIndex + 1;
    char decimal = '0';
    bool skip = true;
    for (int i=0; i<nNeurons; i++){
        if (skip == true){
            n += 1;
            }
            if(skip == false){
                skip = true;
            }
        _neuronCount += 1;
        string name = "w";
        name += 'L';
        name += l;
        name += 'N';
        name += decimal;
        name += n;
        name += ".csv";
        neurons[i]->saveWeights(name);
        if (n == '9'){
            decimal += 1;
            n= '0';
            skip = false;
        }
    }
    return (_neuronCount);
}

void Layer::snapWeights(int _layerIndex){
    std::ofstream wfile;
    char l = '0';
    l += _layerIndex + 1;
    string name = "wL";
    name += l;
    name += ".csv";
    wfile.open(name);
    for (int i=0; i<nNeurons; i++){
        for (int j=0; j<nInputs; j++){
            wfile << neurons[i]->getWeights(j) << " ";
        }
        wfile << "\n";
    }
    wfile.close();
}

Neuron* Layer::getNeuron(int _neuronIndex){
    assert(_neuronIndex < nNeurons);
    return (neurons[_neuronIndex]);
}

void Layer::printLayer(){
    cout<< "\t This layer has " << nNeurons << " Neurons" <<endl;
    cout<< "\t There are " << nInputs << " inputs to this layer" <<endl;
    for (int i=0; i<nNeurons; i++){
        cout<< "\t Neuron number " << i << ":" <<endl;
        neurons[i]->printNeuron();
    }

}
