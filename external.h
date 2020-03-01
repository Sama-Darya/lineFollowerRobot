#pragma once
#include <vector>

#include "opencv2/opencv.hpp"

#include <boost/circular_buffer.hpp>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>

#include "cvui.h"

using namespace cv;
using namespace std;

class Extern {

	public:
	Extern();

	void onStepCompleted(Mat &statFrame, double deltaSensorData, vector<double> &predictorDeltas);
	double calcError(Mat &statFrame, vector<char> &sensorCHAR);
	void calcPredictors(Mat &frame, vector<double> &predictorDeltaMeans);
	int getNpredictors();
    int16_t getExtDifferentialVelocity();
    int16_t getExtLeftVelocity();
    int16_t getExtRightVelocity();

	private:
	using clk = std::chrono::system_clock;
	clk::time_point start_time;

	double calibBlack[8+1]  = {100,110,115,125,   125,120,110,100,0};//x1 Red,Orange,Yellow,Green,Blue,Violet,Pink,White
	double threshBlack[8+1] = {135,140,140,150,   150,140,140,135,0};
	double threshWhite[8+1] = {140,145,145,155,   155,145,145,140,1};
	double calibWhite[8+1]  = {150,160,160,160,   160,160,160,150,2}; //x2 Red,Orange,Yellow,Green,Blue,Violet,Pink,White
	double diffCalib[8+1]   = {1,1,1,1,           1,1,1,1,1};

    int16_t extLeftVelocity;
    int16_t extRightVelocity;
    int16_t extDifferentialVelocity;

};
