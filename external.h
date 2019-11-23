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

	int onStepCompleted(Mat &statFrame, float deltaSensorData, vector<float> &predictorDeltas);
	float calcError(Mat &statFrame, vector<char> &sensorCHAR);
	void calcPredictors(Mat &frame, vector<float> &predictorDeltaMeans);
	int getNpredictors();

	private:

	using clk = std::chrono::system_clock;
	clk::time_point start_time;

	float calibBlack[8+1]  = {100,110,115,125,   125,120,110,100,0};//x1 Red,Orange,Yellow,Green,Blue,Violet,Pink,White
	float threshBlack[8+1] = {135,140,140,150,   150,140,140,135,0};
	float threshWhite[8+1] = {140,145,145,155,   155,145,145,140,1};
	float calibWhite[8+1]  = {150,160,160,160,   160,160,160,150,2}; //x2 Red,Orange,Yellow,Green,Blue,Violet,Pink,White

	float diffCalib[8+1]   = {1,1,1,1,           1,1,1,1,1};
};
