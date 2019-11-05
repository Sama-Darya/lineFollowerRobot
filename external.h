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
                       
	private:
	
	using clk = std::chrono::system_clock;
	clk::time_point start_time;

	float calibBlack[8+1]  = {95 ,115,115,120,   120,110,100, 95,0};//x1 Red,Orange,Yellow,Green,Blue,Violet,Pink,White
	float calibWhite[8+1]  = {115,125,120,125,   125,120,115,115,2}; //x2 Red,Orange,Yellow,Green,Blue,Violet,Pink,White
	float diffCalib[8+1]   = {0,0,0,0,           0,0,0,0,0};
	float threshWhite[8+1] = {115,125,120,125,   125,120,115,115,2};
    float threshBlack[8+1] = {95 ,115,115,120,   120,110,100, 95,0};
};
