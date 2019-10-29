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
	
	int onStepCompleted(Mat &statFrame, double deltaSensorData, vector<double> &predictorDeltas);
	double calcError(Mat &statFrame, vector<char> &sensorCHAR);
	void calcPredictors(Mat &frame, vector<double> &predictorDeltaMeans);
                       
	private:
	
	using clk = std::chrono::system_clock;
	clk::time_point start_time;

	float calibBlack[8+1] = {100 ,100 ,100 ,100 ,100 ,100, 100 ,100 ,1}; //x1
	float calibWhite[8+1] = {120,120,120,120,120,120,120,120,2}; //x2
	float diffCalib[8+1] = {0,0,0,0,0,0,0,0,0};
	float threshWhite[8+1] = {0,0,0,0,0,0,0,0,0};
    float threshBlack[8+1] = {0,0,0,0,0,0,0,0,0};

	static constexpr int nPredictorCols = 6;
	static constexpr int nPredictorRows = 8;
	static constexpr int nPredictors = nPredictorCols * nPredictorRows * 2; 
};
