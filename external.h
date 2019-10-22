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
	
	
	static void initSensorFilters(float sampleRate = 30.f);
	int onStepCompleted(Mat &statFrame, double deltaSensorData, vector<double> &predictorDeltas);
	double calcError(Mat &statFrame, vector<char> &sensorCHAR);
	void calcPredictors(Mat &frame, vector<double> &predictorDeltaMeans);
                       
	private:
	
	using clk = std::chrono::system_clock;
	clk::time_point start_time;
	double errorMult = 2.5;
	double nnMult = 0;
	
	static constexpr int nPredictorCols = 6;
	static constexpr int nPredictorRows = 8;
	static constexpr int nPredictors = nPredictorCols * nPredictorRows * 2; 
};
