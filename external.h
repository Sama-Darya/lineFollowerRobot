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

/**
 * Main class for robot communication
 */

class Extern {

	public:
    /**
     * Constructor
     */
	Extern();
	/**
	 * Destructor
	 */
    ~Extern();

    /**
     * This is called at every time-step, it calls the neural network internally
     * @param statFrame The frame where the data is plotted
     * @param deltaSensorData The error signal from the sensors
     * @param predictorDeltas The predictor signals from the camera
     * @return returns the differential speed to be sent to the motors
     */
	int onStepCompleted(Mat &statFrame, double deltaSensorData, vector<double> &predictorDeltas);

	/**
	 * This function calculates the closed-loop error from the raw data that are received from the Arduino.
	 * It plots the data on the Stat Frame, it also calculates the integral error and monitors the 'success condition'.
	 * @param statFrame The frame where the data is plotted
	 * @param sensorCHAR An array of Characters: the raw data from the ground sensors
	 * @return Returns the closed-loop error
	 */
	double calcError(Mat &statFrame, vector<uint8_t> &sensorCHAR);

	/**
	 * This function calculates the predictor signals.
	 * @param frame The camera view
	 * @param predictorDeltaMeans A pointer to an array where the predictor signals are stored
	 */
	void calcPredictors(Mat &frame, vector<double> &predictorDeltaMeans);

	/**
	 * It reports on the number of predictors (pixel clusters) used
	 * @return Returns the number of predictors
	 */
	int getNpredictors();

	private:

	using clk = std::chrono::system_clock;
	clk::time_point start_time;
	double calibBlack[8+1]  = {100,110,115,125,   125,120,110,100,0};//x1 Red,Orange,Yellow,Green,Blue,Violet,Pink,White
	double threshBlack[8+1] = {135,140,140,150,   150,140,140,135,0};
	double threshWhite[8+1] = {140,145,145,155,   155,145,145,140,1};
	double calibWhite[8+1]  = {150,160,160,160,   160,160,160,150,2}; //x2 Red,Orange,Yellow,Green,Blue,Violet,Pink,White
	double diffCalib[8+1]   = {1,1,1,1,           1,1,1,1,1};
};
