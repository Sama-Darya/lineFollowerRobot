#include "opencv2/opencv.hpp"

#include <boost/circular_buffer.hpp>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>
#include <numeric>

#include "neural.h"
#include "external.h"
#include "cvui.h"

#include "LowPassFilter.hpp"

#include "bandpass.h"

#include <initializer_list>
#include <memory>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;
using namespace cv;


Extern::Extern(){

}
int samplingFreq = 30; // 30Hz is the sampling frequency
int figureLength = 5; //seconds

boost::circular_buffer<double> prevErrors(samplingFreq * figureLength);

boost::circular_buffer<double> sensor0(samplingFreq * figureLength);
boost::circular_buffer<double> sensor1(samplingFreq * figureLength);
boost::circular_buffer<double> sensor2(samplingFreq * figureLength);
boost::circular_buffer<double> sensor3(samplingFreq * figureLength);
boost::circular_buffer<double> sensor4(samplingFreq * figureLength);
boost::circular_buffer<double> sensor5(samplingFreq * figureLength);
boost::circular_buffer<double> sensor6(samplingFreq * figureLength);
boost::circular_buffer<double> sensor7(samplingFreq * figureLength);

std::ofstream datafs("speedDiffdata.csv");

double errorMult = 2;
double nnMult = 1;
double nnMultScale = 0;
int ampUp = 0;
int startLearning = 0;

int Extern::onStepCompleted(cv::Mat &statFrame, double deltaSensorData, std::vector<double> &predictorDeltas) {
  prevErrors.push_back(deltaSensorData); //puts the errors in a buffer for plotting

  double error = deltaSensorData;
  cvui::text(statFrame, 10, 250, "Sensor Error Multiplier: ");
  cvui::trackbar(statFrame, 180, 250, 400, &errorMult, (double)0.0, (double)10.0, 1, "%.2Lf", 0, 0.5);
  cvui::text(statFrame, 10, 300, "Net Output Multiplier: ");
  cvui::trackbar(statFrame, 180, 300, 400, &nnMult, (double)0.0, (double)10.0, 1, "%.2Lf", 0, 0.5);
	cvui::trackbar(statFrame, 180, 350, 400, &nnMultScale, (double)0, (double)20, 1, "%.2Lf", 0, 0.5);
  assert(std::isfinite(error));
  double errorGain = 1;
  double errroForLearning = errorGain * error;
  if (nnMult == 0){
    errroForLearning = 0;
  }
  double result = run_samanet(statFrame, predictorDeltas, errroForLearning); //does one learning iteration, why divide by 5?
  {
    std::vector<double> error_list(prevErrors.begin(), prevErrors.end());
    cvui::sparkline(statFrame, error_list, 10, 50, 580, 200, 0x000000);
    double elapsed_s = std::chrono::duration_cast<std::chrono::milliseconds>(clk::now() - start_time) .count() / 1000.f;
    double chart_start_t = prevErrors.full() ? elapsed_s - 60 : 0.f;
    //cvui::printf(statFrame, 10, 250, "%.2fs", chart_start_t);
    //cvui::printf(statFrame, 540, 250, "%.2fs", elapsed_s);
  }
  double reflex = error * errorMult;
  double learning = result * nnMult * pow(10,nnMultScale);
  cvui::text(statFrame, 220, 10, "Net out:");
  cvui::printf(statFrame, 300, 10, "%+.4lf (%+.4lf)", result, learning);
  cvui::text(statFrame, 220, 30, "Error:");
  cvui::printf(statFrame, 300, 30, "%+.4lf (%+.4lf)", deltaSensorData, reflex);
  int gain = 1;
  double errorSpeed = (reflex + learning) * gain;
  using namespace std::chrono;
  milliseconds ms = duration_cast<milliseconds>(system_clock::now().time_since_epoch());

  datafs << deltaSensorData << " "
         << error << " "
         << reflex << " "
         << result << " "
         << learning << " "
         << errorSpeed << "\n";

  return (int)errorSpeed;
}
Bandpass sensorFilters[8];

double cutOff = 10;
double sampFreq = 0.033;
LowPassFilter lpf0(cutOff, sampFreq);
LowPassFilter lpf1(cutOff, sampFreq);
LowPassFilter lpf2(cutOff, sampFreq);
LowPassFilter lpf3(cutOff, sampFreq);
LowPassFilter lpf4(cutOff, sampFreq);
LowPassFilter lpf5(cutOff, sampFreq);
LowPassFilter lpf6(cutOff, sampFreq);
LowPassFilter lpf7(cutOff, sampFreq);

const int loopLength = 800;
boost::circular_buffer<double> movingIntegralVector(loopLength); // each loop of path is 1500 samples
int checkSucess = 0;
int consistency = 0;
int stepCount = 1;
int successDone = 0;

std::ofstream errorSuccessDatafs("errorSuccessData.csv");
std::ofstream successRatef("successTime.csv");

int sensorInUse = 4;
double thresholdInteg = 10.2;
int getThreshold = 1;
double maxMovingIntegral = 0;
int setFirstEncounter = 1;
int firstEncounter = 0 ;
double totalIntegral = 0;


double Extern::calcError(cv::Mat &statFrame, vector<uint8_t> &sensorCHAR){
	const int numSensors = 8;
	int startIndex = 8;
	int sensorINT[numSensors+1]= {0,0,0,0,0,0,0,0,0};
	for (int i = 0; i < numSensors+1 ; i++){
		sensorINT[i] = (int)sensorCHAR[i];
		if (sensorINT[i] == 0){
			startIndex = i + 1;
		}
	}
    double sensorVAL[numSensors+1]= {0,0,0,0,0,0,0,0,0};
    double mapBlack = 50; //y1
    double mapWhite = 250; //y2
    double m [8+1] = {1,1,1,1,1,1,1,1,1};
    char colorName[8] = {'R', 'O', 'Y', 'G', 'B', 'V', 'P', 'W'}; // Red,Orange,Yellow,Green,Blue,Violet,Pink,White
    const int colorCode[8] = {0xff0000, 0xff9900, 0xffff00, 0x00ff00, 0x00ffff, 0x9900ff, 0xff00ff, 0xffffff};

    for (int i = 0; i < numSensors; i++){
      int remainIndex = (startIndex + i) % (numSensors+1);
      sensorVAL[i] = sensorINT[remainIndex];
      if (sensorVAL[i] > threshWhite[i] ){calibWhite[i] = sensorVAL[i];}
      if (sensorVAL[i] < threshBlack[i] ){calibBlack[i] = sensorVAL[i];}
      diffCalib[i] = calibWhite[i] - calibBlack[i];
      assert(std::isfinite(diffCalib[i]));
      m[i] = (mapWhite - mapBlack)/(diffCalib[i]);
      sensorVAL[i] = m[i] * (sensorINT[remainIndex] - calibBlack[i]) + mapBlack;
      assert(std::isfinite(sensorVAL[i]));
      // cout << colorName[i] << " Bcal: " << (int)calibBlack[i] << " " << (int)threshBlack[i]
      //       << " raw: " << (int)sensorINT[remainIndex]
      //       << " Wcal: " << (int)threshWhite[i] << " " << (int)calibWhite[i]
      //       << " cal: " << (int)sensorVAL[i] << endl;

    }
    //cout << " ------------------------------- "<< endl;

    sensorVAL[0] = lpf0.update(sensorVAL[0]);
    sensorVAL[1] = lpf1.update(sensorVAL[1]);
    sensorVAL[2] = lpf2.update(sensorVAL[2]);
    sensorVAL[3] = lpf3.update(sensorVAL[3]);
    sensorVAL[4] = lpf4.update(sensorVAL[4]);
    sensorVAL[5] = lpf5.update(sensorVAL[5]);
    sensorVAL[6] = lpf6.update(sensorVAL[6]);
    sensorVAL[7] = lpf7.update(sensorVAL[7]);

    double errorWeights[numSensors/2] = {7,5,3,1};
    double error = 0;
    for (int i = 0 ; i < 2 ; i++){
       error += (errorWeights[i]) * (sensorVAL[i] - sensorVAL[numSensors -1 -i]);
    }
    error = error / (mapWhite - mapBlack);
    assert(std::isfinite(error));

    //plot the sensor values:
    double minVal = 40;
    double maxVal = 260;

    sensor0.push_back(sensorVAL[0]); //puts the errors in a buffer for plotting
    sensor0[0] = minVal;
    sensor0[1] = maxVal;
    std::vector<double> sensor_list0(sensor0.begin(), sensor0.end());
    cvui::sparkline(statFrame, sensor_list0, 10, 50, 580, 200, 0xff0000);

    sensor1.push_back(sensorVAL[1]); //puts the errors in a buffer for plotting
    sensor1[0] = minVal;
    sensor1[1] = maxVal;
    std::vector<double> sensor_list1(sensor1.begin(), sensor1.end());
    cvui::sparkline(statFrame, sensor_list1, 10, 50, 580, 200, 0xff9900);

    sensor2.push_back(sensorVAL[2]); //puts the errors in a buffer for plotting
    sensor2[0] = minVal;
    sensor2[1] = maxVal;
    std::vector<double> sensor_list2(sensor2.begin(), sensor2.end());
    cvui::sparkline(statFrame, sensor_list2, 10, 50, 580, 200, 0xffff00);

    sensor3.push_back(sensorVAL[3]); //puts the errors in a buffer for plotting
    sensor3[0] = minVal;
    sensor3[1] = maxVal;
    std::vector<double> sensor_list3(sensor3.begin(), sensor3.end());
    cvui::sparkline(statFrame, sensor_list3, 10, 50, 580, 200, 0x00ff00);

    sensor4.push_back(sensorVAL[4]); //puts the errors in a buffer for plotting
    sensor4[0] = minVal;
    sensor4[1] = maxVal;
    std::vector<double> sensor_list4(sensor4.begin(), sensor4.end());
    cvui::sparkline(statFrame, sensor_list4, 10, 50, 580, 200, 0x00ffff);

    sensor5.push_back(sensorVAL[5]); //puts the errors in a buffer for plotting
    sensor5[0] = minVal;
    sensor5[1] = maxVal;
    std::vector<double> sensor_list5(sensor5.begin(), sensor5.end());
    cvui::sparkline(statFrame, sensor_list5, 10, 50, 580, 200, 0x9900ff);

    sensor6.push_back(sensorVAL[6]); //puts the errors in a buffer for plotting
    sensor6[0] = minVal;
    sensor6[1] = maxVal;
    std::vector<double> sensor_list6(sensor6.begin(), sensor6.end());
    cvui::sparkline(statFrame, sensor_list6, 10, 50, 580, 200, 0xff00ff);

    sensor7.push_back(sensorVAL[7]); //puts the errors in a buffer for plotting
    sensor7[0] = minVal;
    sensor7[1] = maxVal;
    std::vector<double> sensor_list7(sensor7.begin(), sensor7.end());
    cvui::sparkline(statFrame, sensor_list7, 10, 50, 580, 200, 0xffffff);

    //average the error over the last N samples:
    stepCount += 1;
    checkSucess += 1;
    if (fabs(error) > 0.01 && setFirstEncounter == 1){
      firstEncounter = stepCount;
      setFirstEncounter =0;
    }
    movingIntegralVector.push_back(abs(error));
    double movingIntegralSum = std::accumulate(movingIntegralVector.begin(), movingIntegralVector.end(), 0.00);
    double movingIntegralAve = movingIntegralSum/loopLength;
    totalIntegral += abs(error);
    double totalIntegralAve = totalIntegral/stepCount;
    maxMovingIntegral = max (maxMovingIntegral,fabs(movingIntegralAve));
    thresholdInteg = 0.1; //

    errorSuccessDatafs << error << " "
           << movingIntegralAve << "\n";
    
    int actualSteps = stepCount-firstEncounter;
    cvui::text(statFrame, 100, 20, "Step:");
    cvui::printf(statFrame, 150, 20, "%d", actualSteps);

    if (nnMult == 0){ // this is for reflex
      if ( stepCount - firstEncounter > loopLength * 2 && successDone == 0){
          cout << "DONE!" << endl;
          cout << "movingIntegralAve: " << movingIntegralAve << endl;
          cout << "maxMovingIntegral: " << maxMovingIntegral << endl;
          cout << "totalIntegral: " << totalIntegral << endl;
          cout << "totalIntegralAve: " << totalIntegralAve << endl;
        successDone = 1;
        successRatef << firstEncounter << " " << actualSteps 
                    << " " << movingIntegralAve << " " << maxMovingIntegral 
                    << " " << totalIntegral << " " << totalIntegralAve << "\n";
        exit(19);
      }
    }else{ // this is for learning
      if (checkSucess > firstEncounter + loopLength/2 && fabs(movingIntegralAve) < thresholdInteg && successDone == 0){
          //start checking for success 100 steps after it has seen the line first
        consistency += 1;
        if (consistency > 1){ 
          cout << "SUCCESS! on Step: " << actualSteps << endl;
          cout << "movingIntegralAve: " << movingIntegralAve << endl;
          cout << "maxMovingIntegral: " << maxMovingIntegral << endl;
          cout << "totalIntegral: " << totalIntegral << endl;
          cout << "totalIntegralAve: " << totalIntegralAve << endl;
          successDone = 1;
          successRatef << firstEncounter << " " << stepCount - firstEncounter 
                    << " " << movingIntegralAve << " " << maxMovingIntegral
                    << " " << totalIntegral << " " << totalIntegralAve << "\n";                    
          exit(14);
        }
      }else{consistency = 0;}
    }
    return error;
}

static constexpr int nPredictorCols = 6;
static constexpr int nPredictorRows = 8;
static constexpr int nPredictors = nPredictorCols * nPredictorRows;

int Extern::getNpredictors (){
    return nPredictors;
}

void Extern::calcPredictors(Mat &frame, vector<double> &predictorDeltaMeans){
	// Define the rect area that we want to consider.
    int areaWidth = 600;
    int areaHeight = 120;
    int offsetFromTop = 350;
    // VERTICAL RESOLUTION OF CAMERA SHOULD ADJUST
    int startX = (640 - areaWidth) / 2;
    auto area = Rect{startX, offsetFromTop, areaWidth, areaHeight};
    int predictorWidth = area.width / 2 / nPredictorCols;
    int predictorHeight = area.height / nPredictorRows;
	Mat edges;
	cvtColor(frame, edges, COLOR_BGR2GRAY);
    rectangle(edges, area, Scalar(122, 144, 255));
    predictorDeltaMeans.clear();

	int areaMiddleLine = area.width / 2 + area.x;
  double predThreshW[nPredictorCols][nPredictorRows] = {{170,180,190,200,210,220,220,220},
                                                        {170,180,190,190,200,210,220,210},
                                                        {160,170,180,190,190,200,200,200},
                                                        {150,160,170,180,180,180,190,190},
                                                        {140,150,150,160,160,170,170,170},
                                                        {130,130,130,140,140,140,140,140}};
  double predThreshWAdjustment = 20;
  double predThreshWDiff = 50;
	for (int k = 0; k < nPredictorRows; ++k) {
      for (int j = 0; j < nPredictorCols ; ++j) {
         auto lPred =
            Rect(areaMiddleLine - (j + 1) * predictorWidth,
                 area.y + k * predictorHeight, predictorWidth, predictorHeight);
        auto rPred =
            Rect(areaMiddleLine + (j)*predictorWidth,
                 area.y + k * predictorHeight, predictorWidth, predictorHeight);
        auto grayMeanL = mean(Mat(edges, lPred))[0];
        auto grayMeanR = mean(Mat(edges, rPred))[0];
        if (grayMeanL < predThreshW[j][k] - predThreshWDiff){grayMeanL = predThreshW[j][k] - predThreshWDiff;}
        if (grayMeanR < predThreshW[j][k] - predThreshWDiff){grayMeanR = predThreshW[j][k] - predThreshWDiff;}
        if (grayMeanL > predThreshW[j][k] - predThreshWAdjustment){grayMeanL = predThreshW[j][k] - predThreshWAdjustment;}
        if (grayMeanR > predThreshW[j][k] - predThreshWAdjustment){grayMeanR = predThreshW[j][k] - predThreshWAdjustment;}
        double predScale = 0.1;
        auto predValue = ((grayMeanL - grayMeanR) / predThreshWDiff) * predScale;
        predictorDeltaMeans.push_back(predValue);
        putText(frame, std::to_string((int)(grayMeanL - grayMeanR)),
                Point{lPred.x + lPred.width / 2 - 13,
                      lPred.y + lPred.height / 2 + 5},
                FONT_HERSHEY_TRIPLEX, 0.4, {0, 0, 0});
        putText(frame, std::to_string((int)grayMeanR),
                Point{rPred.x + rPred.width / 2 - 13,
                      rPred.y + rPred.height / 2 + 5},
                FONT_HERSHEY_TRIPLEX, 0.4, {0, 0, 0});
        rectangle(frame, lPred, Scalar(50, 50, 50));
        rectangle(frame, rPred, Scalar(50, 50, 50));
      }
    }
    line(frame, {areaMiddleLine, 0}, {areaMiddleLine, frame.rows}, Scalar(50, 50, 255));
    imshow("robot view", frame);
}
