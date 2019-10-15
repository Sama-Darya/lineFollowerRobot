#include "opencv2/opencv.hpp"
#include "serialib.h"


#include <boost/circular_buffer.hpp>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>

#include "neural.h"
#define CVUI_IMPLEMENTATION
#include "cvui.h"

#if defined(_WIN32) || defined(_WIN64)
#define DEVICE_PORT "COM4" // COM1 for windows
#endif

#ifdef __linux__
#define DEVICE_PORT "/dev/ttyUSB0" // This is for Arduino, ttyS0 for linux, otherwise ttyUSB0, if it does not open try: sudo chmod 666 /dev/ttyS0 or ttyUSB0
#endif

#define STAT_WINDOW "statistics & options"

using namespace cv;
using namespace std;
constexpr int ESC_key = 27;

int startIndex = 8;
int doneShift = 0;

static constexpr int nPredictorCols = 6;
static constexpr int nPredictorRows = 8;
static constexpr int nPredictors = nPredictorCols * nPredictorRows * 2;

double errorMult = 2.5;
double nnMult = 0;

std::ofstream datafs("data.csv");

using clk = std::chrono::system_clock;
clk::time_point start_time;

int samplingFreq = 30; // 30Hz is the sampling frequency
int figureLength = 10; //seconds

boost::circular_buffer<double> prevErrors(samplingFreq * figureLength); // this accumulate data for one minute

boost::circular_buffer<double> sensor0(samplingFreq * figureLength); // this accumulate data for one minute
boost::circular_buffer<double> sensor1(samplingFreq * figureLength); // this accumulate data for one minute
boost::circular_buffer<double> sensor2(samplingFreq * figureLength); // this accumulate data for one minute
boost::circular_buffer<double> sensor3(samplingFreq * figureLength); // this accumulate data for one minute
boost::circular_buffer<double> sensor4(samplingFreq * figureLength); // this accumulate data for one minute
boost::circular_buffer<double> sensor5(samplingFreq * figureLength); // this accumulate data for one minute
boost::circular_buffer<double> sensor6(samplingFreq * figureLength); // this accumulate data for one minute
boost::circular_buffer<double> sensor7(samplingFreq * figureLength); // this accumulate data for one minute


int16_t onStepCompleted(cv::Mat &statFrame, double deltaSensorData,
                        std::vector<float> &predictorDeltas) {
  prevErrors.push_back(deltaSensorData); //puts the errors in a buffer for plotting

  double errorGain = 1;
  double error = errorGain * deltaSensorData;

  cvui::text(statFrame, 10, 320, "Sensor Error Multiplier: ");
  cvui::trackbar(statFrame, 180, 300, 400, &errorMult, (double)0.0, (double)15.0,
                 1, "%.2Lf", 0, 0.05);

  cvui::text(statFrame, 10, 370, "Net Output Multiplier: ");
  cvui::trackbar(statFrame, 180, 350, 400, &nnMult, (double)0.0, (double)5.0,
                 1, "%.2Lf", 0, 0.05);

  double errorN = error;
   //if (errorN < 0.2 && errorN > -0.2){ errorN = 0; }
  double result = run_samanet(statFrame, predictorDeltas, errorN); //does one learning iteration, why divide by 5?
	//cout<< "inside onStepComplete result: " << result << endl

  {
    std::vector<double> error_list(prevErrors.begin(), prevErrors.end());
    cvui::sparkline(statFrame, error_list, 10, 50, 580, 200);
    float elapsed_s = std::chrono::duration_cast<std::chrono::milliseconds>(
                          clk::now() - start_time)
                          .count() /
                      1000.f;
    float chart_start_t = prevErrors.full() ? elapsed_s - 60 : 0.f;
    cvui::printf(statFrame, 10, 250, "%.2fs", chart_start_t);
    cvui::printf(statFrame, 540, 250, "%.2fs", elapsed_s);
  }
  double reflex = error * errorMult;
  double learning = result * nnMult * 1;
  
  cvui::text(statFrame, 220, 10, "Net out:");
  cvui::printf(statFrame, 300, 10, "%+.4lf (%+.4lf)", result, learning);

  cvui::text(statFrame, 220, 30, "Error:");
  cvui::printf(statFrame, 300, 30, "%+.4lf (%+.4lf)", deltaSensorData, reflex);
  
  int gain = 1;
  double error2 = (reflex + learning) * gain;
  
  int16_t differentialOut = (int16_t)(error2 * 1);
  //float differentialOut = (float)(error2);

  using namespace std::chrono;
  milliseconds ms =
      duration_cast<milliseconds>(system_clock::now().time_since_epoch());

  datafs << deltaSensorData << " "   // error from error units
         << reflex << " "            // reflex
         << learning << " "            // net output
         << differentialOut << "\n"; // final differential output

  return differentialOut;
}

/*
double calculateErrorValue(Mat &origframe, Mat &output) {
  constexpr int numErrorSensors = 5;
  int areaWidth = 580;
  int areaHeight = 30;
  int offsetFromBottom = 0;
  int blackSensorThreshold = 70;
  int startX = (origframe.cols - areaWidth) / 2;
  auto area = Rect{startX, origframe.rows - areaHeight - offsetFromBottom,
                   areaWidth, areaHeight};

  int areaMiddleLine = area.width / 2 + area.x;

  int sensorWidth = area.width / 2 / numErrorSensors;
  int sensorHeight = areaHeight;

  std::array<double, numErrorSensors> sensorWeights;

    sensorWeights[0] = 0;
    sensorWeights[1] = 0.5;
    sensorWeights[2] = 1;
    sensorWeights[3] = 1.5;
    sensorWeights[4] = 2;


  int numTriggeredPairs = 0;
  double error = 0;
  int countError = 0;

  
  std::array<double, numErrorSensors> greyMeansL;
  std::array<double, numErrorSensors> greyMeansR;


  for (int i = 0; i < numErrorSensors; ++i) {
      auto lPred = Rect(areaMiddleLine - 30 - (i + 1) * sensorWidth, area.y,
                        sensorWidth, sensorHeight);
      auto rPred = Rect(areaMiddleLine + 30 + (i) * sensorWidth, area.y,
                        sensorWidth, sensorHeight);

      double grayMeanL = mean(Mat(origframe, lPred))[0]; // (mean(Mat(origframe, lPred))[0]) < blackSensorThreshold;
      double grayMeanR = mean(Mat(origframe, rPred))[0]; // (mean(Mat(origframe, rPred))[0]) < blackSensorThreshold;
      
      greyMeansL [i] = grayMeanL;
      greyMeansR [i] = grayMeanR;
      double diff = grayMeanL - grayMeanR; // if binary use R - L

      if ( diff > 50 || diff < -50) {
        countError += 1;
        error += diff * sensorWeights[i]; 
        
      putText(
        output, std::to_string((int)grayMeanL),
        Point{lPred.x  + lPred.width / 2 - 5, lPred.y  + lPred.height / 2 + 5},
        FONT_HERSHEY_TRIPLEX, 0.6, {0, 0, 0});
      putText(
        output, std::to_string((int)grayMeanR),
        Point{rPred.x  + rPred.width / 2 - 5, rPred.y  + rPred.height / 2 + 5},
        FONT_HERSHEY_TRIPLEX, 0.6, {0, 0, 0});
      rectangle(output, lPred, Scalar(50, 50, 50));
      rectangle(output, rPred, Scalar(50, 50, 50));
    }
  }
    return error/(255);
} */
   
int main(int, char **) {
  srand(0); //random number generator
  cv::namedWindow("robot view");
  cvui::init(STAT_WINDOW);

  auto statFrame = cv::Mat(400, 600, CV_8UC3);
  initialize_samanet(nPredictors);
  serialib LS; // for arduino
  char Ret = LS.Open(DEVICE_PORT, 115200); // for arduino
  if (Ret != 1) { // If an error occured...
    printf("Error while opening port. Permission problem ?\n"); // ... display a
                                                                // message ...
    return Ret; // ... quit the application
  }
  printf("Serial port opened successfully !\n");
  VideoCapture cap(0); // open the on-board camera. This parameter might need to change. this has to be 0 for RaspberryPi

  if (!cap.isOpened()) {
    printf("The selected video capture device is not available.\n");
    return -1;
  }

  Mat edges;

  std::vector<float> predictorDeltaMeans;
  predictorDeltaMeans.reserve(nPredictors);

  start_time = std::chrono::system_clock::now();

  for (;;) {
    statFrame = cv::Scalar(49, 52, 49);
    predictorDeltaMeans.clear();

    Mat origframe, frame;
    cap >> origframe; // get a new frame from camera

      //flip the image
    flip(origframe,frame,-1); // 0 horizontal, 1 vertical, -1 both

    cvtColor(frame, edges, COLOR_BGR2GRAY);



    // Define the rect area that we want to consider.

    int areaWidth = 600; // 500;
    int areaHeight = 300;
    int offsetFromTop = 50;
    int startX = (frame.cols - areaWidth) / 2;
    auto area = Rect{startX, offsetFromTop, areaWidth, areaHeight};

    int predictorWidth = area.width / 2 / nPredictorCols;
    int predictorHeight = area.height / nPredictorRows;

    rectangle(edges, area, Scalar(122, 144, 255));

    int areaMiddleLine = area.width / 2 + area.x;

    for (int k = 0; k < nPredictorRows; ++k) {
      for (int j = 0; j < nPredictorCols * 2 ; ++j) {
        auto Pred = Rect(area.x + j * predictorWidth, area.y + k * predictorHeight, predictorWidth, predictorHeight);

        auto grayMean = mean(Mat(edges, Pred))[0];
        predictorDeltaMeans.push_back((grayMean) / 255);
        putText(frame, std::to_string((int)grayMean),
                Point{Pred.x + Pred.width / 2 - 13,
                      Pred.y + Pred.height / 2 + 5},
                FONT_HERSHEY_TRIPLEX, 0.4, {0, 0, 0});
      
        rectangle(frame, Pred, Scalar(50, 50, 50));
      }
    }
    
    double sensorError = calculateErrorValue(edges, frame);

    line(frame, {areaMiddleLine, 0}, {areaMiddleLine, frame.rows},
         Scalar(50, 50, 255));
    imshow("robot view", frame);

    /*
    char lightSensor[9]= {'a','a','a','a','a','a','a','a','a'} ;
    double errorSensor[9]= {0,0,0,0,0,0,0,0,0};
    int check[9]= {0,0,0,0,0,0,0,0,0};
  
    LS.Read(&lightSensor, sizeof(lightSensor));

      for (int i = 0; i < 9 ; i++){
        errorSensor[i] = (int)lightSensor[i];
        //cout << i << " check: "<< check[i] << " char: " << lightSensor[i] << " double: " << errorSensor[i] << endl;
        if (errorSensor[i] == 0){
          startIndex = i + 1;
          //cout << "start Index: " << startIndex << endl;
        }
      }

    //cout << "----------------------------" << endl;
    
    double errorSensorshifted[9]= {0,0,0,0,0,0,0,0,0};
    for (int i = 0; i < 9; i++){
      int remainIndex = (startIndex + i) % 9;
      errorSensorshifted[i] = (int)lightSensor[remainIndex];
      //cout <<  " before: " << remainIndex << " " << errorSensor[remainIndex]  << " after: " << i << " " << errorSensorshifted[i] << endl;
      //cout << errorSensorshifted[i] << endl;
    }
        //cout << "----------------------------" << endl;


    
    //plot the sensor values:
    
      double minVal = 30; 
      double maxVal = 170;
      sensor0.push_back(errorSensorshifted[0]); //puts the errors in a buffer for plotting
      sensor0[0] = minVal;
      sensor0[1] = maxVal;
      std::vector<double> sensor_list0(sensor0.begin(), sensor0.end());
      cvui::sparkline(statFrame, sensor_list0, 10, 50, 580, 200, 0xcc0000);
      
      sensor1.push_back(errorSensorshifted[1]); //puts the errors in a buffer for plotting
      sensor1[0] = minVal;
      sensor1[1] = maxVal;
      std::vector<double> sensor_list1(sensor1.begin(), sensor1.end());
      cvui::sparkline(statFrame, sensor_list1, 10, 50, 580, 200, 0xe69138);
      
      sensor2.push_back(errorSensorshifted[2]); //puts the errors in a buffer for plotting
      sensor2[0] = minVal;
      sensor2[1] = maxVal;
      std::vector<double> sensor_list2(sensor2.begin(), sensor2.end());
      cvui::sparkline(statFrame, sensor_list2, 10, 50, 580, 200, 0xf1c232);
      
      sensor3.push_back(errorSensorshifted[3]); //puts the errors in a buffer for plotting
      sensor3[0] = minVal;
      sensor3[1] = maxVal;
      std::vector<double> sensor_list3(sensor3.begin(), sensor3.end());
      cvui::sparkline(statFrame, sensor_list3, 10, 50, 580, 200, 0x6aa84f);
      
      sensor4.push_back(errorSensorshifted[4]); //puts the errors in a buffer for plotting
      sensor4[0] = minVal;
      sensor4[1] = maxVal;
      std::vector<double> sensor_list4(sensor4.begin(), sensor4.end());
      cvui::sparkline(statFrame, sensor_list4, 10, 50, 580, 200, 0x45818e);
      
      sensor5.push_back(errorSensorshifted[5]); //puts the errors in a buffer for plotting
      sensor5[0] = minVal;
      sensor5[1] = maxVal;
      std::vector<double> sensor_list5(sensor5.begin(), sensor5.end());
      cvui::sparkline(statFrame, sensor_list5, 10, 50, 580, 200, 0x674ea7);
      
      sensor6.push_back(errorSensorshifted[6]); //puts the errors in a buffer for plotting
      sensor6[0] = minVal;
      sensor6[1] = maxVal;
      std::vector<double> sensor_list6(sensor6.begin(), sensor6.end());
      cvui::sparkline(statFrame, sensor_list6, 10, 50, 580, 200, 0xa64d79);
      
      sensor7.push_back(errorSensorshifted[7]); //puts the errors in a buffer for plotting
      sensor7[0] = minVal;
      sensor7[1] = maxVal;
      std::vector<double> sensor_list7(sensor7.begin(), sensor7.end());
      cvui::sparkline(statFrame, sensor_list7, 10, 50, 580, 200, 0x3c78d8);

      */
   
      int16_t speedError = 10 * onStepCompleted(statFrame, sensorError, predictorDeltaMeans);
      Ret = LS.Write(&speedError, sizeof(speedError));
      //cout<<"speed error is: "<< speedError <<endl; 

    cvui::update();

    // Show everything on the screen
    cv::imshow(STAT_WINDOW, statFrame);
    if (waitKey(20) == ESC_key)
      break;
  }\

  //save_samanet();
  return 0;
}
