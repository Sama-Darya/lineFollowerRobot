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

using namespace cv;
using namespace std;
constexpr int ESC_key = 27;

static constexpr int nPredictorCols = 6;
static constexpr int nPredictorRows = 8;
static constexpr int nPredictors = nPredictorCols * nPredictorRows;

double errorMult = 1.5;
double nnMult = 10.0;

std::ofstream datafs("data.csv");

using clk = std::chrono::system_clock;
clk::time_point start_time;

boost::circular_buffer<double> prevErrors(30 * 60);

int16_t onStepCompleted(cv::Mat &statFrame, double deltaSensorData,
                        std::vector<float> &predictorDeltas) {
  prevErrors.push_back(deltaSensorData); //puts the errors in a buffer

  double errorGain = 5;
  double error = errorGain * deltaSensorData;

  int gain = 15;

  cvui::text(statFrame, 10, 320, "Sensor Error Multiplier: ");
  cvui::trackbar(statFrame, 180, 300, 400, &errorMult, (double)0.0, (double)5.0,
                 1, "%.2Lf", 0, 0.05);

  cvui::text(statFrame, 10, 370, "Net Output Multiplier: ");
  cvui::trackbar(statFrame, 180, 350, 400, &nnMult, (double)0.0, (double)10.0,
                 1, "%.2Lf", 0, 0.05);

  double result = run_samanet(statFrame, predictorDeltas, deltaSensorData / 5); //does one learning iteration

  cvui::text(statFrame, 220, 10, "Net out:");
  cvui::printf(statFrame, 300, 10, "%+.4lf", result);

  cvui::text(statFrame, 220, 30, "Error:");
  cvui::printf(statFrame, 300, 30, "%+.4lf", deltaSensorData);

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
  double error2 = (error * errorMult + result * nnMult) * gain;
  int16_t differentialOut = (int16_t)(error2 * 0.5);

  using namespace std::chrono;
  milliseconds ms =
      duration_cast<milliseconds>(system_clock::now().time_since_epoch());

  datafs << ms.count() << ","        // timestamp
         << deltaSensorData << ","   // error from error units
         << result << ","            // net output
         << differentialOut << "\n"; // final differential output

  return differentialOut;
}

#if defined(_WIN32) || defined(_WIN64)
#define DEVICE_PORT "COM4" // COM1 for windows
#endif

#ifdef __linux__
#define DEVICE_PORT "/dev/ttyS0" // ttyS0 for linux, otherwise ttyUSB0
#endif

double calculateErrorValue(Mat &frame, Mat &output) {
  constexpr int numErrorSensors = 5;
  int areaWidth = 400;
  int areaHeight = 30;
  int offsetFromBottom = 0;
  int whiteSensorThreshold = 190;
  int startX = (frame.cols - areaWidth) / 2;
  auto area = Rect{startX, frame.rows - areaHeight - offsetFromBottom,
                   areaWidth, areaHeight};

  int areaMiddleLine = area.width / 2 + area.x;

  int sensorWidth = area.width / 2 / numErrorSensors;
  int sensorHeight = areaHeight;

  std::array<double, numErrorSensors> sensorWeights;

  sensorWeights[numErrorSensors - 1] = 1;
  for (int j = numErrorSensors - 2; j >= 0; --j) {
    sensorWeights[j] = sensorWeights[j + 1] * 0.60;
  }

  int numTriggeredPairs = 0;
  double error = 0;

  for (int j = 0; j < numErrorSensors; ++j) {
    auto lPred = Rect(areaMiddleLine - 30 - (j + 1) * sensorWidth, area.y,
                      sensorWidth, sensorHeight);
    auto rPred = Rect(areaMiddleLine + 30 + (j)*sensorWidth, area.y,
                      sensorWidth, sensorHeight);

    double grayMeanL = (mean(Mat(frame, lPred))[0]) > whiteSensorThreshold;
    double grayMeanR = (mean(Mat(frame, rPred))[0]) > whiteSensorThreshold;

    auto diff = (grayMeanR - grayMeanL);
    numTriggeredPairs += (diff != 0);

    error += diff * sensorWeights[j];

    putText(
        output, std::to_string((int)grayMeanL),
        Point{lPred.x + lPred.width / 2 - 5, lPred.y + lPred.height / 2 + 5},
        FONT_HERSHEY_TRIPLEX, 0.6, {0, 0, 0});
    putText(
        output, std::to_string((int)grayMeanR),
        Point{rPred.x + rPred.width / 2 - 5, rPred.y + rPred.height / 2 + 5},
        FONT_HERSHEY_TRIPLEX, 0.6, {0, 0, 0});
    rectangle(output, lPred, Scalar(50, 50, 50));
    rectangle(output, rPred, Scalar(50, 50, 50));
  }

  return numTriggeredPairs ? error / numTriggeredPairs : 0;
}

#define STAT_WINDOW "statistics & options"
int main(int, char **) {
  srand(0);
  cv::namedWindow("robot view");
  cvui::init(STAT_WINDOW);

  auto statFrame = cv::Mat(400, 600, CV_8UC3);
  initialize_samanet(nPredictors);
  serialib LS;
  char Ret = LS.Open(DEVICE_PORT, 115200);
  if (Ret != 1) { // If an error occured...
    printf("Error while opening port. Permission problem ?\n"); // ... display a
                                                                // message ...
    return Ret; // ... quit the application
  }
  printf("Serial port opened successfully !\n");
  VideoCapture cap(
      2); // open the on-board camera. This parameter might need to change.

  if (!cap.isOpened()) {
    printf("The selected video capture device is not available.\n");
    return -1;
  }

  Mat edges;

  std::vector<float> predictorDeltaMeans;
  predictorDeltaMeans.reserve(nPredictorCols * nPredictorRows);

  start_time = std::chrono::system_clock::now();

  for (;;) {
    statFrame = cv::Scalar(49, 52, 49);
    predictorDeltaMeans.clear();

    Mat frame;
    cap >> frame; // get a new frame from camera
    cvtColor(frame, edges, COLOR_BGR2GRAY);

    // Define the rect area that we want to consider.

    int areaWidth = 400; // 500;
    int areaHeight = 200;
    int offsetFromTop = 150;
    int startX = (frame.cols - areaWidth) / 2;
    auto area = Rect{startX, offsetFromTop, areaWidth, areaHeight};

    int predictorWidth = area.width / 2 / nPredictorCols;
    int predictorHeight = area.height / nPredictorRows;

    rectangle(edges, area, Scalar(122, 144, 255));

    int areaMiddleLine = area.width / 2 + area.x;

    for (int k = 0; k < nPredictorRows; ++k) {
      for (int j = 0; j < nPredictorCols; ++j) {
        auto lPred =
            Rect(areaMiddleLine - (j + 1) * predictorWidth,
                 area.y + k * predictorHeight, predictorWidth, predictorHeight);
        auto rPred =
            Rect(areaMiddleLine + (j)*predictorWidth,
                 area.y + k * predictorHeight, predictorWidth, predictorHeight);

        auto grayMeanL = mean(Mat(edges, lPred))[0];
        auto grayMeanR = mean(Mat(edges, rPred))[0];
        predictorDeltaMeans.push_back((grayMeanL - grayMeanR) / 255);
        putText(frame, std::to_string((int)grayMeanL),
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

    double err = calculateErrorValue(edges, frame);

    line(frame, {areaMiddleLine, 0}, {areaMiddleLine, frame.rows},
         Scalar(50, 50, 255));
    imshow("robot view", frame);

    int8_t ping = 0;
    Ret = LS.Read(&ping, sizeof(ping));

    if (Ret > 0) {

      int16_t error = onStepCompleted(statFrame, err, predictorDeltaMeans);

      Ret = LS.Write(&error, sizeof(error));
    }

    cvui::update();

    // Show everything on the screen
    cv::imshow(STAT_WINDOW, statFrame);
    if (waitKey(20) == ESC_key)
      break;
  }

  dump_samanet();
  return 0;
}