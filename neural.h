#pragma once
#include <vector>


namespace cv {
class Mat;
}

#define RAW_PRED
void initialize_samanet(int numInputLayers, double sampleRate = 30.f);
void run_samanet(cv::Mat &statFrame, std::vector<double> &in, double error);
void save_samanet();
double getResults(int returnCase);

