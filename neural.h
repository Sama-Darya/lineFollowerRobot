#pragma once
#include <vector>


namespace cv {
class Mat;
}

void initialize_samanet(int numInputLayers, double sampleRate = 30.f);

double run_samanet(cv::Mat &statFrame, std::vector<double> &in, double error);
void save_samanet();
