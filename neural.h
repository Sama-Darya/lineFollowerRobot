#pragma once
#include <vector>

namespace cv {
class Mat;
}

void initialize_samanet(int numInputLayers, float sampleRate = 30.f);
double run_samanet(cv::Mat &statFrame, std::vector<float> &in, double error);
void dump_samanet();
