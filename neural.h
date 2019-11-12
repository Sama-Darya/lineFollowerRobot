#pragma once
#include <vector>


namespace cv {
class Mat;
}

void initialize_samanet(int numInputLayers, float sampleRate = 30.f);

float run_samanet(cv::Mat &statFrame, std::vector<float> &in, float error);
void save_samanet();
