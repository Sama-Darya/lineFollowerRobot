
#ifndef _LowPassFilter_hpp_
#define _LowPassFilter_hpp_

#include <iostream>
#include <cmath>

class LowPassFilter{
public:
	//constructors
	LowPassFilter();
	LowPassFilter(float iCutOffFrequency);
	LowPassFilter(float iCutOffFrequency, float iDeltaTime);
	//functions
	float update(float input);
	float update(float input, float deltaTime);
	//get and set funtions
	float getOutput();
	float getCutOffFrequency();
	void setCutOffFrequency(float input);
	void setDeltaTime(float input);
private:
	float output = 0;
	float cutOffFrequency = 0;
	float ePow = 0;
};

#endif //_LowPassFilter_hpp_
