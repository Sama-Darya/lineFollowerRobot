#include <Servo.h>

Servo servoLeft;
Servo servoRight;

bool computer = true;
bool onlysensors = false;

float ErrorIn = 0.00; // Error from sensors
float ErrorOut = 0.00;
float sensorLeft = 0;
float sensorRight = 0;
int motorStopL = 1500; // Motor stop uS
int motorStopR = 1520; // Motor stop uS
int motorSpeed = 50; // Constant fwd speed
int errorMulti = 100; // Error multiplier
int speedLeft = motorStopL + motorSpeed; // Left constant speed
int speedRight = motorStopR - motorSpeed; // Right constant speed
String received;

char Buffer[32];
int myTimeout = 50;
void setup() {
	pinMode(LED_BUILTIN, OUTPUT);
	Serial.begin(115200); // opens serial port, sets data rate to 115200 bps
	Serial.setTimeout(myTimeout);
	//Serial.print("Arduino is ready \n");
	servoLeft.attach(9);
	servoRight.attach(6);

	servoLeft.writeMicroseconds(speedLeft);
	servoRight.writeMicroseconds(speedRight);
	}

void lineSensors() {
	pinMode(A1, OUTPUT); // Set sensor pins as output
	pinMode(A5, OUTPUT);
	digitalWrite(A1, HIGH); // Set sensor pins high
	digitalWrite(A5, HIGH);
	delay(1); // Charge capacitor
	pinMode(A1, INPUT); // Set sensor pins as input
	pinMode(A5, INPUT);
	delay(1); // Discharge capacitor
} 


int prevError = 0;

void loop() {
	//delay(100);
	//lineSensors();
	//sensorLeft = digitalRead(A1); // Read sensor values (1,black 0,white)
	//sensorRight = digitalRead(A5);

   // int8_t deltaError = sensorLeft - sensorRight;
  int8_t deltaError = 0;
	Serial.write((uint8_t *)&deltaError, sizeof(deltaError));

	while(Serial.available() > 0) {
		//received = Serial.readString();
			int16_t error = 0;
			/////// Receive and set motor speed
			Serial.readBytes((uint8_t *)&error, sizeof(int16_t));
			
		/*	if (error == 50) {
				digitalWrite(LED_BUILTIN, HIGH);
				delay(100);
				digitalWrite(LED_BUILTIN, LOW);
			}*/

			if (prevError != error) {
				servoLeft.writeMicroseconds(speedLeft + error);
				servoRight.writeMicroseconds(speedRight + error);
				prevError = error;
			}
	}
}

