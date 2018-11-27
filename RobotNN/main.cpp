#include "aria.h"
#include <algorithm>
#include <iostream>
#include <Eigen/Dense>
#include <fstream>
#include <math.h>
#include "Helpers.h"
#include "TwoLayerNN.h"

using namespace std;
using namespace Eigen;

double kp = 0.6;

double desiredDistance = 400;
double e = 0.0;
double ei = 0.0;
double ed = 0.0;
double ePrev = 0.0;
double baseVel = 100;

double* pid(double first, double second) {
	double* output = new double[2];

	double minDist = first < second ? first : second;
	e = desiredDistance - minDist;
	ei = ei + e;
	ed = e - ePrev;

	ePrev = e;

	double val = kp * e;

	val = -2 * val / 260;

	output[0] = baseVel - val * 260 / 2;
	output[1] = baseVel + val * 260 / 2;

	return output;
}

int main(int argc, char **argv) {
	// Initialisations 

	// create instances 
	Aria::init();
	ArRobot robot;
	ArPose pose;
	// parse command line arguments 
	ArArgumentParser argParser(&argc, argv);
	argParser.loadDefaultArguments();

	// connect to robot (and laser, etc) 
	ArRobotConnector robotConnector(&argParser, &robot);
	if (robotConnector.connectRobot())
		std::cout << "Robot connected!" << std::endl;
	robot.runAsync(false);
	robot.lock();
	robot.enableMotors();
	robot.unlock();

	TwoLayerNN model;
	model.LoadModel("../output/");

	VectorXd input(2, 1);

	while (true) {
		// run 
		// add to initialisation −> create instances 
		// add to run 
		double first = robot.getSonarReading(0)->getRange();
		double second = robot.getSonarReading(3)->getRange();

		input(0, 0) = first;
		input(1, 0) = second;

		VectorXd output = model.Predict(input, true);

		// Use the NN instead of "pid"
		//double* output = pid(first, second);

		robot.setVel2(output(0, 0), output(1, 0));

		ArUtil::sleep(100);
	}
	// termination 
	// stop the robot 
	robot.lock();
	robot.stop();
	robot.unlock();
	// terminate all threads and exit
	Aria::exit();
	return 0;
}