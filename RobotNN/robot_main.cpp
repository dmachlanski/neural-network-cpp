#include "aria.h"
#include <algorithm>
#include <iostream>
#include <Eigen/Dense>
#include <fstream>
#include <math.h>

using namespace std;
using namespace Eigen;

double kp = 0.6;

double desiredDistance = 400;
double e = 0.0;
double ei = 0.0;
double ed = 0.0;
double ePrev = 0.0;
double baseVel = 100;

MatrixXd Sigmoid(MatrixXd m)
{
	MatrixXd result(m.rows(), m.cols());

	for (int i = 0; i < m.rows(); i++)
	{
		for (int j = 0; j < m.cols(); j++)
		{
			result(i, j) = exp(-m(i, j));
		}
	}

	return (1.0 / (1.0 + result.array())).matrix();
}

MatrixXd FeedForward(MatrixXd X, MatrixXd W1, VectorXd b1, MatrixXd W2, VectorXd b2, MatrixXd &A1, MatrixXd &Z1)
{
	Z1 = (W1 * X).colwise() + b1;
	A1 = Sigmoid(Z1);

	return ((W2 * A1).colwise() + b2);
}

MatrixXd Normalize(VectorXd input, VectorXd mean, VectorXd range)
{
	return ((input.array() - mean.array()) / range.array()).matrix();
}

MatrixXd UnNormalize(VectorXd input, VectorXd mean, VectorXd range)
{
	return ((input.array() * range.array()) + range.array()).matrix();
}

MatrixXd FileToMatrix(string path)
{
	ifstream file(path);

	string data;
	double value;

	getline(file, data, ',');
	int rows = stoi(data);

	getline(file, data);
	int cols = stoi(data);

	MatrixXd m(rows, cols);

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			if (j < (cols - 1))
			{
				getline(file, data, ',');
			}
			else
			{
				getline(file, data);
			}

			value = stod(data);
			m(i, j) = value;
		}
	}

	file.close();

	return m;
}

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

	// Init the NN (read the weights)
	MatrixXd W1 = FileToMatrix("W1.csv");
	MatrixXd b1 = FileToMatrix("b2.csv");
	MatrixXd W2 = FileToMatrix("W2.csv");
	MatrixXd b2 = FileToMatrix("b2.csv");
	MatrixXd mean_X = FileToMatrix("Mean_X.csv");
	MatrixXd range_X = FileToMatrix("Range_X.csv");
	MatrixXd mean_Y = FileToMatrix("Mean_Y.csv");
	MatrixXd range_Y = FileToMatrix("Range_Y.csv");

	VectorXd input(2, 1);
	MatrixXd input_norm, A1, Z1, output_NN, output_unnorm;

	while (true) {
		// run 
		// add to initialisation −> create instances 
		// add to run 
		double first = robot.getSonarReading(1)->getRange();
		double second = robot.getSonarReading(0)->getRange();

		input(0, 0) = first;
		input(1, 0) = second;

		input_norm = Normalize(input, mean_X, range_X);

		output_NN = FeedForward(input_norm, W1, b1, W2, b2, A1, Z1);

		output_unnorm = UnNormalize(output_NN, mean_Y, range_Y);

		// Use the NN instead of "pid"
		//double* output = pid(first, second);

		robot.setVel2(output_unnorm(0,0), output_unnorm(1,0));

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