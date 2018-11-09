#include <iostream>
#include <Eigen/Dense>
#include <fstream>
#include <math.h>

using namespace std;
using namespace Eigen;

int main()
{
	const string DATA_PATH = "../data/";
	const string OUTPUT_PATH = "../output/";

	// Load X and Y matrixes
	//MatrixXd input = FileToMatrix(DATA_PATH + "input.csv");
	//MatrixXd output = FileToMatrix(DATA_PATH + "output.csv");

	// Find their mean and range needed for normalization
	//VectorXd mean_X, range_X, mean_Y, range_Y;
	//FindNormParams(input, mean_X, range_X);
	//FindNormParams(output, mean_Y, range_Y);

	// Normalize X and Y
	//MatrixXd X_norm = Normalize(input, mean_X, range_X);
	//MatrixXd Y_norm = Normalize(output, mean_Y, range_Y);

	// Set NN structure
	//int input_size = input.rows();
	//int hidden_size = 10;
	//int output_size = output.rows();

	// Initilize weights
	//MatrixXd W1 = MatrixXd::Random(hidden_size, input_size);
	//VectorXd b1 = VectorXd::Zero(hidden_size);
	//MatrixXd W2 = MatrixXd::Random(output_size, hidden_size);
	//VectorXd b2 = VectorXd::Zero(output_size);

	//double alpha = 0.03;
	//int epochs = 10;

	// Train the model
	//VectorXd J = Train(X_norm, Y_norm, W1, b1, W2, b2, alpha, epochs);

	// Save weights, loss and normalization data
	//MatrixToFile(OUTPUT_PATH + "W1.csv", W1);
	//MatrixToFile(OUTPUT_PATH + "b1.csv", b1);
	//MatrixToFile(OUTPUT_PATH + "W2.csv", W2);
	//MatrixToFile(OUTPUT_PATH + "b2.csv", b2);
	//MatrixToFile(OUTPUT_PATH + "MeanX.csv", mean_X);
	//MatrixToFile(OUTPUT_PATH + "RangeX.csv", range_X);
	//MatrixToFile(OUTPUT_PATH + "MeanY.csv", mean_Y);
	//MatrixToFile(OUTPUT_PATH + "RangeY.csv", range_Y);
	//MatrixToFile(OUTPUT_PATH + "J.csv", J, false);

	return 0;
}