#include <iostream>
#include <Eigen/Dense>
#include "TwoLayerNN.h"

using namespace std;
using namespace Eigen;

int main()
{
	const string DATA_PATH = "../data/";
	const string OUTPUT_PATH = "../output/";

	string inputPath = DATA_PATH + "input.csv";
	string outputPath = DATA_PATH + "output.csv";

	TwoLayerNN model;
	
	model.InitializeModel(inputPath, outputPath, 4, true, false);
	
	model.Train(0.2, 0.9, 1, 100, -1);
	
	model.SaveModel(OUTPUT_PATH, true);
	
	// Simple test
	MatrixXd test(2, 1);
	test(0, 0) = 547.508;
	test(1, 0) = 2151.29;
	
	MatrixXd result = model.Predict(test, true);
	
	// Expected:
	// 115
	// 140.054
	cout << result << endl << endl;

	return 0;
}