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

	model.InitializeModel(inputPath, outputPath, 15, false);

	model.Train(0.02, 10, 1, 500.0);

	model.SaveModel(OUTPUT_PATH, true);

	return 0;
}