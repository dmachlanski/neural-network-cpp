#include <iostream>
#include <Eigen/Dense>
#include "TwoLayerNN.h"

using namespace std;
using namespace Eigen;

int main()
{
	// Expected outputs:
	// - 115
	// - 140.054
	double input1 = 547.508, input2 = 2151.29;

	VectorXd input_raw(2, 1);
	input_raw(0, 0) = input1;
	input_raw(1, 0) = input2;

	cout << input_raw << endl << endl;

	TwoLayerNN model;

	model.LoadModel("../output/");

	MatrixXd result = model.Predict(input_raw, true);
	
	cout << result << endl;

	return 0;
}