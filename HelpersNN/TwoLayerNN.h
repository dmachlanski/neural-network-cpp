#include <iostream>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

#pragma once
class TwoLayerNN
{
public:
	TwoLayerNN();
	void InitializeModel(string inputDataPath, string outputDataPath, int hiddenUnits);
	void LoadModel(string path);
	void SaveModel(string path);
	void Train(int learningRate, int epochs);

private:
	MatrixXd FeedForward();
	void Backprop(MatrixXd Yhat);

	int inputSize, hiddenSize, outputSize;
	MatrixXd X_train, Y_train, W1, W2, dW1, dW2, A1, Z1;
	VectorXd b1, b2, db1, db2, muX, sigmaX, muY, sigmaY, J;
};