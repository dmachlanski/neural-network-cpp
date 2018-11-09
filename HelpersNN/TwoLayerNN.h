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
	void Train(double learningRate, int epochs);
	MatrixXd Predict(MatrixXd input);
	void TestNormalization(MatrixXd input);

private:
	MatrixXd FeedForward(MatrixXd input);
	void Backprop(MatrixXd Yhat);

	int inputSize, hiddenSize, outputSize;
	MatrixXd X_train, Y_train, W1, W2, dW1, dW2, A1, Z1;
	VectorXd b1, b2, db1, db2, muX, sigmaX, muY, sigmaY, J;
};