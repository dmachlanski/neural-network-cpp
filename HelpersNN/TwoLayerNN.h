#include <iostream>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

#pragma once
class TwoLayerNN
{
public:
	TwoLayerNN();
	void InitializeModel(string inputDataPath, string outputDataPath, int hiddenUnits, bool normalizeOutput);
	void LoadModel(string path);
	void SaveModel(string path, bool saveBestWeights);
	void Train(double learningRate, int batchSize, int epochs, double altStop);
	MatrixXd Predict(MatrixXd input, bool unNormalizeOutput);
	void TestNormalization(MatrixXd input);

private:
	MatrixXd FeedForward(MatrixXd input, bool isTraining);
	void Backprop(MatrixXd X, MatrixXd Y, MatrixXd Yhat);
	void SplitDataSet(MatrixXd x, MatrixXd y, double trainRatio, double validRatio, double testRatio);
	void UpdateBestWeights();

	int inputSize, hiddenSize, outputSize;
	bool normalizeOutput;
	MatrixXd X_train, Y_train, X_valid, Y_valid, X_test, Y_test, W1, W2, dW1, dW2, A1, bestW1, bestW2;
	VectorXd b1, b2, db1, db2, muX, sigmaX, muY, sigmaY, J, Valid, bestB1, bestB2;
};