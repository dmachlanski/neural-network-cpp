#include <iostream>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

#pragma once
class TwoLayerNN
{
public:
	TwoLayerNN();
	void InitializeModel(string inputDataPath, string outputDataPath, int hiddenUnits, bool normalizeBeforeSplit, bool isOutputLinear, double lambda);
	void LoadModel(string path);
	void SaveModel(string path, bool saveBestWeights);
	void Train(double learningRate, double momentum, int batchSize, int epochs, bool earlyStopping);
	MatrixXd Predict(MatrixXd input, bool unNormalizeOutput);
	void TestNormalization(MatrixXd input);

private:
	MatrixXd FeedForward(MatrixXd input, bool isTraining, bool isOutputLinear);
	void Backprop(MatrixXd X, MatrixXd Y, MatrixXd Yhat);
	void SplitDataSet(MatrixXd x, MatrixXd y, double trainRatio, double validRatio, double testRatio);
	void UpdateBestWeights();
	void UseBestWeights();

	int inputSize, hiddenSize, outputSize;
	double lambda;
	bool isOutputLinear;
	MatrixXd X_train, Y_train, X_valid, Y_valid, X_test, Y_test, W1, W2, gradW1, gradW2, A1, A2, bestW1, bestW2, dW1, dW2;
	VectorXd b1, b2, gradB1, gradB2, muX, sigmaX, muY, sigmaY, JEpoch, ValidEpoch, bestB1, bestB2, db1, db2;
};