#include <iostream>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

#pragma once
class TwoLayerNN
{
public:
	TwoLayerNN();
	void InitializeModel(string inputPath, string outputPath);
	void LoadModel();
	void SaveModel();

private:
	VectorXd Train(MatrixXd X, MatrixXd Y, MatrixXd &W1, VectorXd &b1, MatrixXd &W2, VectorXd &b2, double alpha, int epochs);
	MatrixXd FeedForward(MatrixXd X, MatrixXd W1, VectorXd b1, MatrixXd W2, VectorXd b2, MatrixXd &A1, MatrixXd &Z1);
	void Backprop(MatrixXd X, MatrixXd Y, MatrixXd Yhat, MatrixXd A1, MatrixXd Z1, MatrixXd W2, MatrixXd &dW1, VectorXd &db1, MatrixXd &dW2, VectorXd &db2);

	int hidden_units;
	MatrixXd X_train, Y_train, W1, W2, dW1, dW2, A1, Z1;
	VectorXd b1, b2, db1, db2;
};