#include "TwoLayerNN.h"
#include "Helpers.h"
#include <fstream>

TwoLayerNN::TwoLayerNN()
{
}

#pragma region Public methods

void TwoLayerNN::InitializeModel(string inputDataPath, string outputDataPath, int hiddenUnits)
{
	// Load the data
	MatrixXd X = Helpers::FileToMatrix(inputDataPath);
	MatrixXd Y = Helpers::FileToMatrix(outputDataPath);

	// TODO: Divide into train/valid/test sets
	X_train = X;
	Y_train = Y;

	// Set the sizes for all the matrixes
	inputSize = X.rows();
	hiddenSize = hiddenUnits;
	outputSize = Y.rows();

	// Initialize weights
	W1 = MatrixXd::Random(hiddenSize, inputSize);
	b1 = VectorXd::Zero(hiddenSize);
	W2 = MatrixXd::Random(outputSize, hiddenSize);
	b2 = VectorXd::Zero(outputSize);

	// Find normalization params
	Helpers::FindNormParams(X_train, muX, sigmaX);
	Helpers::FindNormParams(Y_train, muY, sigmaY);

	// Normalize the data
	X_train = Helpers::Normalize(X_train, muX, sigmaX);
	Y_train = Helpers::Normalize(Y_train, muY, sigmaY);
}

void TwoLayerNN::LoadModel(string path)
{
	W1 = Helpers::FileToMatrix(path + "W1.csv");
	b1 = Helpers::FileToMatrix(path + "b1.csv");
	W2 = Helpers::FileToMatrix(path + "W2.csv");
	b2 = Helpers::FileToMatrix(path + "b2.csv");
	muX = Helpers::FileToMatrix(path + "muX.csv");
	sigmaX = Helpers::FileToMatrix(path + "sigmaX.csv");
	muY = Helpers::FileToMatrix(path + "muY.csv");
	sigmaY = Helpers::FileToMatrix(path + "sigmaY.csv");
}

void TwoLayerNN::SaveModel(string path)
{
	Helpers::MatrixToFile(path + "W1.csv", W1);
	Helpers::MatrixToFile(path + "b1.csv", b1);
	Helpers::MatrixToFile(path + "W2.csv", W2);
	Helpers::MatrixToFile(path + "b2.csv", b2);
	Helpers::MatrixToFile(path + "muX.csv", muX);
	Helpers::MatrixToFile(path + "sigmaX.csv", sigmaX);
	Helpers::MatrixToFile(path + "muY.csv", muY);
	Helpers::MatrixToFile(path + "sigmaY.csv", sigmaY);
	Helpers::MatrixToFile(path + "J.csv", J, false);
}

void TwoLayerNN::Train(int learningRate, int epochs)
{
	J = VectorXd::Zero(epochs);

	for (int i = 0; i < epochs; i++)
	{
		// Main training loop
		MatrixXd Yhat = FeedForward();
		J(i) = Helpers::MeanSquaredError(Y_train, Yhat);
		Backprop(Yhat);

		// Apply gradients
		W1 -= (learningRate * dW1.array()).matrix();
		b1 -= (learningRate * db1.array()).matrix();
		W2 -= (learningRate * dW2.array()).matrix();
		b2 -= (learningRate * db2.array()).matrix();

		cout << "Epoch " << i << ", loss: " << J(i) << endl;
	}
}

#pragma endregion


#pragma region Private methods

MatrixXd TwoLayerNN::FeedForward()
{
	Z1 = (W1 * X_train).colwise() + b1;
	A1 = Helpers::Sigmoid(Z1);

	return ((W2 * A1).colwise() + b2);
}

void TwoLayerNN::Backprop(MatrixXd Yhat)
{
	double m = X_train.cols();

	MatrixXd dZ2 = Yhat - Y_train;
	dW2 = ((dZ2 * A1.transpose()).array() / m).matrix();
	db2 = (dZ2.rowwise().sum().array() / m).matrix();
	MatrixXd dZ1 = ((W2.transpose() * dZ2).array() * (Helpers::Sigmoid(Z1).array() * (1.0 - Helpers::Sigmoid(Z1).array()))).matrix();
	dW1 = ((dZ1 * X_train.transpose()).array() / m).matrix();
	db1 = (dZ1.rowwise().sum().array() / m).matrix();
}

#pragma endregion