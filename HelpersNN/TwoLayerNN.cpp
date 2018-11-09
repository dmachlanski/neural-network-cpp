#include "TwoLayerNN.h"
#include "Helpers.h"
#include <fstream>

TwoLayerNN::TwoLayerNN()
{
}

#pragma region Private methods
VectorXd TwoLayerNN::Train(MatrixXd X, MatrixXd Y, MatrixXd &W1, VectorXd &b1, MatrixXd &W2, VectorXd &b2, double alpha, int epochs)
{
	VectorXd J = VectorXd::Zero(epochs);

	for (int i = 0; i < epochs; i++)
	{
		// Main training loop
		MatrixXd A1, Z1, dW1, dW2;
		VectorXd db1, db2;

		MatrixXd Yhat = FeedForward(X, W1, b1, W2, b2, A1, Z1);
		J(i) = Helpers::MeanSquaredError(Y, Yhat);
		Backprop(X, Y, Yhat, A1, Z1, W2, dW1, db1, dW2, db2);

		// Apply gradients
		W1 -= (alpha * dW1.array()).matrix();
		b1 -= (alpha * db1.array()).matrix();
		W2 -= (alpha * dW2.array()).matrix();
		b2 -= (alpha * db2.array()).matrix();

		cout << "Epoch " << i << ", loss: " << J(i) << endl;
	}

	return J;
}

MatrixXd TwoLayerNN::FeedForward(MatrixXd X, MatrixXd W1, VectorXd b1, MatrixXd W2, VectorXd b2, MatrixXd &A1, MatrixXd &Z1)
{
	Z1 = (W1 * X).colwise() + b1;
	A1 = Helpers::Sigmoid(Z1);

	return ((W2 * A1).colwise() + b2);
}

void TwoLayerNN::Backprop(MatrixXd X, MatrixXd Y, MatrixXd Yhat, MatrixXd A1, MatrixXd Z1, MatrixXd W2, MatrixXd &dW1, VectorXd &db1, MatrixXd &dW2, VectorXd &db2)
{
	double m = X.cols();

	MatrixXd dZ2 = Yhat - Y;
	dW2 = ((dZ2 * A1.transpose()).array() / m).matrix();
	db2 = (dZ2.rowwise().sum().array() / m).matrix();
	MatrixXd dZ1 = ((W2.transpose() * dZ2).array() * (Helpers::Sigmoid(Z1).array() * (1.0 - Helpers::Sigmoid(Z1).array()))).matrix();
	dW1 = ((dZ1 * X.transpose()).array() / m).matrix();
	db1 = (dZ1.rowwise().sum().array() / m).matrix();
}

#pragma endregion