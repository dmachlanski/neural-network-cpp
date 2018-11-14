#include "TwoLayerNN.h"
#include "Helpers.h"
#include <fstream>

TwoLayerNN::TwoLayerNN()
{
}

#pragma region Public methods

void TwoLayerNN::InitializeModel(string inputDataPath, string outputDataPath, int hiddenUnits, bool normalizeOutput)
{
	TwoLayerNN::normalizeOutput = normalizeOutput;

	// Load the data
	MatrixXd X = Helpers::FileToMatrix(inputDataPath);
	MatrixXd Y = Helpers::FileToMatrix(outputDataPath);

	//Helpers::FindNormParams(X, muX, sigmaX);
	//Helpers::FindNormParams(Y, muY, sigmaY);
	//X = Helpers::Normalize(X, muX, sigmaX);
	//Y = Helpers::Normalize(Y, muY, sigmaY);

	// Split the data (70/15/15)
	SplitDataSet(X, Y, 0.7, 0.15, 0.15);

	// Set the sizes for all the matrixes
	inputSize = X.rows();
	hiddenSize = hiddenUnits;
	outputSize = Y.rows();

	// Initialize weights
	W1 = MatrixXd::Random(hiddenSize, inputSize);
	b1 = VectorXd::Zero(hiddenSize);
	W2 = MatrixXd::Random(outputSize, hiddenSize);
	b2 = VectorXd::Zero(outputSize);

	dW1 = MatrixXd::Zero(hiddenSize, inputSize);
	db1 = VectorXd::Zero(hiddenSize);
	dW2 = MatrixXd::Zero(outputSize, hiddenSize);
	db2 = VectorXd::Zero(outputSize);

	// Find normalization params
	Helpers::FindNormParams(X_train, muX, sigmaX);
	Helpers::FindNormParams(Y_train, muY, sigmaY);
	X_train = Helpers::Normalize(X_train, muX, sigmaX);

	if (normalizeOutput)
	{
		Y_train = Helpers::Normalize(Y_train, muY, sigmaY);
	}
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

void TwoLayerNN::SaveModel(string path, bool saveBestWeights)
{
	if (saveBestWeights)
	{
		W1 = bestW1;
		b1 = bestB1;
		W2 = bestW2;
		b2 = bestB2;
	}

	Helpers::MatrixToFile(path + "W1.csv", W1);
	Helpers::MatrixToFile(path + "b1.csv", b1);
	Helpers::MatrixToFile(path + "W2.csv", W2);
	Helpers::MatrixToFile(path + "b2.csv", b2);
	Helpers::MatrixToFile(path + "muX.csv", muX);
	Helpers::MatrixToFile(path + "sigmaX.csv", sigmaX);
	Helpers::MatrixToFile(path + "muY.csv", muY);
	Helpers::MatrixToFile(path + "sigmaY.csv", sigmaY);
	Helpers::MatrixToFile(path + "J.csv", J, false);
	Helpers::MatrixToFile(path + "Valid.csv", Valid, false);
}

void TwoLayerNN::Train(double learningRate, double momentum, int batchSize, int epochs, double altStop, int printOn)
{
	int m = X_train.cols();
	int iterations = m / batchSize;
	int all = iterations * batchSize;

	if (all < m)
	{
		iterations++;
	}

	J = VectorXd::Zero(epochs * iterations);
	Valid = VectorXd::Zero(epochs * iterations);
	double bestValid = DBL_MAX;
	int bestValidIndex;
	int loopIndex = 0;

	for (int i = 0; i < epochs; i++)
	{
		for (int j = 0; j < iterations; j++)
		{
			int start = j * batchSize;
			int end = start + batchSize - 1;

			// Make sure it doesn't go beyond last index
			if (end >= m)
			{
				end = m - 1;
			}

			MatrixXd batchX = Helpers::GetSubMatrix(X_train, start, end);
			MatrixXd batchY = Helpers::GetSubMatrix(Y_train, start, end);

			// Main training loop
			MatrixXd Yhat = FeedForward(batchX, true);
			J(loopIndex) = Helpers::MeanSquaredError(batchY, Yhat);

			Backprop(batchX, batchY, Yhat);

			// Calculate deltas
			dW1 = (-learningRate * gradW1.array()).matrix() + (momentum * dW1.array()).matrix();
			db1 = (-learningRate * gradB1.array()).matrix() + (momentum * db1.array()).matrix();
			dW2 = (-learningRate * gradW2.array()).matrix() + (momentum * dW2.array()).matrix();
			db2 = (-learningRate * gradB2.array()).matrix() + (momentum * db2.array()).matrix();

			// Update weights
			W1 += dW1;
			b1 += db1;
			W2 += dW2;
			b2 += db2;

			MatrixXd Yvalid = Predict(X_valid, TwoLayerNN::normalizeOutput);
			Valid(loopIndex) = Helpers::MeanSquaredError(Y_valid, Yvalid);

			if (loopIndex % printOn == 0)
			{
				cout << "Epoch " << i << ", iter " << j << ", loss: " << J(loopIndex) << ", valid: " << Valid(loopIndex) << endl;
			}

			if (Valid(loopIndex) < bestValid)
			{
				bestValid = Valid(loopIndex);
				bestValidIndex = loopIndex;
				UpdateBestWeights();
			}

			if (Valid(loopIndex) < altStop)
			{
				cout << endl << "Early stopping\n\n";
				MatrixXd Ytest = Predict(X_test, TwoLayerNN::normalizeOutput);
				cout << "Test error: " << Helpers::MeanSquaredError(Y_test, Ytest) << endl;
				return;
			}

			loopIndex++;
		}
	}

	cout << "Best validation error achieved:\n";
	cout << "Loss: " << J(bestValidIndex) << ", valid: " << Valid(bestValidIndex) << endl;

	UseBestWeights();

	MatrixXd Ytest = Predict(X_test, TwoLayerNN::normalizeOutput);
	cout << "Test error: " << Helpers::MeanSquaredError(Y_test, Ytest) << endl;
}

void TwoLayerNN::TestNormalization(MatrixXd input)
{
	MatrixXd norm = Helpers::Normalize(input, muX, sigmaX);

	MatrixXd result = Helpers::UnNormalize(norm, muX, sigmaX);

	cout << "Before:\n" << input << endl;

	cout << "After:\n" << result << endl;
}

MatrixXd TwoLayerNN::Predict(MatrixXd input, bool unNormalizeOutput)
{
	MatrixXd input_norm = Helpers::Normalize(input, muX, sigmaX);

	MatrixXd Yhat = FeedForward(input_norm, false);

	if (unNormalizeOutput)
	{
		return Helpers::UnNormalize(Yhat, muY, sigmaY);
	}

	return Yhat;
}

#pragma endregion


#pragma region Private methods

MatrixXd TwoLayerNN::FeedForward(MatrixXd input, bool isTraining)
{
	MatrixXd Z1 = (W1 * input).colwise() + b1;
	MatrixXd activatedZ1 = Helpers::Sigmoid(Z1);

	if (isTraining)
	{
		A1 = activatedZ1;
	}

	return ((W2 * activatedZ1).colwise() + b2);
}

void TwoLayerNN::Backprop(MatrixXd X, MatrixXd Y, MatrixXd Yhat)
{
	double m = X.cols();

	MatrixXd dZ2 = Yhat - Y;
	gradW2 = ((dZ2 * A1.transpose()).array() / m).matrix();
	gradB2 = (dZ2.rowwise().sum().array() / m).matrix();
	MatrixXd dZ1 = ((W2.transpose() * dZ2).array() * (A1.array() * (1.0 - A1.array()))).matrix();
	gradW1 = ((dZ1 * X.transpose()).array() / m).matrix();
	gradB1 = (dZ1.rowwise().sum().array() / m).matrix();
}

void TwoLayerNN::UpdateBestWeights()
{
	bestW1 = W1;
	bestB1 = b1;
	bestW2 = W2;
	bestB2 = b2;
}

void TwoLayerNN::UseBestWeights()
{
	W1 = bestW1;
	b1 = bestB1;
	W2 = bestW2;
	b2 = bestB2;
}

void TwoLayerNN::SplitDataSet(MatrixXd x, MatrixXd y, double trainRatio, double validRatio, double testRatio)
{
	int m = x.cols();
	int* range = Helpers::ShuffledRange(m);

	int trainCount = trainRatio * m;
	int validCount = validRatio * m;
	int testCount = testRatio * m;

	int sum = trainCount + validCount + testCount;
	int diff = m - sum;

	if (diff > 0)
	{
		trainCount += diff;
	}

	int numRows = x.rows();

	X_train = MatrixXd::Zero(numRows, trainCount);
	Y_train = MatrixXd::Zero(numRows, trainCount);
	X_valid = MatrixXd::Zero(numRows, validCount);
	Y_valid = MatrixXd::Zero(numRows, validCount);
	X_test = MatrixXd::Zero(numRows, testCount);
	Y_test = MatrixXd::Zero(numRows, testCount);

	int col = 0;
	int localCol = 0;
	while (localCol < trainCount)
	{
		for (int row = 0; row < numRows; row++)
		{
			X_train(row, localCol) = x(row, range[col]);
			Y_train(row, localCol) = y(row, range[col]);
		}

		localCol++;
		col++;
	}

	localCol = 0;
	while (localCol < validCount)
	{
		for (int row = 0; row < numRows; row++)
		{
			X_valid(row, localCol) = x(row, range[col]);
			Y_valid(row, localCol) = y(row, range[col]);
		}

		localCol++;
		col++;
	}

	localCol = 0;
	while (localCol < testCount)
	{
		for (int row = 0; row < numRows; row++)
		{
			X_test(row, localCol) = x(row, range[col]);
			Y_test(row, localCol) = y(row, range[col]);
		}

		localCol++;
		col++;
	}
}

#pragma endregion