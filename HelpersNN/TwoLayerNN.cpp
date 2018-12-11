#include "TwoLayerNN.h"
#include "Helpers.h"
#include <fstream>

TwoLayerNN::TwoLayerNN()
{
}

#pragma region Public methods

void TwoLayerNN::InitializeModel(string inputDataPath, string outputDataPath, int hiddenUnits, bool normalizeBeforeSplit, bool isOutputLinear)
{
	// Wether the output should be linear
	TwoLayerNN::isOutputLinear = isOutputLinear;

	// Load the data set from files
	MatrixXd X = Helpers::FileToMatrix(inputDataPath);
	MatrixXd Y = Helpers::FileToMatrix(outputDataPath);

	// Whether normalization should be done before splitting data into valid/test sets
	if (normalizeBeforeSplit)
	{
		// Calculate normalization parameters that can be later used to (un)normalize the data
		Helpers::FindNormParams(X, muX, sigmaX);
		Helpers::FindNormParams(Y, muY, sigmaY);

		// Finally, normalize inputs
		X = Helpers::Normalize(X, muX, sigmaX);

		// Only normalize output if it isn't linear
		if (!isOutputLinear)
		{
			Y = Helpers::Normalize(Y, muY, sigmaY);
		}
	}

	// Split the data into training/validation/test sets (70/15/15)
	// Also shuffles indexes before splitting
	SplitDataSet(X, Y, 0.7, 0.15, 0.15);

	// Set the sizes for all the neurons
	inputSize = X.rows();
	hiddenSize = hiddenUnits;
	outputSize = Y.rows();

	// Initialize weights with random values between 0 and 1
	W1 = Helpers::GetRandomMatrix(hiddenSize, inputSize);
	W2 = Helpers::GetRandomMatrix(outputSize, hiddenSize);

	// And biases with zeros
	b1 = VectorXd::Zero(hiddenSize);	
	b2 = VectorXd::Zero(outputSize);

	// Initialize deltas with zeros
	dW1 = MatrixXd::Zero(hiddenSize, inputSize);
	db1 = VectorXd::Zero(hiddenSize);
	dW2 = MatrixXd::Zero(outputSize, hiddenSize);
	db2 = VectorXd::Zero(outputSize);

	if (!normalizeBeforeSplit)
	{
		Helpers::FindNormParams(X_train, muX, sigmaX);
		Helpers::FindNormParams(Y_train, muY, sigmaY);
		X_train = Helpers::Normalize(X_train, muX, sigmaX);
		X_valid = Helpers::Normalize(X_valid, muX, sigmaX);
		X_test = Helpers::Normalize(X_test, muX, sigmaX);

		if (!isOutputLinear)
		{
			Y_train = Helpers::Normalize(Y_train, muY, sigmaY);
			Y_valid = Helpers::Normalize(Y_valid, muY, sigmaY);
			Y_test = Helpers::Normalize(Y_test, muY, sigmaY);
		}
	}
}

void TwoLayerNN::LoadModel(string path)
{
	// Loads the weights
	W1 = Helpers::FileToMatrix(path + "W1.csv");
	b1 = Helpers::FileToMatrix(path + "b1.csv");
	W2 = Helpers::FileToMatrix(path + "W2.csv");
	b2 = Helpers::FileToMatrix(path + "b2.csv");

	// And normalization parameters
	muX = Helpers::FileToMatrix(path + "muX.csv");
	sigmaX = Helpers::FileToMatrix(path + "sigmaX.csv");
	muY = Helpers::FileToMatrix(path + "muY.csv");
	sigmaY = Helpers::FileToMatrix(path + "sigmaY.csv");
}

void TwoLayerNN::SaveModel(string path, bool saveBestWeights)
{
	// Whether the best weights obtained during training should be saved as the final ones
	if (saveBestWeights)
	{
		W1 = bestW1;
		b1 = bestB1;
		W2 = bestW2;
		b2 = bestB2;
	}

	// Saves all the weights
	Helpers::MatrixToFile(path + "W1.csv", W1);
	Helpers::MatrixToFile(path + "b1.csv", b1);
	Helpers::MatrixToFile(path + "W2.csv", W2);
	Helpers::MatrixToFile(path + "b2.csv", b2);

	// Normalization parameters
	Helpers::MatrixToFile(path + "muX.csv", muX);
	Helpers::MatrixToFile(path + "sigmaX.csv", sigmaX);
	Helpers::MatrixToFile(path + "muY.csv", muY);
	Helpers::MatrixToFile(path + "sigmaY.csv", sigmaY);

	// And error rate after each epoch of training and validation set respectively
	Helpers::MatrixToFile(path + "JEpoch.csv", JEpoch, false);
	Helpers::MatrixToFile(path + "ValidEpoch.csv", ValidEpoch, false);
}

void TwoLayerNN::Train(double learningRate, double momentum, int batchSize, int epochs)
{
	// Number of training examples
	int m = X_train.cols();

	if (batchSize < 0)
	{
		// In this case the batch size is the entire training set
		batchSize = m;
	}

	int iterations = m / batchSize;
	int all = iterations * batchSize;

	if (all < m)
	{
		iterations++;
	}

	// Initialize vectors to store error rates during training
	JEpoch = VectorXd::Zero(epochs);
	ValidEpoch = VectorXd::Zero(epochs);
	double bestValid = DBL_MAX;
	int bestValidIndex;

	// This will store intermediate training error rates when iterating through training examples
	// It will be needed to get the loss after whole epoch
	MatrixXd JTemp(X_train.rows(), X_train.cols());

	for (int i = 0; i < epochs; i++)
	{
		for (int j = 0; j < iterations; j++)
		{
			// Calculate first and last index of the next batch
			int start = j * batchSize;
			int end = start + batchSize - 1;

			// Make sure it doesn't go beyond last index
			if (end >= m)
			{
				end = m - 1;
			}

			MatrixXd batchX = Helpers::GetSubMatrix(X_train, start, end);
			MatrixXd batchY = Helpers::GetSubMatrix(Y_train, start, end);

			// Forward pass
			MatrixXd Yhat = FeedForward(batchX, true, TwoLayerNN::isOutputLinear);

			// Save temporary loss to calculate final loss after entire epoch
			int JTempCol = 0;
			for (int c = start; c <= end; c++)
			{	
				for (int r = 0; r < Yhat.rows(); r++)
				{
					JTemp(r, c) = Yhat(r, JTempCol);
				}

				JTempCol++;
			}

			// Backpropagation
			Backprop(batchX, batchY, Yhat);

			// Calculate deltas
			// new_delta = (-learning_rate * gradient) + (momentum * old_delta)
			dW1 = (-learningRate * gradW1.array()).matrix() + (momentum * dW1.array()).matrix();
			db1 = (-learningRate * gradB1.array()).matrix() + (momentum * db1.array()).matrix();
			dW2 = (-learningRate * gradW2.array()).matrix() + (momentum * dW2.array()).matrix();
			db2 = (-learningRate * gradB2.array()).matrix() + (momentum * db2.array()).matrix();

			// Update weights
			// new_weight = old_weight + delta
			W1 += dW1;
			b1 += db1;
			W2 += dW2;
			b2 += db2;
		}

		// Calculate training error
		JEpoch(i) = Helpers::RootMeanSquaredError(Y_train, JTemp);

		// Calculate validation error
		MatrixXd Yvalid = FeedForward(X_valid, false, TwoLayerNN::isOutputLinear);
		ValidEpoch(i) = Helpers::RootMeanSquaredError(Y_valid, Yvalid);

		std::cout << "Epoch: " << i << ", validation error: " << ValidEpoch(i) << endl;

		// Save the weights if they are the best so far
		if (ValidEpoch(i) < bestValid)
		{
			bestValid = ValidEpoch(i);
			bestValidIndex = i;
			UpdateBestWeights();
		}
	}

	// Report the best validation error obtained
	std::cout << "Best validation error achieved:\n";
	std::cout << ValidEpoch(bestValidIndex) << endl;

	// Use the best weights as the current ones
	UseBestWeights();

	// Calculate test error (using the best weights)
	MatrixXd Ytest = FeedForward(X_test, false, TwoLayerNN::isOutputLinear);
	std::cout << "Test error: " << Helpers::RootMeanSquaredError(Y_test, Ytest) << endl;
}

void TwoLayerNN::TestNormalization(MatrixXd input)
{
	MatrixXd norm = Helpers::Normalize(input, muX, sigmaX);

	MatrixXd result = Helpers::UnNormalize(norm, muX, sigmaX);

	std::cout << "Before:\n" << input << endl;

	std::cout << "After:\n" << result << endl;
}

MatrixXd TwoLayerNN::Predict(MatrixXd input, bool unNormalizeOutput)
{
	MatrixXd input_norm = Helpers::Normalize(input, muX, sigmaX);

	MatrixXd Yhat = FeedForward(input_norm, false, !unNormalizeOutput);

	if (unNormalizeOutput)
	{
		return Helpers::UnNormalize(Yhat, muY, sigmaY);
	}

	return Yhat;
}

#pragma endregion


#pragma region Private methods

MatrixXd TwoLayerNN::FeedForward(MatrixXd input, bool isTraining, bool isOutputLinear)
{
	MatrixXd result;
	MatrixXd Z1 = (W1 * input).colwise() + b1;
	MatrixXd activatedZ1 = Helpers::Sigmoid(Z1);

	if (isTraining)
	{
		// If this is part of training, save activated hidden layer
		A1 = activatedZ1;
	}

	MatrixXd Z2 = (W2 * activatedZ1).colwise() + b2;

	if (isOutputLinear)
	{
		// If the output is linear, don't apply sigmoid to the last layer
		result = Z2;
	}
	else
	{
		MatrixXd activatedZ2 = Helpers::Sigmoid(Z2);

		if (isTraining)
		{
			A2 = activatedZ2;
		}

		result = activatedZ2;
	}

	return result;
}

void TwoLayerNN::Backprop(MatrixXd X, MatrixXd Y, MatrixXd Yhat)
{
	double m = X.cols();

	// The order here is deliberately changed to avoid adding '-' in front of the error in the equation
	MatrixXd e = Yhat - Y;
	MatrixXd dZ2;

	if (TwoLayerNN::isOutputLinear)
	{
		dZ2 = e;
	}
	else
	{
		// We need a derivative of the activation function if the output is not linear
		// A2 is the same as sigmoid(Z2) so no need to compute sigmoid function every time
		dZ2 = (e.array() * (A2.array() * (1.0 - A2.array()))).matrix();
	}

	// Compute the gradients
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