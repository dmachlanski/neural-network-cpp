#include "TwoLayerNN.h"
#include "Helpers.h"
#include <fstream>

TwoLayerNN::TwoLayerNN()
{
}

#pragma region Public methods

void TwoLayerNN::InitializeModel(string inputDataPath, string outputDataPath, int hiddenUnits, bool normalizeBeforeSplit, bool isOutputLinear)
{
	TwoLayerNN::isOutputLinear = isOutputLinear;

	// Load the data
	MatrixXd X = Helpers::FileToMatrix(inputDataPath);
	MatrixXd Y = Helpers::FileToMatrix(outputDataPath);

	if (normalizeBeforeSplit)
	{
		Helpers::FindNormParams(X, muX, sigmaX);
		Helpers::FindNormParams(Y, muY, sigmaY);
		X = Helpers::Normalize(X, muX, sigmaX);

		if (!isOutputLinear)
		{
			Y = Helpers::Normalize(Y, muY, sigmaY);
		}
	}

	// Split the data (70/15/15)
	SplitDataSet(X, Y, 0.7, 0.15, 0.15);

	// Set the sizes for all the matrixes
	inputSize = X.rows();
	hiddenSize = hiddenUnits;
	outputSize = Y.rows();

	// Initialize weights
	W1 = Helpers::GetRandomMatrix(hiddenSize, inputSize);
	b1 = VectorXd::Zero(hiddenSize);
	W2 = Helpers::GetRandomMatrix(outputSize, hiddenSize);
	b2 = VectorXd::Zero(outputSize);

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
	//Helpers::MatrixToFile(path + "J.csv", J, false);
	//Helpers::MatrixToFile(path + "Valid.csv", Valid, false);
	Helpers::MatrixToFile(path + "JEpoch.csv", JEpoch, false);
	Helpers::MatrixToFile(path + "ValidEpoch.csv", ValidEpoch, false);
}

void TwoLayerNN::Train(double learningRate, double momentum, int batchSize, int epochs, int printOn)
{
	int m = X_train.cols();

	if (batchSize < 0)
	{
		batchSize = m;
	}

	int iterations = m / batchSize;
	int all = iterations * batchSize;

	if (all < m)
	{
		iterations++;
	}

	J = VectorXd::Zero((epochs * iterations) + 1);
	JEpoch = VectorXd::Zero(epochs + 1);
	Valid = VectorXd::Zero((epochs * iterations) + 1);
	ValidEpoch = VectorXd::Zero(epochs + 1);
	double bestValid = DBL_MAX;
	int bestValidIndex;
	int loopIndex = 1;

	if (printOn < 0)
	{
		printOn = m - 1;
	}

	// Calculate initial loss
	MatrixXd Yhat = FeedForward(X_train, true, TwoLayerNN::isOutputLinear);
	J(0) = Helpers::RootMeanSquaredError(Y_train, Yhat);
	JEpoch(0) = J(0);
	MatrixXd Yvalid = FeedForward(X_valid, false, TwoLayerNN::isOutputLinear);
	Valid(0) = Helpers::RootMeanSquaredError(Y_valid, Yvalid);
	ValidEpoch(0) = Valid(0);

	std::cout << "Initial errors:\n";
	std::cout << "Train error: " << J(0) << ", Validation error: " << Valid(0) << endl;

	MatrixXd JTemp(X_train.rows(), X_train.cols());

	for (int i = 1; i <= epochs; i++)
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
			Yhat = FeedForward(batchX, true, TwoLayerNN::isOutputLinear);
			//J(loopIndex) = Helpers::RootMeanSquaredError(batchY, Yhat);

			// update JTemp
			int JTempCol = 0;
			for (int c = start; c <= end; c++)
			{	
				for (int r = 0; r < Yhat.rows(); r++)
				{
					JTemp(r, c) = Yhat(r, JTempCol);
				}

				JTempCol++;
			}

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

			//Yvalid = FeedForward(X_valid, false, TwoLayerNN::isOutputLinear);
			//Valid(loopIndex) = Helpers::RootMeanSquaredError(Y_valid, Yvalid);

			//if (loopIndex % printOn == 0)
			//{
			//	std::cout << "Epoch " << i << ", iter " << j << ", loss: " << J(loopIndex) << ", valid: " << Valid(loopIndex) << endl;
			//}

			//loopIndex++;
		}
		// calculate RMSE based on JEpoch
		JEpoch(i) = Helpers::RootMeanSquaredError(Y_train, JTemp);

		// calculate validation error (RMSE)
		Yvalid = FeedForward(X_valid, false, TwoLayerNN::isOutputLinear);
		ValidEpoch(i) = Helpers::RootMeanSquaredError(Y_valid, Yvalid);

		std::cout << "Epoch: " << i << ", validation error: " << ValidEpoch(i) << endl;

		if (ValidEpoch(i) < bestValid)
		{
			bestValid = ValidEpoch(i);
			bestValidIndex = i;
			UpdateBestWeights();
		}
	}

	std::cout << "Best validation error achieved:\n";
	std::cout << ValidEpoch(bestValidIndex) << endl;

	UseBestWeights();

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
		// Try dropout
		//double p = 0.5;
		//MatrixXd mask = Helpers::GetDropuotMatrix(activatedZ1.rows(), activatedZ1.cols(), p);
		//activatedZ1 = (activatedZ1.array() * mask.array()).matrix();

		A1 = activatedZ1;
	}

	MatrixXd Z2 = (W2 * activatedZ1).colwise() + b2;

	if (isOutputLinear)
	{
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

	MatrixXd e = Yhat - Y;
	MatrixXd dZ2;

	if (TwoLayerNN::isOutputLinear)
	{
		dZ2 = e;
	}
	else
	{
		dZ2 = (e.array() * (A2.array() * (1.0 - A2.array()))).matrix();
	}

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