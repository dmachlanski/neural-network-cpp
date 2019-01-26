#include <iostream>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

namespace Helpers
{
	void MatrixToFile(string path, MatrixXd m, bool includeSize = true);
	MatrixXd FileToMatrix(string path);
	void FindNormParams(MatrixXd m, VectorXd &mu, VectorXd &sigma);
	MatrixXd Normalize(MatrixXd m, VectorXd mu, VectorXd sigma);
	MatrixXd UnNormalize(MatrixXd m, VectorXd mu, VectorXd sigma);
	MatrixXd Sigmoid(MatrixXd m, double lambda);
	double RootMeanSquaredError(MatrixXd desired, MatrixXd approx);
	double MeanSquaredError(MatrixXd desired, MatrixXd approx);
	int* ShuffledRange(int range);
	MatrixXd GetSubMatrix(MatrixXd m, int startIndex, int endIndex);
	MatrixXd GetRandomMatrix(int rows, int cols);
	MatrixXd GetDropuotMatrix(int rows, int cols, double p);
}