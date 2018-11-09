#include "Helpers.h"
#include <fstream>

namespace Helpers
{
	MatrixXd FileToMatrix(string path)
	{
		ifstream file(path);

		string data;
		double value;

		getline(file, data, ',');
		int rows = stoi(data);

		getline(file, data);
		int cols = stoi(data);

		MatrixXd m(rows, cols);

		for (int i = 0; i < rows; i++)
		{
			for (int j = 0; j < cols; j++)
			{
				if (j < (cols - 1))
				{
					getline(file, data, ',');
				}
				else
				{
					getline(file, data);
				}

				value = stod(data);
				m(i, j) = value;
			}
		}

		file.close();

		return m;
	}

	void MatrixToFile(string path, MatrixXd m, bool includeSize = true)
	{
		ofstream file(path);

		if (file)
		{
			if (includeSize)
			{
				file << m.rows() << "," << m.cols() << endl;
			}

			for (int i = 0; i < m.rows(); i++)
			{
				for (int j = 0; j < m.cols(); j++)
				{
					file << m(i, j);

					if (j < (m.cols() - 1))
					{
						file << ",";
					}
				}

				if (i < (m.rows() - 1))
				{
					file << endl;
				}
			}

			file.close();
		}
	}

	void FindNormParams(MatrixXd m, VectorXd &mean, VectorXd &range)
	{
		mean = m.rowwise().mean();
		range = m.rowwise().maxCoeff() - m.rowwise().minCoeff();
	}

	MatrixXd Normalize(MatrixXd m, VectorXd mu, VectorXd sigma)
	{
		return ((m.colwise() - mu).array().colwise() / sigma.array()).matrix();
	}

	MatrixXd Sigmoid(MatrixXd m)
	{
		MatrixXd result(m.rows(), m.cols());

		for (int i = 0; i < m.rows(); i++)
		{
			for (int j = 0; j < m.cols(); j++)
			{
				result(i, j) = exp(-m(i, j));
			}
		}

		return (1.0 / (1.0 + result.array())).matrix();
	}

	double MeanSquaredError(MatrixXd desired, MatrixXd approx)
	{
		double m = desired.cols();

		ArrayXXd diff = (desired - approx).array();

		return ((diff*diff).sum() / (2.0 * m));
	}
}