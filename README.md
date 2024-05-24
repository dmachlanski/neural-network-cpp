# neural-network-cpp

A simple implementation of a Multi-Layer Perceptron (MLP), including forward and backpropagation steps implemented from scratch in C++.

The data was collected from a PID-controlled robot while following a wall of a rectangular shape. The data type was numerical (sensor readings).

After training the neural network (NN) offline, it was deployed again onto the same robot within the same environment to test whether the NN-based controller can imitate PID well enough.

## Important code
- HelpersNN - main NN-related code
- TrainNN - instatntiate and train the NN on collected data
- RobotNN - deploy the NN onto the robot
