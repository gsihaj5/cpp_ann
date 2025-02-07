#pragma once
#include <Eigen/Dense>

class ANN {
public:
  ANN(int input_size, int hidden_size, int output_size, double learning_rate);
  Eigen::MatrixXd forward(const Eigen::MatrixXd &X);
  void train(const Eigen::MatrixXd &X, const Eigen::MatrixXd &y, int epochs);

private:
  Eigen::MatrixXd W1, W2;
  Eigen::VectorXd b1, b2;
  double lr; // Learning rate
};
