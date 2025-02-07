#include "ANN.h"
#include <Eigen/Dense>
#include <iostream>
#include <random>
using namespace std;

// Activation functions
Eigen::MatrixXd relu(const Eigen::MatrixXd &x) { return x.cwiseMax(0.0); }

Eigen::MatrixXd sigmoid(const Eigen::MatrixXd &x) {
  return 1.0 / (1.0 + (-x.array()).exp());
}

// Derivatives for backpropagation
Eigen::MatrixXd relu_derivative(const Eigen::MatrixXd &x) {
  return (x.array() > 0).cast<double>();
}

Eigen::MatrixXd sigmoid_derivative(const Eigen::MatrixXd &x) {
  Eigen::MatrixXd s = sigmoid(x);
  return s.array() * (1 - s.array());
}

// Helper function to initialize weights with small random values
Eigen::MatrixXd randomMatrix(int rows, int cols) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<double> dist(0.0, 1.0);

  Eigen::MatrixXd mat(rows, cols);
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      mat(i, j) =
          dist(gen) * 0.01; // Small values to prevent exploding gradients
    }
  }
  return mat;
}

// ANN Constructor
ANN::ANN(int input_size, int hidden_size, int output_size, double learning_rate)
    : lr(learning_rate) {

  // Initialize weights with small random values
  W1 = randomMatrix(hidden_size, input_size);
  W2 = randomMatrix(output_size, hidden_size);

  // Initialize biases to zero
  b1 = Eigen::VectorXd::Zero(hidden_size);
  b2 = Eigen::VectorXd::Zero(output_size);
}

Eigen::MatrixXd ANN::forward(const Eigen::MatrixXd &X) {
  Eigen::MatrixXd Z1 = (W1 * X).colwise() + b1;
  Eigen::MatrixXd A1 = relu(Z1);

  Eigen::MatrixXd Z2 = (W2 * A1).colwise() + b2;
  Eigen::MatrixXd A2 = sigmoid(Z2);

  return A2; // Output
}
void ANN::train(const Eigen::MatrixXd &X, const Eigen::MatrixXd &y,
                int epochs) {
  int m = X.cols(); // Number of training examples

  for (int epoch = 0; epoch < epochs; epoch++) {
    // Forward Pass
    Eigen::MatrixXd Z1 = (W1 * X).colwise() + b1;
    Eigen::MatrixXd A1 = relu(Z1);

    Eigen::MatrixXd Z2 = (W2 * A1).colwise() + b2;
    Eigen::MatrixXd A2 = sigmoid(Z2);

    // Compute Loss (Mean Squared Error)
    Eigen::MatrixXd loss = (A2 - y).array().square();
    double cost = loss.mean();

    // Backward Pass (Gradient Calculation)
    Eigen::MatrixXd dZ2 = A2 - y;
    Eigen::MatrixXd dW2 = (dZ2 * A1.transpose()) / m;
    Eigen::VectorXd db2 = dZ2.rowwise().mean();

    Eigen::MatrixXd dZ1 =
        (W2.transpose() * dZ2).cwiseProduct(relu_derivative(Z1));
    Eigen::MatrixXd dW1 = (dZ1 * X.transpose()) / m;
    Eigen::VectorXd db1 = dZ1.rowwise().mean();

    // Gradient Descent Update
    W1 -= lr * dW1;
    b1 -= lr * db1;
    W2 -= lr * dW2;
    b2 -= lr * db2;

    if (epoch % 100 == 0) {
      std::cout << "Epoch " << epoch << ", Loss: " << cost << std::endl;
    }
  }
}
