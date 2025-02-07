#include <iostream>
#include <Eigen/Dense>
#include "ANN.h"

int main() {
    Eigen::MatrixXd X(2, 4);  // 2 input features, 4 training samples
    X << 0, 0, 1, 1,
         0, 1, 0, 1; 

    Eigen::MatrixXd y(1, 4);  // Expected output (XOR problem)
    y << 0, 1, 1, 0;  

    ANN ann(2, 3, 1, 0.1);  // 2-input, 3-hidden, 1-output, learning rate=0.1
    ann.train(X, y, 1000);

    Eigen::MatrixXd prediction = ann.forward(X);
    std::cout << "Predicted Output:\n" << prediction << std::endl;

    return 0;
}
