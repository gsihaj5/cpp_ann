#pragma once
#include <string>
#include <vector>

class Layer {
private:
  std::vector<float> nodesVector;
  std::string activationFunction;

public:
  Layer(int nodeSize);
  void printNodes();
  void setNode(std::vector<float> newNodesVector);
  int getNodeSize();
  float getNode(int index);
  void forward(std::vector<std::vector<float>> weightsarray, Layer prevLayer);
};
