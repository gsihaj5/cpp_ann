#include "Layer.h"
#include <iostream>
#include <ostream>
#include <vector>

Layer::Layer(int nodeSize) { nodesVector = std::vector<float>(nodeSize); };

void Layer::printNodes() {
  for (int i = 0; i < nodesVector.size(); i++) {
    std::cout << nodesVector[i] << std::endl;
  }
}

void Layer::setNode(std::vector<float> newNodesVector) {
  nodesVector = newNodesVector;
}
int Layer::getNodeSize() { return nodesVector.size(); };

float Layer::getNode(int index) { return nodesVector[index]; };

void Layer::forward(std::vector<std::vector<float>> weightsarray,
                    Layer prevLayer) {
  for (int i = 0; i < nodesVector.size(); i++) {

    float tempsum = 0;
    for (int j = 0; j < prevLayer.getNodeSize(); j++) {
      tempsum += prevLayer.getNode(j) * weightsarray[i][j];
    }
    nodesVector[i] = tempsum;
  }
}
