#include "ANN.h"
#include <iostream>
#include <vector>
using namespace std;

ANN::ANN(std::vector<Layer> layers) : layersArray(layers) {
  int prevSize = layers[0].getNodeSize();

  for (int i = 1; i < layers.size(); i++) {
    Layer layer = layers.at(i);
    std::vector<std::vector<float>> weightVector;
    for (int j = 0; j < layer.getNodeSize(); j++) {
      std::vector<float> arr(prevSize);
      weightVector.emplace_back(arr);
    }
    cout << "prev size " << prevSize << endl;
    cout << "current size " << weightVector.size() << endl << endl;

    weightsArray.emplace_back(weightVector);
    prevSize = layer.getNodeSize();
  }
  populateWeights();
};

void ANN::populateWeights() {
  for (int i = 0; i < weightsArray.size(); i++) {
    for (int j = 0; j < weightsArray[i].size(); j++) {
      for (int k = 0; k < weightsArray[i][j].size(); k++) {
        weightsArray[i][j][k] =
            static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
      }
    }
  }
};

void ANN::forward(std::vector<float> inputValue) {
  layersArray[0].setNode(inputValue);
  for (int i = 0; i < layersArray.size() - 1; i++) {
    layersArray[i + 1].forward(weightsArray[i], layersArray[i]);
  }
}
