#pragma once
#include "Layer.h"
#include <vector>

class ANN {
private:
  std::vector<Layer> layersArray;
  std::vector<std::vector<std::vector<float>>> weightsArray;
  void populateWeights();

public:
  ANN(std::vector<Layer>);
  void train();
  void forward(std::vector<float> inputValue);
};
