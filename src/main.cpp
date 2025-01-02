#include "ANN.h"
#include "Layer.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
using namespace std;

void readCSV(const std::string &filename) {
  std::ifstream file(filename);

  if (!file.is_open()) {
    std::cerr << "Error: Could not open the file!" << std::endl;
    return;
  }

  std::string line;
  while (std::getline(file, line)) {
    std::stringstream ss(line);
    std::string value;
    std::vector<std::string> row;

    while (std::getline(ss, value, ',')) {
      row.push_back(value);
    }

    // Print the row for demonstration
    for (const auto &col : row) {
      std::cout << col << " ";
    }
    std::cout << std::endl;
  }

  file.close();
}
int main() {
  std::cout << "HI" << "\n";

  /* readCSV("D:\\gerry\\ann\\mushrooms.csv"); */
  std::vector<Layer> layers;

  layers.emplace_back(Layer(3));
  layers.emplace_back(Layer(10));
  layers.emplace_back(Layer(10));
  layers.emplace_back(Layer(2));

  cout << "done initiating layers" << endl;

  ANN myNetwork(layers);
  vector<float> test = {21.3, 40, 4};
  myNetwork.forward(test);
}
