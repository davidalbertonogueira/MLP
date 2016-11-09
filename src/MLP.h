//============================================================================
// Name : MLP.cpp
// Author : David Nogueira
//============================================================================
#ifndef MLP_H
#define MLP_H

#include "Layer.h"
#include "Sample.h"
#include "Utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <algorithm>

class MLP {
public:
  //desired call sintax :  MLP({64*64,20,4}, {"sigmoid", "linear"},
  MLP(const std::vector<uint64_t> & layers_nodes,
      const std::vector<std::string> & layers_activfuncs,
      bool use_constant_weight_init = true,
      double constant_weight_init = 0.5) {
    assert(layers_nodes.size() >= 2);
    assert(layers_activfuncs.size() + 1 == layers_nodes.size());

    CreateMLP(layers_nodes,
              layers_activfuncs,
              use_constant_weight_init,
              constant_weight_init);
  }


  ~MLP() {
    m_num_inputs = 0;
    m_num_outputs = 0;
    m_num_hidden_layers = 0;
    m_layers_nodes.clear();
    m_layers.clear();
  };

  bool ExportNNWeights(std::vector<double> *weights)const;
  bool ImportNNWeights(const std::vector<double> & weights);

  void GetOutput(const std::vector<double> &input,
                 std::vector<double> * output,
                 std::vector<std::vector<double>> * all_layers_activations = nullptr) const;
  void GetOutputClass(const std::vector<double> &output, size_t * class_id) const;

  void UpdateMiniBatch(const std::vector<TrainingSample> &training_sample_set_with_bias,
                       double learning_rate,
                       int max_iterations = 5000,
                       double min_error_cost = 0.001);
protected:
  void UpdateWeights(const std::vector<std::vector<double>> & all_layers_activations,
                     const std::vector<double> &error,
                     double learning_rate);
private:
  void CreateMLP(const std::vector<uint64_t> & layers_nodes,
                 const std::vector<std::string> & layers_activfuncs,
                 bool use_constant_weight_init,
                 double constant_weight_init = 0.5) {
    m_layers_nodes = layers_nodes;
    m_num_inputs = m_layers_nodes[0];
    m_num_outputs = m_layers_nodes[m_layers_nodes.size() - 1];
    m_num_hidden_layers = m_layers_nodes.size() - 2;

    for (int i = 0; i < m_layers_nodes.size() - 1; i++) {
      m_layers.emplace_back(Layer(m_layers_nodes[i],
                                  m_layers_nodes[i + 1],
                                  layers_activfuncs[i],
                                  use_constant_weight_init,
                                  constant_weight_init));
    }
  }
  int m_num_inputs{ 0 };
  int m_num_outputs{ 0 };
  int m_num_hidden_layers{ 0 };
  std::vector<uint64_t> m_layers_nodes;
  std::vector<Layer> m_layers;
};

#endif //MLP_H