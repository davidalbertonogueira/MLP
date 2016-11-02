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
  MLP(int num_inputs,
      int num_outputs,
      int num_hidden_layers,
      int num_nodes_per_hidden_layer,
      bool use_constant_weight_init = true,
      double constant_weight_init = 0.5) {

    m_num_inputs = num_inputs;
    m_num_outputs = num_outputs;
    m_num_hidden_layers = num_hidden_layers;
    m_num_nodes_per_hidden_layer = num_nodes_per_hidden_layer;

    CreateMLP(use_constant_weight_init,
              constant_weight_init);
  }

  ~MLP() {
    m_num_inputs = 0;
    m_num_outputs = 0;
    m_num_hidden_layers = 0;
    m_num_nodes_per_hidden_layer = 0;
    m_layers.clear();
  };

  bool ExportNNWeights(std::vector<double> *weights)const;
  bool ImportNNWeights(const std::vector<double> & weights);

  void GetOutput(const std::vector<double> &input,
                 std::vector<double> * output,
                 std::vector<std::vector<double>> * all_layers_activations = nullptr,
                 bool apply_softmax = false) const;
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
  void CreateMLP(bool use_constant_weight_init,
                 double constant_weight_init = 0.5) {
    if (m_num_hidden_layers > 0) {
      //first layer
      m_layers.emplace_back(Layer(m_num_nodes_per_hidden_layer,
                                  m_num_inputs,
                                  use_constant_weight_init,
                                  constant_weight_init));
      //subsequent layers
      for (int i = 0; i < m_num_hidden_layers - 1; i++) {
        m_layers.emplace_back(Layer(m_num_nodes_per_hidden_layer,
                                    m_num_nodes_per_hidden_layer,
                                    use_constant_weight_init,
                                    constant_weight_init));
      }
      //last layer
      m_layers.emplace_back(Layer(m_num_outputs,
                                  m_num_nodes_per_hidden_layer,
                                  use_constant_weight_init,
                                  constant_weight_init));
    } else {
      m_layers.emplace_back(Layer(m_num_outputs,
                                  m_num_inputs,
                                  use_constant_weight_init,
                                  constant_weight_init));
    }
  }


  int m_num_inputs{ 0 };
  int m_num_outputs{ 0 };
  int m_num_hidden_layers{ 0 };
  int m_num_nodes_per_hidden_layer{ 0 };

  std::vector<Layer> m_layers;
};

#endif //MLP_H