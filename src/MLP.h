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
      double learning_rate,
      int max_iterations,
      double threshold) {

    m_num_inputs = num_inputs;
    m_num_outputs = num_outputs;
    m_num_hidden_layers = num_hidden_layers;
    m_num_nodes_per_hidden_layer = num_nodes_per_hidden_layer;

    m_learning_rate = learning_rate;
    m_max_iterations = max_iterations;
    m_threshold = threshold;
  };

  ~MLP() {
    m_layers.clear();
  };

  void CreateMLP() {
    if (m_num_hidden_layers > 0) {
      //first layer
      m_layers.emplace_back(Layer(m_num_nodes_per_hidden_layer, m_num_inputs));
      //subsequent layers
      for (int i = 0; i < m_num_hidden_layers - 1; i++) {
        m_layers.emplace_back(Layer(m_num_nodes_per_hidden_layer,
                                    m_num_nodes_per_hidden_layer));
      }
      //last layer
      m_layers.emplace_back(Layer(m_num_outputs, m_num_nodes_per_hidden_layer));
    } else {
      m_layers.emplace_back(Layer(m_num_outputs, m_num_inputs));
    }
  }


  std::vector<double> & GetOutputValues(const std::vector<double> &input);
  int GetOutputClass(const std::vector<double> &input);

  void Train(const std::vector<TrainingSample> &training_sample_set,
                       bool bias_already_in);

  //void UpdateWeight(const std::vector<double> &x,
  //                  double error);
private:

  int m_num_inputs;
  int m_num_outputs;
  int m_num_hidden_layers;
  int m_num_nodes_per_hidden_layer;

  double m_learning_rate;
  int m_max_iterations;
  double m_threshold;

  std::vector<Layer> m_layers;
};

#endif //MLP_H