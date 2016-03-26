//============================================================================
// Name : Layer.h
// Author : David Nogueira
//============================================================================
#ifndef LAYER_H
#define LAYER_H

#include "Node.h"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cassert> // for assert()

class Layer {
public:
  Layer() {
    m_num_nodes = 0;
    m_nodes.clear();
  };


  Layer(int num_nodes, int num_inputs_per_node) {
    m_num_nodes = num_nodes;
    m_num_inputs_per_node = num_inputs_per_node;
    m_nodes = std::vector<Node>(num_nodes, Node(num_inputs_per_node));
  };

  ~Layer() {
    m_nodes.clear();
  };

  void GetOutput(const std::vector<double> &input, std::vector<double> * output) const {
    assert(input.size() == m_num_inputs_per_node);
  
    output->resize(m_num_nodes);

    for (int i = 0; i < m_num_nodes; ++i) {
      (*output)[i] = m_nodes[i].GetOutput(input);
    }
  }

  void UpdateWeights(const std::vector<double> &x,
                     double m_learning_rate,
                     double error) {
    assert(x.size() == m_num_inputs_per_node);

    for (size_t i = 0; i < m_nodes.size(); i++)
      m_nodes[i].UpdateWeights(x, m_learning_rate, error);
  };

protected:
  int m_num_nodes;
  int m_num_inputs_per_node;
  std::vector<Node> m_nodes;
};

#endif //LAYER_H