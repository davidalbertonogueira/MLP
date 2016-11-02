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

  Layer(int num_nodes,
        int num_inputs_per_node,
        bool use_constant_weight_init = true,
        double constant_weight_init = 0.5) {
    m_num_nodes = num_nodes;
    m_num_inputs_per_node = num_inputs_per_node;
    m_nodes.resize(num_nodes,
                   std::move(Node(num_inputs_per_node,
                                  use_constant_weight_init,
                                  constant_weight_init)));
  };


  ~Layer() {
    m_num_nodes = 0;
    m_num_inputs_per_node = 0;
    m_nodes.clear();
  };

  //std::vector<Node> & GetNodes() {
  //  return m_nodes;
  //}

  const std::vector<Node> & GetNodes() const {
    return m_nodes;
  }

  void GetOutputAfterSigmoid(const std::vector<double> &input, std::vector<double> * output) const {
    assert(input.size() == m_num_inputs_per_node);

    output->resize(m_num_nodes);

    for (int i = 0; i < m_num_nodes; ++i) {
      m_nodes[i].GetOutputAfterSigmoid(input, &((*output)[i]));
    }
  }

  void UpdateWeights(const std::vector<double> &input_layer_activation,
                     const std::vector<double> &deriv_error,
                     double m_learning_rate,
                     std::vector<double> * deltas) {
    assert(input_layer_activation.size() == m_num_inputs_per_node);
    assert(deriv_error.size() == m_nodes.size());

    deltas->resize(m_num_inputs_per_node, 0);

    for (size_t i = 0; i < m_nodes.size(); i++) {
      double net_sum;
      m_nodes[i].GetInputInnerProdWithWeights(input_layer_activation, &net_sum);

      //dE/dwij = dE/doj . doj/dnetj . dnetj/dwij
      double dE_doj = 0.0;
      double doj_dnetj = 0.0;
      double dnetj_dwij = 0.0;

      dE_doj = deriv_error[i];
      doj_dnetj = utils::deriv_sigmoid(net_sum);


      for (int j = 0; j < m_num_inputs_per_node; j++) {
        (*deltas)[j] += dE_doj * doj_dnetj * m_nodes[i].GetWeights()[j];

        dnetj_dwij = input_layer_activation[j];

        m_nodes[i].UpdateWeight(j,
                                -(dE_doj * doj_dnetj * dnetj_dwij),
                                m_learning_rate);
      }
    }
  };

protected:
  int m_num_nodes{ 0 };
  int m_num_inputs_per_node{ 0 };
  std::vector<Node> m_nodes;
};

#endif //LAYER_H