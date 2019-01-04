//============================================================================
// Name : Layer.h
// Author : David Nogueira
//============================================================================
#ifndef LAYER_H
#define LAYER_H

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cassert> // for assert()
#include "Node.h"
#include "Utils.h"

class Layer {
public:
  Layer() {
    m_num_nodes = 0;
    m_nodes.clear();
  };

  Layer(int num_inputs_per_node,
        int num_nodes,
        const std::string & activation_function,
        bool use_constant_weight_init = true,
        double constant_weight_init = 0.5) {
    m_num_inputs_per_node = num_inputs_per_node;
    m_num_nodes = num_nodes;
    m_nodes.resize(num_nodes);

    for (int i = 0; i < num_nodes; i++) {
      m_nodes[i].WeightInitialization(num_inputs_per_node,
                                      use_constant_weight_init,
                                      constant_weight_init);
    }

    std::pair<std::function<double(double)>,
      std::function<double(double)> > *pair;
    bool ret_val = utils::ActivationFunctionsManager::Singleton().
      GetActivationFunctionPair(activation_function,
                                &pair);
    assert(ret_val);
    m_activation_function = (*pair).first;
    m_deriv_activation_function = (*pair).second;
    m_activation_function_str = activation_function;
  };

  ~Layer() {
    m_num_inputs_per_node = 0;
    m_num_nodes = 0;
    m_nodes.clear();
  };

  int GetInputSize() const {
    return m_num_inputs_per_node;
  };

  int GetOutputSize() const {
    return m_num_nodes;
  };

  const std::vector<Node> & GetNodes() const {
    return m_nodes;
  }

  /**
   * Return the internal list of nodes, but modifiable.
   */
  std::vector<Node> & GetNodesChangeable() {
    return m_nodes;
  }


  void GetOutputAfterActivationFunction(const std::vector<double> &input,
                                        std::vector<double> * output) const {
    assert(input.size() == m_num_inputs_per_node);

    output->resize(m_num_nodes);

    for (size_t i = 0; i < m_num_nodes; ++i) {
      m_nodes[i].GetOutputAfterActivationFunction(input,
                                                  m_activation_function,
                                                  &((*output)[i]));
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
      m_nodes[i].GetInputInnerProdWithWeights(input_layer_activation,
                                              &net_sum);

      //dE/dwij = dE/doj . doj/dnetj . dnetj/dwij
      double dE_doj = 0.0;
      double doj_dnetj = 0.0;
      double dnetj_dwij = 0.0;

      dE_doj = deriv_error[i];
      doj_dnetj = m_deriv_activation_function(net_sum);

      for (size_t j = 0; j < m_num_inputs_per_node; j++) {
        (*deltas)[j] += dE_doj * doj_dnetj * m_nodes[i].GetWeights()[j];

        dnetj_dwij = input_layer_activation[j];

        m_nodes[i].UpdateWeight(j,
                                -(dE_doj * doj_dnetj * dnetj_dwij),
                                m_learning_rate);
      }
    }
  };


  void SetWeights( std::vector<std::vector<double>> & weights )
  {
      if( 0 <= weights.size() && weights.size() <= m_num_nodes )
      {
          // traverse the list of nodes
          size_t node_i = 0;
          for( Node & node : m_nodes )
          {
              node.SetWeights( weights[node_i] );
              node_i++;
          }
      }
      else
          throw new std::logic_error("Incorrect layer number in SetWeights call");
  };

  void SaveLayer(FILE * file) const {
    fwrite(&m_num_nodes, sizeof(m_num_nodes), 1, file);
    fwrite(&m_num_inputs_per_node, sizeof(m_num_inputs_per_node), 1, file);

    size_t str_size = m_activation_function_str.size();
    fwrite(&str_size, sizeof(size_t), 1, file);
    fwrite(m_activation_function_str.c_str(), sizeof(char), str_size, file);

    for (size_t i = 0; i < m_nodes.size(); i++) {
      m_nodes[i].SaveNode(file);
    }
  };
  void LoadLayer(FILE * file) {
    m_nodes.clear();

    fread(&m_num_nodes, sizeof(m_num_nodes), 1, file);
    fread(&m_num_inputs_per_node, sizeof(m_num_inputs_per_node), 1, file);

    size_t str_size = 0;
    fread(&str_size, sizeof(size_t), 1, file);
    m_activation_function_str.resize(str_size);
    fread(&(m_activation_function_str[0]), sizeof(char), str_size, file);

    std::pair<std::function<double(double)>,
      std::function<double(double)> > *pair;
    bool ret_val = utils::ActivationFunctionsManager::Singleton().
      GetActivationFunctionPair(m_activation_function_str,
                                &pair);
    assert(ret_val);
    m_activation_function = (*pair).first;
    m_deriv_activation_function = (*pair).second;
    
    m_nodes.resize(m_num_nodes);
    for (size_t i = 0; i < m_nodes.size(); i++) {
      m_nodes[i].LoadNode(file);
    }

  };

protected:
  size_t m_num_inputs_per_node{ 0 };
  size_t m_num_nodes{ 0 };
  std::vector<Node> m_nodes;

  std::string m_activation_function_str;
  std::function<double(double)>  m_activation_function;
  std::function<double(double)>  m_deriv_activation_function;
};

#endif //LAYER_H
