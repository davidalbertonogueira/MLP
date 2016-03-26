//============================================================================
// Name : Node.h
// Author : David Nogueira
//============================================================================
#ifndef NODE_H
#define NODE_H

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cassert> // for assert()

#define ZERO_WEIGHT_INITIALIZATION 1
#define USE_SIGMOID 1

class Node {
public:
  Node() {
    m_bias = 0.0;
    m_num_inputs = 0;
    m_weights.clear();
  };
  Node(int num_inputs) {
    m_bias = 0.0;
    m_num_inputs = num_inputs + 1;
    m_weights.clear();
    m_weights = std::vector<double>(m_num_inputs);

    //initialize weight vector
    std::generate_n(m_weights.begin(),
                    m_num_inputs,
                    (ZERO_WEIGHT_INITIALIZATION) ?
                    utils::gen_rand(0) : utils::gen_rand());
  };
  ~Node() {
    m_weights.clear();
    //m_old_weights.clear();
  };
  int GetInputSize() const {
    return m_num_inputs;
  }
  void SetInputSize(int num_inputs) {
    m_num_inputs = num_inputs;
  }
  double GetBias() const {
    return m_bias;
  }

  void SetBias(double bias) {
    m_bias = bias;
  }

  std::vector<double> & GetWeights() {
    return m_weights;
  }

  const std::vector<double> & GetWeights() const {
    return m_weights;
  }

  size_t GetWeightsVectorSize() const {
    return m_weights.size();
  }

  void GetOutput(const std::vector<double> &input, double * output) const {
    assert(input.size() == m_weights.size());
    double inner_prod = std::inner_product(begin(input),
                                           end(input),
                                           begin(m_weights),
                                           0.0);
    *output = inner_prod;
  }

  void GetFilteredOutput(const std::vector<double> &input, double * bool_output) {
    double inner_prod;
    GetOutput(input, &inner_prod);
#if USE_SIGMOID == 1
    double y = utils::sigmoid(inner_prod);
    *bool_output = (y > 0) ? true : false;
#else
    *bool_output = (inner_prod > 0) ? true : false;
#endif
  };

  void UpdateWeights(const std::vector<double> &x,
                     double m_learning_rate,
                     double error) {
    assert(x.size() == m_weights.size());
    for (size_t i = 0; i < m_weights.size(); i++)
      m_weights[i] += x[i] * m_learning_rate *  error;
  };
protected:
  int m_num_inputs;
  double m_bias;
  std::vector<double> m_weights;
};

#endif //NODE_H