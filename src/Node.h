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

#define ZERO_WEIGHT_INITIALIZATION 1

class Node {
public:
  Node() {
    m_bias = 0.0;
    //m_old_bias = 0.0;
    m_num_inputs = 0;
    m_weights.clear();
    //m_old_weights.clear();
  };
  Node(int num_inputs) {
    m_bias = 0.0;
    //m_old_bias = 0.0;
    m_num_inputs = num_inputs;
    m_weights.clear();
    //m_old_weights.clear();
    m_weights = std::vector<double>(num_inputs);
    //m_old_weights = std::vector<double>(num_inputs);

    //initialize weight vector
    std::generate_n(m_weights.begin(),
                    num_inputs,
                      (ZERO_WEIGHT_INITIALIZATION) ? 
                    utils::gen_rand(0) : utils::gen_rand());
  };
  ~Node() {
    m_weights.clear();
    //m_old_weights.clear();
  };
  int GetInputSize() {
    return m_num_inputs;
  }
  void SetInputSize(int num_inputs) {
    m_num_inputs = num_inputs;
  }
  double GetBias() {
    return m_bias;
  }
  //double GetOldBias() {
  //  return m_old_bias;
  //}
  void SetBias(double bias) {
    m_bias = bias;
  }
  //void SetOldBias(double old_bias) {
  //  m_old_bias = old_bias;
  //}
  std::vector<double> & GetWeights() {
    return m_weights;
  }
  //std::vector<double> & GetOldWeights() {
  //  return m_old_weights;
  //}
  uint32_t GetWeightsVectorSize() const {
    return m_weights.size();
  }

protected:
  int m_num_inputs;
  double m_bias;
  //double m_old_bias;
  std::vector<double> m_weights;
  //std::vector<double> m_old_weights;
};

#endif //NODE_H