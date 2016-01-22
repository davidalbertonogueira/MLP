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

class Layer {
public:
  Layer() {
    m_num_nodes = 0;
    m_nodes.clear();
  };


  Layer(int num_nodes, int num_inputs_per_node) {
    m_num_nodes = num_nodes;
    m_nodes = std::vector<Node>(num_nodes, Node(num_inputs_per_node));
  };

  ~Layer() {

  };
protected:
  int m_num_nodes;
  std::vector<Node> m_nodes;
};

#endif //LAYER_H