//============================================================================
// Name : Utils.h
// Author : David Nogueira
//============================================================================
#ifndef UTILS_H
#define UTILS_H

#include <stdlib.h>
#include <math.h>
#include <numeric>
#include <chrono>
#include <string>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <iterator>
#include <unordered_map>
#include <vector>
#include <cmath>
#include <functional>
#include <typeinfo>
#include <typeindex>
#include <cassert>

#include "Chrono.h"
#ifdef _WIN32
#include <time.h>
#else
#include <sys/time.h>
#endif

namespace utils {
//Typical sigmoid function created from input x
//Returns the sigmoided value
inline double sigmoid(double x) {
  return 1 / (1 + exp(-x));
}

// Derivative of sigmoid function
inline double deriv_sigmoid(double x) {
  return sigmoid(x)*(1 - sigmoid(x));
};

//Compute hyperbolic tangent (tanh)
//Returns the hyperbolic tangent of x.
inline double hyperbolic_tan(double x) {
  return (tanh)(x);
}

// Derivative of hyperbolic tangent function
inline double deriv_hyperbolic_tan(double x) {
  return 1 - (std::pow)(hyperbolic_tan(x), 2);
};

inline double linear(double x) {
  return x;
}

// Derivative of linear function
inline double deriv_linear(double x) {
  return 1;
};

struct ActivationFunctionsManager {
  bool GetActivationFunctionPair(const std::string & activation_name,
                                    std::pair<std::function<double(double)>,
                                    std::function<double(double)> > **pair) {
    auto iter = activation_functions_map.find(activation_name);
    if (iter != activation_functions_map.end())
      *pair = &(iter->second);
    else
      return false;
    return true;
  }

  static ActivationFunctionsManager & Singleton() {
    static ActivationFunctionsManager instance;
    return instance;
  }
private:
  void AddNewPair(std::string function_name,
                  std::function<double(double)> function,
                  std::function<double(double)> deriv_function) {
    activation_functions_map.insert(std::make_pair(function_name,
                                                   std::make_pair(function,
                                                                  deriv_function)));
  };

  ActivationFunctionsManager() {
    AddNewPair("sigmoid", sigmoid, deriv_sigmoid);
    AddNewPair("tanh", hyperbolic_tan, deriv_hyperbolic_tan);
    AddNewPair("linear", linear, deriv_linear);
  };

  std::unordered_map<std::string,
    std::pair<std::function<double(double)>, std::function<double(double)> > >
    activation_functions_map;
};

struct gen_rand {
  double factor;
  double offset;
public:
  gen_rand(double r = 2.0) : factor(r / RAND_MAX), offset(r / 2) {}
  double operator()() {
    return rand() * factor - offset;
  }
};

inline void Softmax(std::vector<double> *output) {
  size_t num_elements = output->size();
  std::vector<double> exp_output(num_elements);
  double exp_total = 0.0;
  for (size_t i = 0; i < num_elements; i++) {
    exp_output[i] = exp((*output)[i]);
    exp_total += exp_output[i];
  }
  for (size_t i = 0; i < num_elements; i++) {
    (*output)[i] = exp_output[i] / exp_total;
  }
}

inline void  GetIdMaxElement(const std::vector<double> &output, size_t * class_id) {
  *class_id = std::distance(output.begin(),
                            std::max_element(output.begin(),
                                             output.end()));
}
}
#endif // UTILS_H
