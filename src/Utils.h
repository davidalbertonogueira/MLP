//============================================================================
// Name : Utils.h
// Author : David Nogueira
//============================================================================
#ifndef UTILS_H
#define UTILS_H

#include "Chrono.h"
#include <stdlib.h>
#include <math.h>
#include <numeric>
#include <chrono>
#ifdef _WIN32
#include <time.h>
#else
#include <sys/time.h>
#endif

#include <algorithm>
#include <iostream>
#include <iterator>
#include <vector>
#include <cmath>



namespace utils {

struct gen_rand {
  double factor;
  double offset;
public:
  gen_rand(double r = 2.0) : factor(r / RAND_MAX), offset(r / 2) {}
  double operator()() {
    return rand() * factor - offset;
  }
};

inline double sigmoid(double x) {
  //Typical sigmoid function created from input x
  //param x: input value
  //return: sigmoided value
  return 1 / (1 + exp(-x));
}

// Derivative of sigmoid function
inline double deriv_sigmoid(double x) {
  return sigmoid(x)*(1 - sigmoid(x));
};

inline void Softmax(std::vector<double> *output) {
  size_t num_elements = output->size();
  std::vector<double> exp_output(num_elements);
  double exp_total = 0.0;
  for (int i = 0; i < num_elements; i++) {
    exp_output[i] = exp((*output)[i]);
    exp_total += exp_output[i];
  }
  for (int i = 0; i < num_elements; i++) {
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