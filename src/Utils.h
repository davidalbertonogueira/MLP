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

void Softmax(std::vector<double> *output) {
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

void  GetIdMaxElement(const std::vector<double> &output, size_t * class_id) {
  *class_id = std::distance(output.begin(),
                                   std::max_element(output.begin(),
                                                    output.end()));
}

class Chronometer {
public:
  Chronometer() {
    time_span = std::chrono::steady_clock::duration::zero();
  };
  virtual ~Chronometer() {};

  void GetTime() {
    clock_begin = std::chrono::steady_clock::now();
  }
  void StopTime() {
    std::chrono::steady_clock::time_point clock_end = std::chrono::steady_clock::now();
    time_span += clock_end - clock_begin;
  }
  //Return elapsed time in seconds
  double GetElapsedTime() {
    return double(time_span.count()) *
      std::chrono::steady_clock::period::num / std::chrono::steady_clock::period::den;
  }
protected:
  std::chrono::steady_clock::time_point clock_begin;
  std::chrono::steady_clock::duration time_span;
};

}
#endif // UTILS_H