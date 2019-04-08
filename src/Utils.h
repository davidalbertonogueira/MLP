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
#include <exception>


#include "Chrono.h"
#ifdef _WIN32
#include <time.h>
#else
#include <sys/time.h>
#endif

namespace utils {

//Typical error function
// error = [measure1 - measure2]^2
inline double quadratic_error(double measure1, double measure2){
    return (std::pow) (measure1 - measure2, 2 );
}

// derivative of the error function
inline double deriv_quadratic_error(double measure1, double measure2){
    return 2 * (measure1 - measure2);
}

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

inline double relu(double x) {
    if( x > 0 )
        return x;
    else
        return 0.0L;
}

inline double deriv_relu(double x){
    if( x > 0 )
        return 1.0L;
    else
        return 0.0L;
}


using functionWithDeriv = std::pair<std::function<double(double)>,
                                    std::function<double(double)> >;
using functionTwoArgDeriv = std::pair<std::function<double(double,double)>,
                                      std::function<double(double,double)> >;

struct ErrorFunctionsManager {
    functionTwoArgDeriv GetErrorFunctionPair( const std::string & error_func_name ) {
        auto iter = error_functions_map.find( error_func_name );
        if( iter != error_functions_map.end() )
            return iter->second;
        else
            throw std::runtime_error("Unknown error function: " +  error_func_name );
    }

    static ErrorFunctionsManager & Singleton(){
        static ErrorFunctionsManager instance;
        return instance;
    }

private:
    void AddNewPair( std::string function_name,
                    std::function<double(double, double)> function,
                    std::function<double(double, double)> deriv) {
        error_functions_map.insert( std::make_pair( function_name,
                std::make_pair( function, deriv ) ) );
    }

    ErrorFunctionsManager() {
        AddNewPair("error", quadratic_error, deriv_quadratic_error );
    }

    std::unordered_map<std::string,
           functionTwoArgDeriv> error_functions_map;
};


struct ActivationFunctionsManager {
  functionWithDeriv GetActivationFunctionPair(const std::string & activation_name ) {
    auto iter = activation_functions_map.find(activation_name);
    if (iter != activation_functions_map.end())
      return iter->second;
    else
      throw std::runtime_error("Unknown activation function: " + activation_name );
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
    AddNewPair("tanh",    hyperbolic_tan, deriv_hyperbolic_tan);
    AddNewPair("linear",  linear, deriv_linear);
    AddNewPair("relu",    relu, deriv_relu);
  };

  std::unordered_map<std::string, functionWithDeriv > activation_functions_map;
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
