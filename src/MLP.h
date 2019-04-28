//============================================================================
// Name : MLP.cpp
// Author : David Nogueira
//============================================================================
#ifndef MLP_H
#define MLP_H

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <exception>
#include <memory>

#include "Layer.h"
#include "Sample.h"
#include "Utils.h"
#include "MlpInspector.h"


class MLP {
public:
  //desired call syntax :  MLP({64*64,20,4}, {"sigmoid", "linear"},
  MLP(const std::vector<uint64_t> & layers_nodes,
      const std::vector<std::string> & layers_activfuncs,
      bool use_constant_weight_init = false,
      double constant_weight_init = 0.5,
      bool use_softmax_on_each_output = true );
  MLP(const std::string & filename);
  ~MLP();

  void SaveMLPNetwork(const std::string & filename)const;
  void LoadMLPNetwork(const std::string & filename);

  void GetOutput(const std::vector<double> &input,
                 std::vector<double> * output,
                 std::vector<std::vector<double>> * all_layers_activations = nullptr ) const;
  void GetOutputClass(const std::vector<double> &output, size_t * class_id) const;

  void Train(const std::vector<TrainingSample> &training_sample_set_with_bias,
                       double learning_rate,
                       int max_iterations = 5000,
                       double min_error_cost = 0.001,
                       bool output_log = true);
  size_t GetNumLayers();
  std::vector<std::vector<double>> GetLayerWeights( size_t layer_i );
  void SetLayerWeights( size_t layer_i, std::vector<std::vector<double>> & weights );

  void AddInspector( std::shared_ptr<MlpInspector> mlpInspector );

protected:
  void UpdateWeights(const std::vector<std::vector<double>> & all_layers_activations,
                     const std::vector<double> &error,
                     double learning_rate);
private:
  void log_weights( const std::string & message ) const;
  void log_input_output(  TrainingSample training_sample_with_bias,
                          std::vector<double> predicted_output ) const;
  void CreateMLP(const std::vector<uint64_t> & layers_nodes,
                 const std::vector<std::string> & layers_activfuncs,
                 bool use_constant_weight_init,
                 double constant_weight_init = 0.5,
                 bool use_softmax_on_each_output = true,
                 std::string error_function = "error");

  void callInspectorBeforeTraining();
  void callInspectorBeforeTrainingSample( const TrainingSample &training_sample );
  void callInspectorAfterTrainingSample( const TrainingSample &training_sample );
  void callInspectorAfterTraining();

  size_t m_num_inputs{ 0 };
  int m_num_outputs{ 0 };
  int m_num_hidden_layers{ 0 };
  std::vector<uint64_t> m_layers_nodes;
  std::vector<Layer> m_layers;
  bool use_softmax_on_each_output = true;
  utils::functionTwoArgDeriv error_function_pair;
  std::vector<std::shared_ptr<MlpInspector>> inspectorVector;

};

#endif //MLP_H






