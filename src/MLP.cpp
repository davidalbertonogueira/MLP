//============================================================================
// Name : MLP.cpp
// Author : David Nogueira
//============================================================================
#include "MLP.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <algorithm>

bool MLP::ExportNNWeights(std::vector<double> *weights) const {
  return true;
};
bool MLP::ImportNNWeights(const std::vector<double> & weights) {
  return true;
};

void MLP::GetOutput(const std::vector<double> &input,
                    std::vector<double> * output,
                    std::vector<std::vector<double>> * all_layers_activations) const {
  assert(input.size() == m_num_inputs);
  int temp_size;
  if (m_num_hidden_layers == 0)
    temp_size = m_num_outputs;
  else
    temp_size = m_num_nodes_per_hidden_layer;

  std::vector<double> temp_in(m_num_inputs, 0.0);
  std::vector<double> temp_out(temp_size, 0.0);
  temp_in = input;

  //m_layers.size() equals (m_num_hidden_layers + 1)
  for (int i = 0; i < (m_num_hidden_layers + 1); ++i) {
    if (i > 0) {
      //Store this layer activation
      if (all_layers_activations != nullptr)
        all_layers_activations->emplace_back(std::move(temp_in));

      temp_in.clear();
      temp_in = temp_out;
      temp_out.clear();
      temp_out.resize((i == m_num_hidden_layers) ?
                      m_num_outputs :
                      m_num_nodes_per_hidden_layer);
    }
    m_layers[i].GetOutputAfterSigmoid(temp_in, &temp_out);
  }

  if (temp_out.size() > 1)
    utils::Softmax(&temp_out);
  *output = temp_out;

  //Add last layer activation
  if (all_layers_activations != nullptr)
    all_layers_activations->emplace_back(std::move(temp_in));
}

void MLP::GetOutputClass(const std::vector<double> &output, size_t * class_id) const {
  utils::GetIdMaxElement(output, class_id);
}

void MLP::UpdateWeights(const std::vector<std::vector<double>> & all_layers_activations,
                        const std::vector<double> &deriv_error,
                        double learning_rate) {

  std::vector<double> temp_deriv_error = deriv_error;
  std::vector<double> deltas{};
  //m_layers.size() equals (m_num_hidden_layers + 1)
  for (int i = m_num_hidden_layers; i >= 0; --i) {
    m_layers[i].UpdateWeights(all_layers_activations[i], temp_deriv_error, learning_rate, &deltas);
    if (i > 0) {
      temp_deriv_error.clear();
      temp_deriv_error = std::move(deltas);
      deltas.clear();
    }
  }
};

void MLP::UpdateMiniBatch(const std::vector<TrainingSample> &training_sample_set_with_bias,
                          double learning_rate,
                          int max_iterations,
                          double min_error_cost) {
  int num_examples = training_sample_set_with_bias.size();
  int num_features = training_sample_set_with_bias[0].GetInputVectorSize();

  {
    int layer_i = -1;
    int node_i = -1;
    std::cout << "Starting weights:" << std::endl;
    for (const auto & layer : m_layers) {
      layer_i++;
      node_i = -1;
      std::cout << "Layer " << layer_i << " :" << std::endl;
      for (const auto & node : layer.GetNodes()) {
        node_i++;
        std::cout << "\tNode " << node_i << " :\t";
        for (auto m_weightselement : node.GetWeights()) {
          std::cout << m_weightselement << "\t";
        }
        std::cout << std::endl;
      }
    }
  }
  size_t i = 0;
  for ( i = 0; i < max_iterations; i++) {
    //std::cout << "******************************" << std::endl;
    //std::cout << "******** ITER " << i << std::endl;
    //std::cout << "******************************" << std::endl;
    double current_iteration_cost_function = 0.0;
    for (auto & training_sample_with_bias : training_sample_set_with_bias) {
      std::vector<double> predicted_output;
      std::vector< std::vector<double> > all_layers_activations;
      GetOutput(training_sample_with_bias.input_vector(),
                &predicted_output,
                &all_layers_activations);
      const std::vector<double> &  correct_output =
        training_sample_with_bias.output_vector();

      assert(correct_output.size() == predicted_output.size());
      std::vector<double> deriv_error_output(predicted_output.size());

      //std::cout << training_sample_with_bias << "\t\t";
      //{
      //  std::cout << "Predicted output: [";
      //  for (int i = 0; i < predicted_output.size(); i++) {
      //    if (i != 0)
      //      std::cout << ", ";
      //    std::cout << predicted_output[i];
      //  }
      //  std::cout << "]" << std::endl;
      //}

      for (int j = 0; j < predicted_output.size(); j++) {
        current_iteration_cost_function +=
          (std::pow)((correct_output[j] - predicted_output[j]), 2);
        deriv_error_output[j] =
          -2 * (correct_output[j] - predicted_output[j]);
      }

      UpdateWeights(all_layers_activations,
                    deriv_error_output,
                    learning_rate);
    }

    if((i% (max_iterations/100))==0)
    std::cout << "Iteration "<< i << " cost function f(error): "
      << current_iteration_cost_function << std::endl;
    if (current_iteration_cost_function < min_error_cost)
      break;

    //{
    //  int layer_i = -1;
    //  int node_i = -1;
    //  std::cout << "Current weights:" << std::endl;
    //  for (const auto & layer : m_layers) {
    //    layer_i++;
    //    node_i = -1;
    //    std::cout << "Layer " << layer_i << " :" << std::endl;
    //    for (const auto & node : layer.GetNodes()) {
    //      node_i++;
    //      std::cout << "\tNode " << node_i << " :\t";
    //      for (auto m_weightselement : node.GetWeights()) {
    //        std::cout << m_weightselement << "\t";
    //      }
    //      std::cout << std::endl;
    //    }
    //  }
    //}
  }

  std::cout << "******************************" << std::endl;
  std::cout << "******* TRAINING ENDED *******" << std::endl;
  std::cout << "******* " << i << " iters *******" << std::endl;
  std::cout << "******************************" << std::endl;
  {
    int layer_i = -1;
    int node_i = -1;
    std::cout << "Final weights:" << std::endl;
    for (const auto & layer : m_layers) {
      layer_i++;
      node_i = -1;
      std::cout << "Layer " << layer_i << " :" << std::endl;
      for (const auto & node : layer.GetNodes()) {
        node_i++;
        std::cout << "\tNode " << node_i << " :\t";
        for (auto m_weightselement : node.GetWeights()) {
          std::cout << m_weightselement << "\t";
        }
        std::cout << std::endl;
      }
    }
  }
};


