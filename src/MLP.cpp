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

#include "easylogging++.h"


//desired call sintax :  MLP({64*64,20,4}, {"sigmoid", "linear"},
MLP::MLP(const std::vector<uint64_t> & layers_nodes,
         const std::vector<std::string> & layers_activfuncs,
         bool use_constant_weight_init,
         double constant_weight_init,
         bool use_softmax_on_each_output ) {
  assert(layers_nodes.size() >= 2);
  assert(layers_activfuncs.size() + 1 == layers_nodes.size());

  CreateMLP(layers_nodes,
            layers_activfuncs,
            use_constant_weight_init,
            constant_weight_init,
            use_softmax_on_each_output );

  callInspectorAfterCreateNetwork();
};


void MLP::callInspectorAfterCreateNetwork(){
    for( std::shared_ptr<MlpInspector> instance : inspectorVector ){
        instance->afterCreateNetwork( m_layers );
    }
}

MLP::MLP(const std::string & filename) {
  LoadMLPNetwork(filename);
  callInspectorAfterCreateNetwork();
}

MLP::~MLP() {
    callInspectorBeforeDestroyNetwork();
    m_num_inputs = 0;
    m_num_outputs = 0;
    m_num_hidden_layers = 0;
    m_layers_nodes.clear();
    m_layers.clear();
};

void MLP::callInspectorBeforeDestroyNetwork(){
    for(std::shared_ptr<MlpInspector> instance : inspectorVector){
        instance->beforeDestroyNetwork( m_layers );
    }
}

void MLP::CreateMLP(const std::vector<uint64_t> & layers_nodes,
                    const std::vector<std::string> & layers_activfuncs,
                    bool use_constant_weight_init,
                    double constant_weight_init,
                    bool use_softmax_on_each_output,
                    std::string error_function ) {

    m_layers_nodes = layers_nodes;
    m_num_inputs = m_layers_nodes[0];
    m_num_outputs = m_layers_nodes[m_layers_nodes.size() - 1];
    m_num_hidden_layers = m_layers_nodes.size() - 2;
    MLP::use_softmax_on_each_output = use_softmax_on_each_output;
    error_function_pair = utils::ErrorFunctionsManager::Singleton().GetErrorFunctionPair( error_function );

    for (size_t i = 0; i < m_layers_nodes.size() - 1; i++) {
    m_layers.emplace_back(Layer(m_layers_nodes[i],
                                m_layers_nodes[i + 1],
                                layers_activfuncs[i],
                                use_constant_weight_init,
                                constant_weight_init));
    }
}

void MLP::SaveMLPNetwork(const std::string & filename)const {
  FILE * file;
  file = fopen(filename.c_str(), "wb");
  fwrite(&m_num_inputs, sizeof(m_num_inputs), 1, file);
  fwrite(&m_num_outputs, sizeof(m_num_outputs), 1, file);
  fwrite(&m_num_hidden_layers, sizeof(m_num_hidden_layers), 1, file);
  if (!m_layers_nodes.empty())
    fwrite(&m_layers_nodes[0], sizeof(m_layers_nodes[0]), m_layers_nodes.size(), file);
  for (size_t i = 0; i < m_layers.size(); i++) {
    m_layers[i].SaveLayer(file);
  }
  fclose(file);
};


void MLP::LoadMLPNetwork(const std::string & filename) {
  m_layers_nodes.clear();
  m_layers.clear();

  FILE * file;
  file = fopen(filename.c_str(), "rb");
  fread(&m_num_inputs, sizeof(m_num_inputs), 1, file);
  fread(&m_num_outputs, sizeof(m_num_outputs), 1, file);
  fread(&m_num_hidden_layers, sizeof(m_num_hidden_layers), 1, file);
  m_layers_nodes.resize(m_num_hidden_layers + 2);
  if (!m_layers_nodes.empty())
    fread(&m_layers_nodes[0], sizeof(m_layers_nodes[0]), m_layers_nodes.size(), file);
  m_layers.resize(m_layers_nodes.size() - 1);
  for (size_t i = 0; i < m_layers.size(); i++) {
    m_layers[i].LoadLayer(file);
  }
  fclose(file);
};

void MLP::GetOutput(const std::vector<double> &input,
                    std::vector<double> * output,
                    std::vector<std::vector<double>> * all_layers_activations ) const {
  assert(input.size() == m_num_inputs);
  int temp_size;
  if (m_num_hidden_layers == 0)
    temp_size = m_num_outputs;
  else
    temp_size = m_layers_nodes[1];

  std::vector<double> temp_in(m_num_inputs, 0.0);
  std::vector<double> temp_out(temp_size, 0.0);
  temp_in = input;

  for (size_t i = 0; i < m_layers.size(); ++i) {
    if (i > 0) {
      //Store this layer activation
      if (all_layers_activations != nullptr)
        all_layers_activations->emplace_back(std::move(temp_in));

      temp_in.clear();
      temp_in = temp_out;
      temp_out.clear();
      temp_out.resize(m_layers[i].GetOutputSize());
    }
    m_layers[i].GetOutputAfterActivationFunction(temp_in, temp_out);
  }

  if( use_softmax_on_each_output )
  {
      if (temp_out.size() > 1)
        utils::Softmax(&temp_out);
  }
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
}

void MLP::log_weights( const std::string & message ) const
{
    int layer_i = -1;
    int node_i = -1;
    LOG(DEBUG) << message << ": ";
    for (const auto & layer : m_layers) {
        layer_i++;
        node_i = -1;
        LOG(DEBUG) << "Layer " << layer_i << " :";
        for (const auto & node : layer.GetNodes()) {
          node_i++;
          std::stringstream node_line;
          node_line << "\tNode " << node_i << " :\t";
          for (auto m_weightselement : node.GetWeights()) {
            node_line << m_weightselement << "\t";
          }
          LOG(DEBUG) << node_line.str();
        }
    }
}


void MLP::log_input_output( TrainingSample training_sample_with_bias,
                        std::vector<double> predicted_output ) const
{
    std::stringstream temp_training;
    temp_training << training_sample_with_bias << "\t\t";

    temp_training << "Predicted output: [";
    for (size_t i = 0; i < predicted_output.size(); i++) {
      if (i != 0)
        temp_training << ", ";
      temp_training << predicted_output[i];
    }
    temp_training << "]";

    LOG(INFO) << temp_training.str();
}

void MLP::Train(const std::vector<TrainingSample> &training_sample_set_with_bias,
                          double learning_rate,
                          int max_iterations,
                          double min_error_cost,
                          bool output_log) {

  if( output_log )
        log_weights( "Starting weights" );

  callInspectorBeforeTraining();
  int i = 0;
  double current_iteration_cost_function = 0.0;

  for (i = 0; i < max_iterations; i++) {
    current_iteration_cost_function = 0.0;
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

      if (output_log && ((i % (max_iterations / 10)) == 0)) {
          log_input_output( training_sample_with_bias, predicted_output );
      }

      for (size_t j = 0; j < predicted_output.size(); j++) {
          current_iteration_cost_function += error_function_pair.first( correct_output[j], predicted_output[j] );
          deriv_error_output[j] = -error_function_pair.second( correct_output[j], predicted_output[j] );
      }

      UpdateWeights(all_layers_activations,
                    deriv_error_output,
                    learning_rate);
    }


    if (output_log && ((i % (max_iterations / 10)) == 0))
      LOG(INFO) << "Iteration " << i << " cost function f(error): "
      << current_iteration_cost_function;

    if (current_iteration_cost_function < min_error_cost)
      break;
  }
  callInspectorAfterTraining();
  LOG(INFO) << "Iteration " << i << " cost function f(error): "
    << current_iteration_cost_function;

  LOG(INFO) << "******************************";
  LOG(INFO) << "******* TRAINING ENDED *******";
  LOG(INFO) << "******* " << i << " iters *******";
  LOG(INFO) << "******************************";

  if( output_log )
      log_weights( "Final weights" );
};


void MLP::callInspectorBeforeTraining(){
    for( std::shared_ptr<MlpInspector> instance : inspectorVector ){
        instance->onEnterTraining( m_layers );
    }
}

void MLP::callInspectorAfterTraining(){
    for( std::shared_ptr<MlpInspector> instance : inspectorVector ){
        instance->onEndTraining( m_layers );
    }
}

size_t MLP::GetNumLayers()
{
    return m_layers.size();
}

std::vector<std::vector<double>> MLP::GetLayerWeights( size_t layer_i )
{
    std::vector<std::vector<double>> ret_val;
    // check parameters
    if( 0 <= layer_i && layer_i < m_layers.size() )
    {
        Layer current_layer = m_layers[layer_i];
        for( Node & node : current_layer.GetNodesChangeable() )
        {
            ret_val.push_back( node.GetWeights() );
        }
        return ret_val;
    }
    else
        throw new std::logic_error("Incorrect layer number in GetLayerWeights call");

}

void MLP::SetLayerWeights( size_t layer_i, std::vector<std::vector<double>> & weights )
{
    // check parameters
    if( 0 <= layer_i && layer_i < m_layers.size() )
    {
        m_layers[layer_i].SetWeights( weights );
    }
    else
        throw new std::logic_error("Incorrect layer number in SetLayerWeights call");
}

void MLP::AddInspector( std::shared_ptr<MlpInspector> mlpInspector )
{
    MLP::inspectorVector.push_back( mlpInspector );

}


