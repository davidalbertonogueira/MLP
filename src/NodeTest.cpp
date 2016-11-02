//============================================================================
// Name : NodeTest.cpp
// Author : David Nogueira
//============================================================================
#include "Node.h"
#include "Sample.h"
#include "Utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <algorithm>
#include "microunit.h"

namespace {
void Train(Node & node,
           const std::vector<TrainingSample> &training_sample_set_with_bias,
           double learning_rate,
           int max_iterations,
           bool use_constant_weight_init = true,
           double constant_weight_init = 0.5) {

  //initialize weight vector
  node.WeightInitialization(training_sample_set_with_bias[0].GetInputVectorSize(),
                            use_constant_weight_init,
                            constant_weight_init);

  std::cout << "Starting weights:\t";
  for (auto m_weightselement : node.GetWeights())
    std::cout << m_weightselement << "\t";
  std::cout << std::endl;

  for (int i = 0; i < max_iterations; i++) {
    int error_count = 0;
    for (auto & training_sample_with_bias : training_sample_set_with_bias) {
      bool prediction;
      node.GetBooleanOutput(training_sample_with_bias.input_vector(), &prediction);
      bool correct_output = training_sample_with_bias.output_vector()[0] > 0.5 ? true : false;
      if (prediction != correct_output) {
        error_count++;
        double error = (correct_output ? 1 : 0) - (prediction ? 1 : 0);
        node.UpdateWeights(training_sample_with_bias.input_vector(),
                           learning_rate,
                           error);
      }
    }
    if (error_count == 0) break;
  }

  std::cout << "Final weights:\t\t";
  for (auto m_weightselement : node.GetWeights())
    std::cout << m_weightselement << "\t";
  std::cout << std::endl;
};
}

UNIT(LearnAND) {
  std::cout << "Train AND function with Node." << std::endl;

  std::vector<TrainingSample> training_set =
  {
    { { 0, 0 },{0.0} },
    { { 0, 1 },{0.0} },
    { { 1, 0 },{0.0} },
    { { 1, 1 },{1.0} }
  };
  bool bias_already_in = false;
  std::vector<TrainingSample> training_sample_set_with_bias(training_set);
  //set up bias
  if (!bias_already_in) {
    for (auto & training_sample_with_bias : training_sample_set_with_bias) {
      training_sample_with_bias.AddBiasValue(1);
    }
  }

  size_t num_examples = training_sample_set_with_bias.size();
  size_t num_features = training_sample_set_with_bias[0].GetInputVectorSize();
  Node my_node(num_features);
  Train(my_node, training_sample_set_with_bias, 0.1, 100);

  for (const auto & training_sample : training_sample_set_with_bias) {
    bool class_id;
    my_node.GetBooleanOutput(training_sample.input_vector(), &class_id);
    bool correct_output = training_sample.output_vector()[0] > 0 ? true : false;
    ASSERT_TRUE(class_id == correct_output);
  }
  std::cout << "Trained with success." << std::endl;
  std::cout << std::endl;
}

UNIT(LearnNAND) {
  std::cout << "Train NAND function with Node." << std::endl;

  std::vector<TrainingSample> training_set =
  {
    { { 0, 0 },{1.0} },
    { { 0, 1 },{1.0} },
    { { 1, 0 },{1.0} },
    { { 1, 1 },{0.0} }
  };
  bool bias_already_in = false;
  std::vector<TrainingSample> training_sample_set_with_bias(training_set);
  //set up bias
  if (!bias_already_in) {
    for (auto & training_sample_with_bias : training_sample_set_with_bias) {
      training_sample_with_bias.AddBiasValue(1);
    }
  }
  size_t num_examples = training_sample_set_with_bias.size();
  size_t num_features = training_sample_set_with_bias[0].GetInputVectorSize();
  Node my_node(num_features);
  Train(my_node, training_sample_set_with_bias, 0.1, 100);

  for (const auto & training_sample : training_sample_set_with_bias) {
    bool class_id;
    my_node.GetBooleanOutput(training_sample.input_vector(), &class_id);
    bool correct_output = training_sample.output_vector()[0] > 0 ? true : false;
    ASSERT_TRUE(class_id == correct_output);
  }
  std::cout << "Trained with success." << std::endl;
  std::cout << std::endl;
}

UNIT(LearnOR) {
  std::cout << "Train OR function with Node." << std::endl;

  std::vector<TrainingSample> training_set =
  {
    { { 0, 0 },{0.0} },
    { { 0, 1 },{1.0} },
    { { 1, 0 },{1.0} },
    { { 1, 1 },{1.0} }
  };
  bool bias_already_in = false;
  std::vector<TrainingSample> training_sample_set_with_bias(training_set);
  //set up bias
  if (!bias_already_in) {
    for (auto & training_sample_with_bias : training_sample_set_with_bias) {
      training_sample_with_bias.AddBiasValue(1);
    }
  }
  size_t num_examples = training_sample_set_with_bias.size();
  size_t num_features = training_sample_set_with_bias[0].GetInputVectorSize();
  Node my_node(num_features);
  Train(my_node, training_sample_set_with_bias, 0.1, 100);

  for (const auto & training_sample : training_sample_set_with_bias) {
    bool class_id;
    my_node.GetBooleanOutput(training_sample.input_vector(), &class_id);
    bool correct_output = training_sample.output_vector()[0] > 0 ? true : false;
    ASSERT_TRUE(class_id == correct_output);
  }
  std::cout << "Trained with success." << std::endl;
  std::cout << std::endl;
}
UNIT(LearnNOR) {
  std::cout << "Train NOR function with Node." << std::endl;

  std::vector<TrainingSample> training_set =
  {
    { { 0, 0 },{1.0} },
    { { 0, 1 },{0.0} },
    { { 1, 0 },{0.0} },
    { { 1, 1 },{0.0} }
  };
  bool bias_already_in = false;
  std::vector<TrainingSample> training_sample_set_with_bias(training_set);
  //set up bias
  if (!bias_already_in) {
    for (auto & training_sample_with_bias : training_sample_set_with_bias) {
      training_sample_with_bias.AddBiasValue(1);
    }
  }
  size_t num_examples = training_sample_set_with_bias.size();
  size_t num_features = training_sample_set_with_bias[0].GetInputVectorSize();
  Node my_node(num_features);
  Train(my_node, training_sample_set_with_bias, 0.1, 100);

  for (const auto & training_sample : training_sample_set_with_bias) {
    bool class_id;
    my_node.GetBooleanOutput(training_sample.input_vector(), &class_id);
    bool correct_output = training_sample.output_vector()[0] > 0 ? true : false;
    ASSERT_TRUE(class_id == correct_output);
  }
  std::cout << "Trained with success." << std::endl;
  std::cout << std::endl;
}

UNIT(LearnNOT) {
  std::cout << "Train NOT function with Node." << std::endl;

  std::vector<TrainingSample> training_set =
  {
    { { 0 },{1.0} },
    { { 1 },{0.0}}
  };
  bool bias_already_in = false;
  std::vector<TrainingSample> training_sample_set_with_bias(training_set);
  //set up bias
  if (!bias_already_in) {
    for (auto & training_sample_with_bias : training_sample_set_with_bias) {
      training_sample_with_bias.AddBiasValue(1);
    }
  }
  size_t num_examples = training_sample_set_with_bias.size();
  size_t num_features = training_sample_set_with_bias[0].GetInputVectorSize();
  Node my_node(num_features);
  Train(my_node, training_sample_set_with_bias, 0.1, 100);

  for (const auto & training_sample : training_sample_set_with_bias) {
    bool class_id;
    my_node.GetBooleanOutput(training_sample.input_vector(), &class_id);
    bool correct_output = training_sample.output_vector()[0] > 0 ? true : false;
    ASSERT_TRUE(class_id == correct_output);
  }
  std::cout << "Trained with success." << std::endl;
  std::cout << std::endl;
}

UNIT(LearnXOR) {
  std::cout << "Train XOR function with Node." << std::endl;

  std::vector<TrainingSample> training_set =
  {
    { { 0, 0 },{0.0} },
    { { 0, 1 },{1.0} },
    { { 1, 0 },{1.0} },
    { { 1, 1 },{0.0} }
  };
  bool bias_already_in = false;
  std::vector<TrainingSample> training_sample_set_with_bias(training_set);
  //set up bias
  if (!bias_already_in) {
    for (auto & training_sample_with_bias : training_sample_set_with_bias) {
      training_sample_with_bias.AddBiasValue(1);
    }
  }
  size_t num_examples = training_sample_set_with_bias.size();
  size_t num_features = training_sample_set_with_bias[0].GetInputVectorSize();
  Node my_node(num_features);
  Train(my_node, training_sample_set_with_bias, 0.1, 100);

  for (const auto & training_sample : training_sample_set_with_bias) {
    bool class_id;
    my_node.GetBooleanOutput(training_sample.input_vector(), &class_id);
    bool correct_output = training_sample.output_vector()[0] > 0 ? true : false;
    if (class_id != correct_output) {
      std::cout << "Failed to train. " <<
        " A simple perceptron cannot learn the XOR function." << std::endl;
      FAIL();
    }
  }
  std::cout << "Trained with success." << std::endl;
  std::cout << std::endl;
}

int main() {
  microunit::UnitTester::Run();
  return 0;
}