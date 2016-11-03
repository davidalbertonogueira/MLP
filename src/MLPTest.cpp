//============================================================================
// Name : Main.cpp
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
#include "microunit.h"

UNIT(LearnAND) {
  std::cout << "Train AND function with mlp." << std::endl;

  std::vector<TrainingSample> training_set =
  {
    { { 0, 0 },{ 0.0 } },
    { { 0, 1 },{ 0.0 } },
    { { 1, 0 },{ 0.0 } },
    { { 1, 1 },{ 1.0 } },
    { { 1, 1 },{ 1.0 } },
    { { 1, 1 },{ 1.0 } }
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
  size_t num_outputs = training_sample_set_with_bias[0].GetOutputVectorSize();
  MLP my_mlp(num_features, num_outputs, 1, 2, false);
  //Train MLP
  my_mlp.UpdateMiniBatch(training_sample_set_with_bias, 0.5, 500, 0.25);

  for (const auto & training_sample : training_sample_set_with_bias) {
    std::vector<double>  output;
    my_mlp.GetOutput(training_sample.input_vector(), &output);
    bool predicted_output = output[0] > 0.5 ? true : false;
    bool correct_output = training_sample.output_vector()[0] > 0.5 ? true : false;
    ASSERT_TRUE(predicted_output == correct_output);
  }
  std::cout << "Trained with success." << std::endl;
  std::cout << std::endl;
}

UNIT(LearnNAND) {
  std::cout << "Train NAND function with mlp." << std::endl;

  std::vector<TrainingSample> training_set =
  {
    { { 0, 0 },{ 1.0 } },
    { { 0, 1 },{ 1.0 } },
    { { 1, 0 },{ 1.0 } },
    { { 1, 1 },{ 0.0 } },
    { { 1, 1 },{ 0.0 } },
    { { 1, 1 },{ 0.0 } }
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
  size_t num_outputs = training_sample_set_with_bias[0].GetOutputVectorSize();
  MLP my_mlp(num_features, num_outputs, 1, 2, false);
  //Train MLP
  my_mlp.UpdateMiniBatch(training_sample_set_with_bias, 0.5, 500, 0.25);

  for (const auto & training_sample : training_sample_set_with_bias) {
    std::vector<double>  output;
    my_mlp.GetOutput(training_sample.input_vector(), &output);
    bool predicted_output = output[0] > 0.5 ? true : false;
    bool correct_output = training_sample.output_vector()[0] > 0.5 ? true : false;
    ASSERT_TRUE(predicted_output == correct_output);
  }
  std::cout << "Trained with success." << std::endl;
  std::cout << std::endl;
}

UNIT(LearnOR) {
  std::cout << "Train OR function with mlp." << std::endl;

  std::vector<TrainingSample> training_set =
  {
    { { 0, 0 },{ 0.0 } },
    { { 0, 0 },{ 0.0 } },
    { { 0, 0 },{ 0.0 } },
    { { 0, 1 },{ 1.0 } },
    { { 1, 0 },{ 1.0 } },
    { { 1, 1 },{ 1.0 } }
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
  size_t num_outputs = training_sample_set_with_bias[0].GetOutputVectorSize();
  MLP my_mlp(num_features, num_outputs, 1, 2, false);
  //Train MLP
  my_mlp.UpdateMiniBatch(training_sample_set_with_bias, 0.5, 500, 0.25);

  for (const auto & training_sample : training_sample_set_with_bias) {
    std::vector<double>  output;
    my_mlp.GetOutput(training_sample.input_vector(), &output);
    bool predicted_output = output[0] > 0.5 ? true : false;
    bool correct_output = training_sample.output_vector()[0] > 0.5 ? true : false;
    ASSERT_TRUE(predicted_output == correct_output);
  }
  std::cout << "Trained with success." << std::endl;
  std::cout << std::endl;
}

UNIT(LearnNOR) {
  std::cout << "Train NOR function with mlp." << std::endl;

  std::vector<TrainingSample> training_set =
  {
    { { 0, 0 },{ 1.0 } },
    { { 0, 0 },{ 1.0 } },
    { { 0, 0 },{ 1.0 } },
    { { 0, 1 },{ 0.0 } },
    { { 1, 0 },{ 0.0 } },
    { { 1, 1 },{ 0.0 } }
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
  size_t num_outputs = training_sample_set_with_bias[0].GetOutputVectorSize();
  MLP my_mlp(num_features, num_outputs, 1, 2, false);
  //Train MLP
  my_mlp.UpdateMiniBatch(training_sample_set_with_bias, 0.5, 500, 0.25);

  for (const auto & training_sample : training_sample_set_with_bias) {
    std::vector<double>  output;
    my_mlp.GetOutput(training_sample.input_vector(), &output);
    bool predicted_output = output[0] > 0.5 ? true : false;
    bool correct_output = training_sample.output_vector()[0] > 0.5 ? true : false;
    ASSERT_TRUE(predicted_output == correct_output);
  }
  std::cout << "Trained with success." << std::endl;
  std::cout << std::endl;
}

UNIT(LearnXOR) {
  std::cout << "Train XOR function with mlp." << std::endl;

  std::vector<TrainingSample> training_set =
  {
    { { 0, 0 },{ 0.0 } },
    { { 0, 1 },{ 1.0 } },
    { { 1, 0 },{ 1.0 } },
    { { 1, 1 },{ 0.0 } }
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
  size_t num_outputs = training_sample_set_with_bias[0].GetOutputVectorSize();
  MLP my_mlp(num_features, num_outputs, 1, 2, false);
  //Train MLP
  my_mlp.UpdateMiniBatch(training_sample_set_with_bias, 0.5, 50'000, 0.25);

  for (const auto & training_sample : training_sample_set_with_bias) {
    std::vector<double>  output;
    my_mlp.GetOutput(training_sample.input_vector(), &output);
    bool predicted_output = output[0] > 0.5 ? true : false;
    bool correct_output = training_sample.output_vector()[0] > 0.5 ? true : false;
    ASSERT_TRUE(predicted_output == correct_output);
  }
  std::cout << "Trained with success." << std::endl;
  std::cout << std::endl;
}

UNIT(LearnNOT) {
  std::cout << "Train NOT function with mlp." << std::endl;

  std::vector<TrainingSample> training_set =
  {
    { { 0},{ 1.0 } },
    { { 1},{ 0.0 } }
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
  size_t num_outputs = training_sample_set_with_bias[0].GetOutputVectorSize();
  MLP my_mlp(num_features, num_outputs, 1, 2, false);
  //Train MLP
  my_mlp.UpdateMiniBatch(training_sample_set_with_bias, 0.5, 500, 0.25);

  for (const auto & training_sample : training_sample_set_with_bias) {
    std::vector<double>  output;
    my_mlp.GetOutput(training_sample.input_vector(), &output);
    bool predicted_output = output[0] > 0.5 ? true : false;
    bool correct_output = training_sample.output_vector()[0] > 0.5 ? true : false;
    ASSERT_TRUE(predicted_output == correct_output);
  }
  std::cout << "Trained with success." << std::endl;
  std::cout << std::endl;
}

UNIT(LearnX1) {
  std::cout << "Train X1 function with mlp." << std::endl;

  std::vector<TrainingSample> training_set =
  {
    { { 0, 0 },{ 0.0 } },
    { { 0, 1 },{ 0.0 } },
    { { 1, 0 },{ 1.0 } },
    { { 1, 1 },{ 1.0 } }
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
  size_t num_outputs = training_sample_set_with_bias[0].GetOutputVectorSize();
  MLP my_mlp(num_features, num_outputs, 1, 2, false);
  //Train MLP
  my_mlp.UpdateMiniBatch(training_sample_set_with_bias, 0.5, 500, 0.25);

  for (const auto & training_sample : training_sample_set_with_bias) {
    std::vector<double>  output;
    my_mlp.GetOutput(training_sample.input_vector(), &output);
    bool predicted_output = output[0] > 0.5 ? true : false;
    bool correct_output = training_sample.output_vector()[0] > 0.5 ? true : false;
    ASSERT_TRUE(predicted_output == correct_output);
  }
  std::cout << "Trained with success." << std::endl;
  std::cout << std::endl;
}


UNIT(LearnX2) {
  std::cout << "Train X2 function with mlp." << std::endl;

  std::vector<TrainingSample> training_set =
  {
    { { 0, 0 },{ 0.0 } },
    { { 0, 1 },{ 1.0 } },
    { { 1, 0 },{ 0.0 } },
    { { 1, 1 },{ 1.0 } }
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
  size_t num_outputs = training_sample_set_with_bias[0].GetOutputVectorSize();
  MLP my_mlp(num_features, num_outputs, 1, 2, false);
  //Train MLP
  my_mlp.UpdateMiniBatch(training_sample_set_with_bias, 0.5, 500, 0.25);

  for (const auto & training_sample : training_sample_set_with_bias) {
    std::vector<double>  output;
    my_mlp.GetOutput(training_sample.input_vector(), &output);
    bool predicted_output = output[0] > 0.5 ? true : false;
    bool correct_output = training_sample.output_vector()[0] > 0.5 ? true : false;
    ASSERT_TRUE(predicted_output == correct_output);
  }
  std::cout << "Trained with success." << std::endl;
  std::cout << std::endl;
}

int main() {
  microunit::UnitTester::Run();
  return 0;
}