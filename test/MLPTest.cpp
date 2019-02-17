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
#include "easylogging++.h"

INITIALIZE_EASYLOGGINGPP

UNIT(LearnAND) {
  LOG(INFO) << "Train AND function with mlp." << std::endl;

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

  size_t num_features = training_sample_set_with_bias[0].GetInputVectorSize();
  size_t num_outputs = training_sample_set_with_bias[0].GetOutputVectorSize();
  MLP my_mlp({ num_features, 2 ,num_outputs }, { "sigmoid", "linear" });
  //Train MLP
  my_mlp.Train(training_sample_set_with_bias, 0.5, 500, 0.25);

  for (const auto & training_sample : training_sample_set_with_bias) {
    std::vector<double>  output;
    my_mlp.GetOutput(training_sample.input_vector(), &output);
    for (size_t i = 0; i < num_outputs; i++) {
      bool predicted_output = output[i] > 0.5 ? true : false;
      bool correct_output = training_sample.output_vector()[i] > 0.5 ? true : false;
      ASSERT_TRUE(predicted_output == correct_output);
    }
  }
  LOG(INFO) << "Trained with success." << std::endl;
}

UNIT(LearnNAND) {
  LOG(INFO) << "Train NAND function with mlp." << std::endl;

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

  size_t num_features = training_sample_set_with_bias[0].GetInputVectorSize();
  size_t num_outputs = training_sample_set_with_bias[0].GetOutputVectorSize();
  MLP my_mlp({ num_features, 2 ,num_outputs }, { "sigmoid", "linear" });
  //Train MLP
  my_mlp.Train(training_sample_set_with_bias, 0.5, 500, 0.25);

  for (const auto & training_sample : training_sample_set_with_bias) {
    std::vector<double>  output;
    my_mlp.GetOutput(training_sample.input_vector(), &output);
    for (size_t i = 0; i < num_outputs; i++) {
      bool predicted_output = output[i] > 0.5 ? true : false;
      bool correct_output = training_sample.output_vector()[i] > 0.5 ? true : false;
      ASSERT_TRUE(predicted_output == correct_output);
    }
  }
  LOG(INFO) << "Trained with success." << std::endl;
}

UNIT(LearnOR) {
  LOG(INFO) << "Train OR function with mlp." << std::endl;

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

  size_t num_features = training_sample_set_with_bias[0].GetInputVectorSize();
  size_t num_outputs = training_sample_set_with_bias[0].GetOutputVectorSize();
  MLP my_mlp({ num_features, 2 ,num_outputs }, { "sigmoid", "linear" });
  //Train MLP
  my_mlp.Train(training_sample_set_with_bias, 0.5, 500, 0.25);

  for (const auto & training_sample : training_sample_set_with_bias) {
    std::vector<double>  output;
    my_mlp.GetOutput(training_sample.input_vector(), &output);
    for (size_t i = 0; i < num_outputs; i++) {
      bool predicted_output = output[i] > 0.5 ? true : false;
      bool correct_output = training_sample.output_vector()[i] > 0.5 ? true : false;
      ASSERT_TRUE(predicted_output == correct_output);
    }
  }
  LOG(INFO) << "Trained with success." << std::endl;
}

UNIT(LearnNOR) {
  LOG(INFO) << "Train NOR function with mlp." << std::endl;

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

  size_t num_features = training_sample_set_with_bias[0].GetInputVectorSize();
  size_t num_outputs = training_sample_set_with_bias[0].GetOutputVectorSize();
  MLP my_mlp({ num_features, 2 ,num_outputs }, { "sigmoid", "linear" });
  //Train MLP
  my_mlp.Train(training_sample_set_with_bias, 0.5, 500, 0.25);

  for (const auto & training_sample : training_sample_set_with_bias) {
    std::vector<double>  output;
    my_mlp.GetOutput(training_sample.input_vector(), &output);
    for (size_t i = 0; i < num_outputs; i++) {
      bool predicted_output = output[i] > 0.5 ? true : false;
      bool correct_output = training_sample.output_vector()[i] > 0.5 ? true : false;
      ASSERT_TRUE(predicted_output == correct_output);
    }
  }
  LOG(INFO) << "Trained with success." << std::endl;
}

UNIT(LearnXOR) {
  LOG(INFO) << "Train XOR function with mlp." << std::endl;

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

  size_t num_features = training_sample_set_with_bias[0].GetInputVectorSize();
  size_t num_outputs = training_sample_set_with_bias[0].GetOutputVectorSize();
  MLP my_mlp({ num_features, 2 ,num_outputs }, { "sigmoid", "linear" });
  //Train MLP
  my_mlp.Train(training_sample_set_with_bias, 0.5, 500, 0.25);

  for (const auto & training_sample : training_sample_set_with_bias) {
    std::vector<double>  output;
    my_mlp.GetOutput(training_sample.input_vector(), &output);
    for (size_t i = 0; i < num_outputs; i++) {
      bool predicted_output = output[i] > 0.5 ? true : false;
      bool correct_output = training_sample.output_vector()[i] > 0.5 ? true : false;
      ASSERT_TRUE(predicted_output == correct_output);
    }
  }
  LOG(INFO) << "Trained with success." << std::endl;
}

UNIT(LearnNOT) {
  LOG(INFO) << "Train NOT function with mlp." << std::endl;

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

  size_t num_features = training_sample_set_with_bias[0].GetInputVectorSize();
  size_t num_outputs = training_sample_set_with_bias[0].GetOutputVectorSize();
  MLP my_mlp({ num_features, 2 ,num_outputs }, { "sigmoid", "linear" });
  //Train MLP
  my_mlp.Train(training_sample_set_with_bias, 0.5, 500, 0.25);

  for (const auto & training_sample : training_sample_set_with_bias) {
    std::vector<double>  output;
    my_mlp.GetOutput(training_sample.input_vector(), &output);
    for (size_t i = 0; i < num_outputs; i++) {
      bool predicted_output = output[i] > 0.5 ? true : false;
      bool correct_output = training_sample.output_vector()[i] > 0.5 ? true : false;
      ASSERT_TRUE(predicted_output == correct_output);
    }
  }
  LOG(INFO) << "Trained with success." << std::endl;
}

UNIT(LearnX1) {
  LOG(INFO) << "Train X1 function with mlp." << std::endl;

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

  size_t num_features = training_sample_set_with_bias[0].GetInputVectorSize();
  size_t num_outputs = training_sample_set_with_bias[0].GetOutputVectorSize();
  MLP my_mlp({ num_features, 2 ,num_outputs }, { "sigmoid", "linear" });
  //Train MLP
  my_mlp.Train(training_sample_set_with_bias, 0.5, 500, 0.25);

  for (const auto & training_sample : training_sample_set_with_bias) {
    std::vector<double>  output;
    my_mlp.GetOutput(training_sample.input_vector(), &output);
    for (size_t i = 0; i < num_outputs; i++) {
      bool predicted_output = output[i] > 0.5 ? true : false;
      bool correct_output = training_sample.output_vector()[i] > 0.5 ? true : false;
      ASSERT_TRUE(predicted_output == correct_output);
    }
  }
  LOG(INFO) << "Trained with success." << std::endl;
}

UNIT(LearnX2) {
  LOG(INFO) << "Train X2 function with mlp." << std::endl;

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

  size_t num_features = training_sample_set_with_bias[0].GetInputVectorSize();
  size_t num_outputs = training_sample_set_with_bias[0].GetOutputVectorSize();
  MLP my_mlp({ num_features, 2 ,num_outputs }, { "sigmoid", "linear" });
  //Train MLP
  my_mlp.Train(training_sample_set_with_bias, 0.5, 500, 0.25);

  for (const auto & training_sample : training_sample_set_with_bias) {
    std::vector<double>  output;
    my_mlp.GetOutput(training_sample.input_vector(), &output);
    for (size_t i = 0; i < num_outputs; i++) {
      bool predicted_output = output[i] > 0.5 ? true : false;
      bool correct_output = training_sample.output_vector()[i] > 0.5 ? true : false;
      ASSERT_TRUE(predicted_output == correct_output);
    }
  }
  LOG(INFO) << "Trained with success." << std::endl;
}



UNIT(GetWeightsSetWeights) {
  LOG(INFO) << "Train X2 function, read internal weights" << std::endl;

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

  size_t num_features = training_sample_set_with_bias[0].GetInputVectorSize();
  size_t num_outputs = training_sample_set_with_bias[0].GetOutputVectorSize();
  MLP my_mlp({ num_features, 2, num_outputs }, { "sigmoid", "linear" });
  //Train MLP
  my_mlp.Train(training_sample_set_with_bias, 0.5, 500, 0.25);

  // get layer weights
  std::vector<std::vector<double>> weights = my_mlp.GetLayerWeights( 1 );

  for (const auto & training_sample : training_sample_set_with_bias) {
    std::vector<double>  output;
    my_mlp.GetOutput(training_sample.input_vector(), &output);
    for (size_t i = 0; i < num_outputs; i++) {
      bool predicted_output = output[i] > 0.5 ? true : false;
      std::cout << "PREDICTED OUTPUT IS NOW: " << output[i] << std::endl;
      bool correct_output = training_sample.output_vector()[i] > 0.5 ? true : false;
      ASSERT_TRUE(predicted_output == correct_output);
    }
  }

  // the expected value of the internal weights
  // after training are 1.65693 -0.538749
  ASSERT_TRUE(  1.6 <= weights[0][0] && weights[0][0] <=  1.7 );
  ASSERT_TRUE( -0.6 <= weights[0][1] && weights[0][1] <= -0.5 );

  // now, we are going to inject a weight value of 0.0
  // and check that the new output value is nonsense
  std::vector<std::vector<double>> zeroWeights = { { 0.0, 0.0 } };

  my_mlp.SetLayerWeights( 1, zeroWeights );

  for (const auto & training_sample : training_sample_set_with_bias) {
    std::vector<double>  output;
    my_mlp.GetOutput(training_sample.input_vector(), &output);
    for (size_t i = 0; i < num_outputs; i++) {
      ASSERT_TRUE( -0.0001L <= output[i] && output[i] <= 0.0001L );
    }
  }

  LOG(INFO) << "Trained with success." << std::endl;
}



int main(int argc, char* argv[]) {
  START_EASYLOGGINGPP(argc, argv);
  microunit::UnitTester::Run();
  return 0;
}
