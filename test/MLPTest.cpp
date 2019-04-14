//============================================================================
// Name : Main.cpp
// Author : David Nogueira
//============================================================================
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <algorithm>
#include "microunit.h"
#include "easylogging++.h"

#include "MLP.h"
#include "MlpInspectorDummy.h"

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


UNIT(LearnPatternWithSimpleInspector){

    LOG(INFO) << "The  table" << std::endl;

    std::vector<TrainingSample> training_set =
    {
    {{ 0.31, 0.15, 0.52, 0.01, 0.34, 0.76}, { 0 } },
    {{ 0.23, 0.01, 0.72, 0.34, 0.51, 0.02}, { 0 } },
    {{ 0.97, 0.47, 0.76, 0.74, 0.74, 0.79}, { 1 } },
    {{ 0.17, 0.27, 0.30, 0.61, 0.91, 0.19}, { 0 } },
    {{ 0.63, 0.68, 0.56, 0.42, 0.64, 0.98}, { 1 } },
    {{ 0.89, 0.32, 0.36, 0.43, 0.73, 0.51}, { 0 } },
    {{ 0.38, 0.25, 0.45, 0.52, 0.93, 0.64}, { 0 } },
    {{ 0.70, 0.14, 0.07, 0.62, 0.42, 0.65}, { 0 } },
    {{ 0.21, 0.94, 0.30, 0.17, 0.82, 0.36}, { 1 } },
    {{ 0.19, 0.45, 0.81, 0.31, 0.24, 0.75}, { 1 } },
    {{ 0.00, 0.31, 0.64, 1.00, 0.55, 0.27}, { 0 } },
    {{ 0.94, 0.01, 0.58, 0.78, 0.17, 0.44}, { 1 } },
    {{ 0.90, 0.12, 0.50, 0.52, 0.77, 0.80}, { 0 } },
    {{ 0.69, 0.30, 0.19, 0.91, 0.45, 0.50}, { 0 } },
    {{ 0.17, 0.29, 0.46, 0.58, 0.87, 0.61}, { 0 } },
    {{ 0.63, 0.39, 0.30, 0.49, 0.51, 0.49}, { 0 } },
    {{ 0.89, 0.14, 0.65, 0.51, 0.42, 0.19}, { 1 } },
    {{ 1.00, 0.13, 0.31, 0.91, 0.56, 0.29}, { 1 } },
    {{ 0.71, 0.11, 0.82, 0.16, 0.10, 0.80}, { 1 } },
    {{ 0.74, 0.49, 0.18, 0.83, 0.57, 0.11}, { 0 } },
    {{ 0.18, 0.62, 0.79, 0.47, 0.03, 0.17}, { 1 } },
    {{ 0.56, 0.94, 0.29, 0.90, 0.72, 0.15}, { 1 } },
    {{ 0.81, 0.24, 0.51, 0.56, 0.76, 0.68}, { 0 } },
    {{ 1.00, 0.30, 0.80, 0.48, 0.37, 0.45}, { 1 } },
    {{ 0.10, 0.21, 0.61, 0.52, 0.62, 0.75}, { 0 } },
    {{ 0.50, 0.89, 0.10, 0.57, 0.57, 0.24}, { 0 } },
    {{ 0.75, 0.73, 0.09, 0.48, 0.73, 0.32}, { 1 } },
    {{ 0.23, 0.18, 0.55, 0.54, 0.97, 0.96}, { 0 } },
    {{ 0.22, 0.12, 0.59, 0.47, 0.14, 0.05}, { 0 } },
    {{ 0.88, 0.04, 0.45, 0.91, 0.52, 0.65}, { 0 } },
    {{ 0.13, 0.20, 0.99, 0.33, 0.60, 0.13}, { 0 } },
    {{ 0.40, 0.08, 0.67, 0.29, 0.27, 0.93}, { 0 } },
    {{ 0.22, 0.63, 0.73, 0.14, 0.88, 0.03}, { 1 } },
    {{ 0.49, 0.91, 0.33, 0.55, 0.72, 0.53}, { 0 } }
    };
    std::vector<TrainingSample> training_sample_set_with_bias(training_set);
    //set up bias
    for (auto & training_sample_with_bias : training_sample_set_with_bias) {
        training_sample_with_bias.AddBiasValue(1);
    }

    size_t input_vector_size = training_sample_set_with_bias[0].GetInputVectorSize();
    size_t output_vector_size = training_sample_set_with_bias[0].GetOutputVectorSize();
    std::vector<std::string> activ_functions_per_layer = { "sigmoid", "sigmoid"};
    bool use_constant_weight_init = false;
    bool use_softmax_on_each_output = false;
    MLP my_mlp( {input_vector_size, 2, output_vector_size},
            activ_functions_per_layer,
            use_constant_weight_init,
            use_softmax_on_each_output );
    std::shared_ptr<MlpInspectorDummy> inspectorDummy {new MlpInspectorDummy()};

    my_mlp.AddInspector( inspectorDummy );

    my_mlp.Train( training_sample_set_with_bias,
                    0.5L,
                    500,
                    0.25,
                    false );

    ASSERT_TRUE( inspectorDummy->getOnEnter() > 0 );
    ASSERT_TRUE( inspectorDummy->getOnEnd() > 0 );
    ASSERT_TRUE( inspectorDummy->getOnEnter() == inspectorDummy->getOnEnd() );

    ASSERT_TRUE( inspectorDummy->getOnBefore() == (long) training_set.size() * 500 );
    ASSERT_TRUE( inspectorDummy->getOnAfter() == (long) training_set.size() * 500  );
    ASSERT_TRUE( inspectorDummy->getOnBefore() == inspectorDummy->getOnAfter() );

    for( const auto & training_sample : training_sample_set_with_bias ){
        std::vector<double> output;
        my_mlp.GetOutput( training_sample.input_vector(), &output );
        for( size_t i = 0; i < output_vector_size; i++ ){
            LOG(INFO) << "OUTPUT: "
                      << output[i]
                      << ", DESIRED: "
                      << training_sample.output_vector()[i];
        }
    }

}


int main(int argc, char* argv[]) {
  START_EASYLOGGINGPP(argc, argv);
  microunit::UnitTester::Run();
  return 0;
}
