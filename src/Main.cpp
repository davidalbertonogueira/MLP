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
    {{ 0, 0 },{1,0}},
    {{ 0, 1 },{1,0}},
    {{ 1, 0 },{1,0}},
    {{ 1, 1 },{0,1}}
  };

  MLP my_mlp(2, 2, 1, 5, 0.1);
  my_mlp.Train(training_set, 100);

  for (const auto & training_sample : training_set){
    size_t class_id;
    my_mlp.GetOutputClass(training_sample.input_vector(), &class_id);
    ASSERT_TRUE(class_id == 
              std::distance(training_sample.output_vector().begin(), 
                            std::max_element(training_sample.output_vector().begin(),
                                           training_sample.output_vector().end()) ));
  }
  std::cout << "Trained with success." << std::endl;
  std::cout << std::endl;
}

UNIT(LearnNAND) {
  std::cout << "Train NAND function with mlp." << std::endl;

  std::vector<TrainingSample> training_set =
  {
    {{ 0, 0 },{0,1}},
    {{ 0, 1 },{0,1}},
    {{ 1, 0 },{0,1}},
    {{ 1, 1 },{1,0}}
  };

  MLP my_mlp(2, 2, 1, 5, 0.1);
  my_mlp.Train(training_set, 100);

  for (const auto & training_sample : training_set) {
    size_t class_id;
    my_mlp.GetOutputClass(training_sample.input_vector(), &class_id);
    ASSERT_TRUE(class_id ==
                std::distance(training_sample.output_vector().begin(),
                              std::max_element(training_sample.output_vector().begin(),
                                               training_sample.output_vector().end())));
  }
  std::cout << "Trained with success." << std::endl;
  std::cout << std::endl;
}

UNIT(LearnOR) {
  std::cout << "Train OR function with mlp." << std::endl;

  std::vector<TrainingSample> training_set =
  {
    {{ 0, 0 },{1,0}},
    {{ 0, 1 },{0,1}},
    {{ 1, 0 },{0,1}},
    {{ 1, 1 },{0,1}}
  };

  MLP my_mlp(2, 2, 1, 5, 0.1);
  my_mlp.Train(training_set, 100);

  for (const auto & training_sample : training_set) {
    size_t class_id;
    my_mlp.GetOutputClass(training_sample.input_vector(), &class_id);
    ASSERT_TRUE(class_id ==
                std::distance(training_sample.output_vector().begin(),
                              std::max_element(training_sample.output_vector().begin(),
                                               training_sample.output_vector().end())));
  }
  std::cout << "Trained with success." << std::endl;
  std::cout << std::endl;
}

UNIT(LearnNOR) {
  std::cout << "Train NOR function with mlp." << std::endl;

  std::vector<TrainingSample> training_set =
  {
    {{ 0, 0 },{0,1}},
    {{ 0, 1 },{1,0}},
    {{ 1, 0 },{1,0}},
    {{ 1, 1 },{1,0}}
  };

  MLP my_mlp(2, 2, 1, 5, 0.1);
  my_mlp.Train(training_set, 100);

  for (const auto & training_sample : training_set) {
    size_t class_id;
    my_mlp.GetOutputClass(training_sample.input_vector(), &class_id);
    ASSERT_TRUE(class_id ==
                std::distance(training_sample.output_vector().begin(),
                              std::max_element(training_sample.output_vector().begin(),
                                               training_sample.output_vector().end())));
  }
  std::cout << "Trained with success." << std::endl;
  std::cout << std::endl;
}

UNIT(LearnXOR) {
  std::cout << "Train XOR function with mlp." << std::endl;

  std::vector<TrainingSample> training_set =
  {
    { { 0, 0 },{ 1,0 } },
    { { 0, 1 },{ 0,1 } },
    { { 1, 0 },{ 0,1 } },
    { { 1, 1 },{ 1,0 } }
  };

  MLP my_mlp(2, 2, 1, 5, 0.1);
  my_mlp.Train(training_set, 100);

  for (const auto & training_sample : training_set) {
    size_t class_id;
    my_mlp.GetOutputClass(training_sample.input_vector(), &class_id);
    ASSERT_TRUE(class_id ==
                std::distance(training_sample.output_vector().begin(),
                              std::max_element(training_sample.output_vector().begin(),
                                               training_sample.output_vector().end())));
  }
  std::cout << "Trained with success." << std::endl;
  std::cout << std::endl;
}

UNIT(LearnNOT) {
  std::cout << "Train NOT function with mlp." << std::endl;

  std::vector<TrainingSample> training_set =
  {
    {{ 0},{0,1}},
    {{ 1},{1,1}}
  };

  MLP my_mlp(1, 2, 1, 5, 0.1);
  my_mlp.Train(training_set, 100);

  for (const auto & training_sample : training_set) {
    size_t class_id;
    my_mlp.GetOutputClass(training_sample.input_vector(), &class_id);
    ASSERT_TRUE(class_id ==
                std::distance(training_sample.output_vector().begin(),
                              std::max_element(training_sample.output_vector().begin(),
                                               training_sample.output_vector().end())));
  }
  std::cout << "Trained with success." << std::endl;
  std::cout << std::endl;
}

int main() {
  microunit::UnitTester::Run();
  return 0;
}