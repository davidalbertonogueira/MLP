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
#include <cassert>

void LearnAND() {
  std::cout << "Train AND function with mlp." << std::endl;

  std::vector<TrainingSample> training_set =
  {
    {{ 1, 0, 0 },{1,0}},
    {{ 1, 0, 1 },{1,0}},
    {{ 1, 1, 0 },{1,0}},
    {{ 1, 1, 1 },{0,1}}
  };

  MLP my_mlp(0.1, 100, 0.5);
  my_mlp.Train(training_set, 1, 1);

  assert(my_mlp.GetOutput({ 1, 0, 0 }) == 0);
  assert(my_mlp.GetOutput({ 1, 0, 1 }) == 0);
  assert(my_mlp.GetOutput({ 1, 1, 0 }) == 0);
  assert(my_mlp.GetOutput({ 1, 1, 1 }) == 1);
  std::cout << "Trained with success." << std::endl;
  std::cout << std::endl;
}

void LearnNAND() {
  std::cout << "Train NAND function with mlp." << std::endl;

  std::vector<TrainingSample> training_set =
  {
    {{ 1, 0, 0 },{0,1}},
    {{ 1, 0, 1 },{0,1}},
    {{ 1, 1, 0 },{0,1}},
    {{ 1, 1, 1 },{1,0}}
  };

  MLP my_mlp(0.1, 100, 0.5);
  my_mlp.Train(training_set, 1, 1);

  assert(my_mlp.GetOutput({ 1, 0, 0 }) == 1);
  assert(my_mlp.GetOutput({ 1, 0, 1 }) == 1);
  assert(my_mlp.GetOutput({ 1, 1, 0 }) == 1);
  assert(my_mlp.GetOutput({ 1, 1, 1 }) == 0);
  std::cout << "Trained with success." << std::endl;
  std::cout << std::endl;
}

void LearnOR() {
  std::cout << "Train OR function with mlp." << std::endl;

  std::vector<TrainingSample> training_set =
  {
    {{ 1, 0, 0 },{1,0}},
    {{ 1, 0, 1 },{0,1}},
    {{ 1, 1, 0 },{0,1}},
    {{ 1, 1, 1 },{0,1}}
  };

  MLP my_mlp(0.1, 100, 0.5);
  my_mlp.Train(training_set, 1, 1);

  assert(my_mlp.GetOutput({ 1, 0, 0 }) == 0);
  assert(my_mlp.GetOutput({ 1, 0, 1 }) == 1);
  assert(my_mlp.GetOutput({ 1, 1, 0 }) == 1);
  assert(my_mlp.GetOutput({ 1, 1, 1 }) == 1);
  std::cout << "Trained with success." << std::endl;
  std::cout << std::endl;
}

void LearnNOR() {
  std::cout << "Train NOR function with mlp." << std::endl;

  std::vector<TrainingSample> training_set =
  {
    {{ 1, 0, 0 },{0,1}},
    {{ 1, 0, 1 },{1,0}},
    {{ 1, 1, 0 },{1,0}},
    {{ 1, 1, 1 },{1,0}}
  };

  MLP my_mlp(0.1, 100, 0.5);
  my_mlp.Train(training_set, 1, 1);

  assert(my_mlp.GetOutput({ 1, 0, 0 }) == 1);
  assert(my_mlp.GetOutput({ 1, 0, 1 }) == 0);
  assert(my_mlp.GetOutput({ 1, 1, 0 }) == 0);
  assert(my_mlp.GetOutput({ 1, 1, 1 }) == 0);
  std::cout << "Trained with success." << std::endl;
  std::cout << std::endl;
}

void LearnXOR() {
  std::cout << "Train XOR function with mlp." << std::endl;

  std::vector<TrainingSample> training_set =
  {
    { { 1, 0, 0 },{ 1,0 } },
    { { 1, 0, 1 },{ 0,1 } },
    { { 1, 1, 0 },{ 0,1 } },
    { { 1, 1, 1 },{ 1,0 } }
  };

  MLP my_mlp(0.1, 100, 0.5);
  my_mlp.Train(training_set, 1, 1);

  assert(my_mlp.GetOutput({ 1, 0, 0 }) == 0);
  assert(my_mlp.GetOutput({ 1, 0, 1 }) == 1);
  assert(my_mlp.GetOutput({ 1, 1, 0 }) == 1);
  assert(my_mlp.GetOutput({ 1, 1, 1 }) == 0);
  std::cout << "Trained with success." << std::endl;
  std::cout << std::endl;
}

void LearnNOT() {
  std::cout << "Train NOT function with mlp." << std::endl;

  std::vector<TrainingSample> training_set =
  {
    {{ 1, 0},{0,1}},
    {{ 1, 1},{1,1}}
  };

  MLP my_mlp(0.1, 100, 0.5);
  my_mlp.Train(training_set, 1, 1);

  assert(my_mlp.GetOutput({ 1, 0 }) == 1);
  assert(my_mlp.GetOutput({ 1, 1 }) == 0);
  std::cout << "Trained with success." << std::endl;
  std::cout << std::endl;
}

int main() {
  LearnAND();
  LearnNAND();
  LearnOR();
  LearnNOR();
  LearnXOR();
  LearnNOT();

  return 0;
}