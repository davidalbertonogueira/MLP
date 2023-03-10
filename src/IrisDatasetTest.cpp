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
#include <array>
#include <algorithm>
#include "microunit.h"
#include "easylogging++.h"

INITIALIZE_EASYLOGGINGPP


// Example illustrating practical use of this MLP lib.
// Disclaimer: This is NOT an example of good machine learning practices
//              regarding training/testing dataset partitioning.

const int input_size = 4;
const int number_classes = 3;
#if defined(_WIN32)
const char *iris_dataset = "../../data/iris.data";
const std::string iris_mlp_weights = "../../data/iris.mlp";
#else
const char *iris_dataset = "./data/iris.data";
const std::string iris_mlp_weights = "./data/iris.mlp";
#endif
const std::array<std::string, number_classes> class_names =
{ "Iris-setosa", "Iris-versicolor", "Iris-virginica" };


bool load_data(int *samples,
  std::vector<double>  *input,
  std::vector<double> *iris_class) {
  // Load the iris data-set. 
  FILE *in = fopen(iris_dataset, "r");
  if (!in) {
    LOG(ERROR) << "Could not open file: " << iris_dataset << ".";
    return false;
  }

  // Loop through the data to get a count.
  char line[1024];
  while (!feof(in) && fgets(line, 1024, in)) {
    ++(*samples);
  }
  fseek(in, 0, SEEK_SET);

  LOG(INFO) << "Loading " << (*samples)
    << " data points from " << iris_dataset << ".";
  // Allocate memory for input and output data.
  input->resize((*samples) * input_size);
  iris_class->resize((*samples) * number_classes);

  // Read the file into our arrays. 
  int i, j;
  for (i = 0; i < (*samples); ++i) {
    double *p = &((*input)[0]) + i * input_size;
    double *c = &((*iris_class)[0]) + i * number_classes;
    for (int k = 0; k < number_classes; k++) {
      c[k] = 0.0;
    }

    fgets(line, 1024, in);

    char *split = strtok(line, ",");
    for (j = 0; j < 4; ++j) {
      p[j] = atof(split);
      split = strtok(NULL, ",");
    }

    if (strlen(split) >= 1 && split[strlen(split) - 1] == '\n')
      split[strlen(split) - 1] = '\0';
    if (strlen(split) >= 2 && split[strlen(split) - 2] == '\r')
      split[strlen(split) - 2] = '\0';

    if (strcmp(split, class_names[0].c_str()) == 0) {
      c[0] = 1.0;
    }
    else if (strcmp(split, class_names[1].c_str()) == 0) {
      c[1] = 1.0;
    }
    else if (strcmp(split, class_names[2].c_str()) == 0) {
      c[2] = 1.0;
    }
    else {
      LOG(ERROR) << "Unknown iris_class " << split
        << ".";
      return false;
    }
  }

  fclose(in);
  return true;
}


int main(int argc, char *argv[]) {
  LOG(INFO) << "Train MLP with IRIS dataset using backpropagation.";
  int samples = 0;
  std::vector<double> input;
  std::vector<double> iris_class;

  // Load the data from file.
  if (!load_data(&samples, &input, &iris_class)) {
    LOG(ERROR) << "Error processing input file.";
    return -1;
  }

  std::vector<TrainingSample> training_set;
  for (int j = 0; j < samples; ++j) {
    std::vector<double> training_set_input;
    std::vector<double> training_set_output;
    training_set_input.reserve(4);
    for (int i = 0; i < 4; i++)
      training_set_input.push_back(*(&(input[0]) + j * 4 + i));
    training_set_output.reserve(3);
    for (int i = 0; i < 3; i++)
      training_set_output.push_back(*(&(iris_class[0]) + j * 3 + i));
    training_set.emplace_back(std::move(training_set_input),
      std::move(training_set_output));
  }
  std::vector<TrainingSample> training_sample_set_with_bias(std::move(training_set));
  //set up bias
  for (auto & training_sample_with_bias : training_sample_set_with_bias) {
    training_sample_with_bias.AddBiasValue(1);
  }

  {
    // 4 inputs + 1 bias.
    // 1 hidden layer(s) of 4 neurons.
    // 3 outputs (1 per iris_class)
    MLP my_mlp({ 4 + 1, 4 ,3 }, { "sigmoid", "linear" }, false);

    int loops = 5000;

    // Train the network with backpropagation.
    LOG(INFO) << "Training for " << loops << " loops over data.";
    my_mlp.Train(training_sample_set_with_bias, .01, loops, 0.10, false);

    my_mlp.SaveMLPNetwork(iris_mlp_weights);
  }
  //Destruction/Construction of a MLP object to show off saving and loading a trained model
  {
    MLP my_mlp(iris_mlp_weights);

    int correct = 0;
    for (int j = 0; j < samples; ++j) {
      std::vector<double> guess;
      my_mlp.GetOutput(training_sample_set_with_bias[j].input_vector(), &guess);
      size_t class_id;
      my_mlp.GetOutputClass(guess, &class_id);

      if (iris_class[j * 3 + 0] == 1.0 && class_id == 0) {
        ++correct;
      }
      else if (iris_class[j * 3 + 1] == 1.0  && class_id == 1) {
        ++correct;
      }
      else if (iris_class[j * 3 + 2] == 1.0 && class_id == 2) {
        ++correct;
      }
    }
    LOG(INFO) << correct << "/" << samples
      << " (" << ((double)correct / samples * 100.0) << "%).";
  }
  return 0;
}
