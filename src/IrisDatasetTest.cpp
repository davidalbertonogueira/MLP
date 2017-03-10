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

const char *iris_dataset = "../../data/iris.data";
const std::array<std::string, 3> class_names =
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
  input->resize((*samples) * 4);
  iris_class->resize((*samples) * 3);

  // Read the file into our arrays. 
  int i, j;
  for (i = 0; i < (*samples); ++i) {
    double *p = &((*input)[0]) + i * 4;
    double *c = &((*iris_class)[0]) + i * 3;
    c[0] = c[1] = c[2] = 0.0;

    fgets(line, 1024, in);

    char *split = strtok(line, ",");
    for (j = 0; j < 4; ++j) {
      p[j] = atof(split);
      split = strtok(0, ",");
    }

    split[strlen(split) - 1] = 0;
    if (strcmp(split, class_names[0].c_str()) == 0) {
      c[0] = 1.0;
    } else if (strcmp(split, class_names[1].c_str()) == 0) {
      c[1] = 1.0;
    } else if (strcmp(split, class_names[2].c_str()) == 0) {
      c[2] = 1.0;
    } else {
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
    /* 4 inputs + 1 bias.
    * 1 hidden layer(s) of 4 neurons.
    * 3 outputs (1 per iris_class)
    */
    MLP my_mlp({ 4 + 1, 4 ,3 }, { "sigmoid", "linear" }, false);


    int loops = 5000;


    // Train the network with backpropagation.
    LOG(INFO) << "Training for " << loops << " loops over data.";
    my_mlp.UpdateMiniBatch(training_sample_set_with_bias, .01, loops, 0.10, false);

    my_mlp.SaveMLPNetwork(std::string("../../data/iris.mlp"));
  }
  //Destruction/Construction of a MLP object to show off saving and loading a trained model
  {
    MLP my_mlp(std::string("../../data/iris.mlp"));

    int correct = 0;
    for (int j = 0; j < samples; ++j) {
      std::vector<double> guess;
      my_mlp.GetOutput(training_sample_set_with_bias[j].input_vector(), &guess);
      size_t class_id;
      my_mlp.GetOutputClass(guess, &class_id);

      if (iris_class[j * 3 + 0] == 1.0 && class_id == 0) {
        ++correct;
      } else if (iris_class[j * 3 + 1] == 1.0  && class_id == 1) {
        ++correct;
      } else if (iris_class[j * 3 + 2] == 1.0 && class_id == 2) {
        ++correct;
      }
    }
    LOG(INFO) << correct << "/" << samples
      << " (" << ((double)correct / samples * 100.0) << "%).";
  }
  return 0;
}
