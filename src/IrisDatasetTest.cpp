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

int inputNeuronNumber = 4;
int outputNeuronNumber = 3;
const long unsigned int outputNeuronNumberWithClass = 3;
std::string path = "./data/iris.mlp";

// Example illustrating practical use of this MLP lib.
// Disclaimer: This is NOT an example of good machine learning practices
//              regarding training/testing dataset partitioning.

const char *iris_dataset = "./data/iris.data";
const std::array<std::string, outputNeuronNumberWithClass> class_names =
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
  char line[16368];
  while (!feof(in) && fgets(line, 16368, in)) {
    ++(*samples);
  }
  fseek(in, 0, SEEK_SET);

  LOG(INFO) << "Loading " << (*samples)
    << " data points from " << iris_dataset << ".";
  // Allocate memory for input and output data.
  input->resize((*samples) * inputNeuronNumber);
  iris_class->resize((*samples) * outputNeuronNumber);

  // Read the file into our arrays. 
  int i, j;
  for (i = 0; i < (*samples); ++i) {
    double *p = &((*input)[0]) + i * inputNeuronNumber;
    double *c = &((*iris_class)[0]) + i * outputNeuronNumber;
    for (int k = 0; k < outputNeuronNumber; k++) {
			c[k] = 0.0;
		};

    fgets(line, 16368, in);

    char *split = strtok(line, ",");
    for (j = 0; j < inputNeuronNumber; ++j) {
      p[j] = atof(split);
      split = strtok(0, ",");
    }

    split[strlen(split) - 1] = 0;
    bool error = true;
		for (int j = 0; j < outputNeuronNumber; j++) {
			if (strcmp(split, class_names[j].c_str()) == 0) {
				error = false;
	      c[j] = 1.0;
	    }
		}
		if (error) {
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
    training_set_input.reserve(inputNeuronNumber);
    for (int i = 0; i < inputNeuronNumber; i++)
      training_set_input.push_back(*(&(input[0]) + j * inputNeuronNumber + i));
    training_set_output.reserve(outputNeuronNumber);
    for (int i = 0; i < outputNeuronNumber; i++)
      training_set_output.push_back(*(&(iris_class[0]) + j * outputNeuronNumber + i));
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

    my_mlp.SaveMLPNetwork(path);
  }
  //Destruction/Construction of a MLP object to show off saving and loading a trained model
  {
    MLP my_mlp(path);

    int correct = 0;
    for (int j = 0; j < samples; ++j) {
      std::vector<double> guess;
      my_mlp.GetOutput(training_sample_set_with_bias[j].input_vector(), &guess);
      size_t class_id;
      my_mlp.GetOutputClass(guess, &class_id);

      for (int i = 0; i < inputNeuronNumber; i++) {
        if (iris_class[j * outputNeuronNumber + i] == 1.0 && class_id == i) {
          ++correct;
        }
      }
    }
    LOG(INFO) << correct << "/" << samples
      << " (" << ((double)correct / samples * 100.0) << "%).";
  }
  return 0;
}
