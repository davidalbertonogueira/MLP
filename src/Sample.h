#ifndef TRAININGSAMPLE_H
#define TRAININGSAMPLE_H

#include <stdlib.h>
#include <vector>

class Sample {
public:
  Sample(const std::vector<double> & input_vector) {

    m_input_vector = input_vector;
  }
  std::vector<double> & input_vector() {
    return m_input_vector;
  }
  uint32_t GetInputVectorSize() const {
    return m_input_vector.size();
  }
  void AddBiasValue(double bias_value) {
    m_input_vector.insert(m_input_vector.begin(), bias_value);
  }
protected:
  std::vector<double> m_input_vector;
};


class TrainingSample : public Sample {
public:
  TrainingSample(const std::vector<double> & input_vector,
                 const std::vector<double> & output_vector) :
    Sample(input_vector) {
    m_output_vector = output_vector;
  }
  std::vector<double> & output_vector() { 
    return m_output_vector; 
  }
  uint32_t GetOutputVectorSize() const {
    return m_output_vector.size();
  }
protected:
  std::vector<double> m_output_vector;
};


#endif // TRAININGSAMPLE_H