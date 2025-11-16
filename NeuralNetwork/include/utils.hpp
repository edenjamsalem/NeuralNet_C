#pragma once
#include <random>
#include "Eigen/Dense"

float gen_random_double();
float sigmoid(float x);
float sigmoidPrime(float a);
float ReLU(float x);
float ReLUPrime(float x);
Eigen::VectorXf softmax(const Eigen::VectorXf &output);
float calculateCost(const Eigen::VectorXf &output, const Eigen::VectorXf &expected_ouput);
