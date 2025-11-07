#pragma once
#include <random>
#include "Eigen/Dense"

float gen_random_double();
float sigmoid(float x);
float ReLU(float x);
float calculateCost(Eigen::VectorXf output, Eigen::VectorXf expected_ouput);
