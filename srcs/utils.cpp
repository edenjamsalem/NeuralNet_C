#include "../includes/utils.hpp"

float gen_random_double() {
    static std::mt19937 gen(std::random_device{}());
    static std::uniform_real_distribution<float> dist(-1.0, 1.0);
    return dist(gen);
}

float sigmoid(float x) {
    return (1.0 / (1.0 + std::exp(-x)));
}

float ReLU(float x) {
    return (x < 0.0 ? 0.0 : x);
}

float calculateCost(Eigen::VectorXf output, Eigen::VectorXf expected_ouput) {
	Eigen::VectorXf diff = output - expected_ouput;
	float cost = diff.squaredNorm() / diff.size();
	return (cost);
}