#pragma once
#include "flatArrays.tpp"
#include "utils.hpp"

struct Neuron {
	double bias;
	double signal;
    Neuron() : bias(gen_random_double()), signal(0.0) {}
};

struct Network {
	size_t num_layers;
	size_t max_len;

	// input values and weights heap allocated as they are potentially large and variable
	std::vector<double> input_layer;
	flat2DArray<double> input_weights;
	
	// rest of network is stack allocated
	flat2DArray<Neuron> network;
	flat3DArray<double> weights;

	Network(std::vector<double> &inputs, std::vector<size_t> &layer_sizes);
	Network(std::vector<double> &inputs, const std::vector<size_t> &layer_sizes);
	// ~Network();

};
