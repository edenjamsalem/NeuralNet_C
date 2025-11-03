#pragma once
#include "flatArrays.tpp"
#include "utils.hpp"

struct Neuron {
	double bias;
	double signal;
    Neuron() : bias(gen_random_double()), signal(0.0) {}
};


class Network {
	private:
		size_t num_layers;
		size_t max_len; // exclusing input layer

		// input values handled separately as they are large
		std::vector<double> input_layer;
		flat2DArray<double> input_weights;
		
		// rest of network is here
		flat2DArray<Neuron> network;
		flat3DArray<double> weights;

	public: 
		Network(std::vector<size_t> &layer_sizes);
		Network(const std::vector<size_t> &layer_sizes);
	
	// ~Network();

};
