#pragma once
#include "mnist/include/mnist/mnist_reader.hpp"
#include "flatArrays.tpp"
#include "utils.hpp"

#ifndef NETWORK_HPP
	#define MAX_LAYERS 4
	#define MINI_BATCH_SIZE 64
#endif

struct Neuron {
	float bias;
	float signal;
    Neuron() : bias(gen_random_double()), signal(0.0) {}
};

class Network {
	private:
		size_t num_layers;
		size_t max_layer_len; 	// excluding input layer 
		std::vector<size_t> layer_sizes;

		// input values handled separately
		std::vector<float> input_layer;
		
		// rest of network is here
		flat2DArray<Neuron> network;
		flat2DArray<float> weights[MAX_LAYERS];

	public: 
		// Constructors
		Network(std::vector<size_t> &layer_sizes);
		Network(const std::vector<size_t> &layer_sizes);
		// ~Network();

		void SGD(mnist::MNIST_dataset<std::__1::vector, std::__1::vector<float, std::__1::allocator<float>>, uint8_t> dataset);
		void trainOn(std::vector<float> image, uint8_t expected_ouput);
		void setInputs(std::vector<float> image);
		void feedForward();
		void backProp();
};
