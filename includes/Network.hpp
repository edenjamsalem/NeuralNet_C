#pragma once
#include "mnist/include/mnist/mnist_reader.hpp"
#include "flatArrays.tpp"
#include "utils.hpp"
#include "Eigen/Dense"

#ifndef HEADER_HPP
	#define MAX_LAYERS 4
	#define MINI_BATCH_SIZE 64
#endif

struct Layer {
	Eigen::VectorXf biases;
	Eigen::VectorXf activations;
	Eigen::MatrixXf weights;
};

class Network {
	private:
		size_t num_layers;
		size_t max_layer_len; 	// excluding input layer 
		std::vector<size_t> layer_sizes;

		Eigen::VectorXf input_layer;
		std::vector<Layer> network;

	public: 
		// Constructors
		Network(std::vector<size_t> &layer_sizes);
		Network(const std::vector<size_t> &layer_sizes);
		// ~Network();

		void SGD(mnist::MNIST_dataset<std::__1::vector, std::__1::vector<float, std::__1::allocator<float>>, uint8_t> dataset);
		void trainOn(std::vector<float> image, Eigen::VectorXf expected_ouput);
		void setInputs(std::vector<float> image);
		void feedForward();
		size_t calculateCost();
		void backProp();
};
