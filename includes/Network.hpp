#pragma once
#include "mnist/include/mnist/mnist_reader.hpp"
#include "utils.hpp"
#include "Eigen/Dense"

struct LayerView {
	Eigen::Map<Eigen::MatrixXf> weights;
	Eigen::Map<Eigen::VectorXf> biases;
    Eigen::Map<Eigen::VectorXf> activations;

	LayerView(float *start, size_t r, size_t c) :
		weights(start, r, c),
		biases(start + (r * c), r),
		activations(start + (r * c) + r, r) 
		{
			weights.setRandom();
			biases.setRandom();
			activations.setZero();
		}
};

class NeuralNetwork {
	private:
		// Attributes
		size_t num_layers;
		std::vector<size_t> layer_sizes;
		std::unique_ptr<float[]> buffer;
		std::vector<LayerView> network;

		// Methods
		void feedForward(const std::vector<float> &image);
		void adjust_parameters(size_t currentBatchCost);
		void backProp();

	public: 
		// Constructors
		NeuralNetwork(std::vector<size_t> &layer_sizes);
		NeuralNetwork(const std::vector<size_t> &layer_sizes);

		// Methods
		void SGD(mnist::MNIST_dataset<std::__1::vector, std::__1::vector<float, std::__1::allocator<float>>, uint8_t> dataset);
};
