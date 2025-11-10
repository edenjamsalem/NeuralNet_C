#pragma once
#include "mnist/include/mnist/mnist_reader.hpp"
#include "utils.hpp"
#include "Eigen/Dense"

struct LayerView {
	// view memory used for network
	Eigen::Map<Eigen::MatrixXf> weights;
	Eigen::Map<Eigen::VectorXf> biases;
    Eigen::Map<Eigen::VectorXf> activations;

	// view of memory used for backProp gradients 
	Eigen::Map<Eigen::MatrixXf> dW;
    Eigen::Map<Eigen::VectorXf> db;

	LayerView(float *start, size_t r, size_t c) :
		weights(start, r, c),
		biases(start + (r * c), r),
		activations(start + (r * c) + r, r) ,
		dW(start + (r * c) + (2 * r), r, c),
		db(start + (2 * r * c) + (2 * r), r)
		{
			weights.setRandom();
			biases.setRandom();
			activations.setZero();
			dW.setZero();
			db.setZero();
		}
};

class NeuralNetwork {
	private:
		// Attributes
		size_t num_layers;
		std::vector<size_t> layer_sizes;
		std::unique_ptr<float[]> buffer;
		std::vector<LayerView> network;
		Eigen::VectorXf input_activations;

		// Methods
		void feedForward(const std::vector<float> &image);
		void backProp(Eigen::VectorXf &expected_output);
		void adjustNetwork(const size_t mini_batch_size);

	public: 
		// Constructors
		NeuralNetwork(std::vector<size_t> &layer_sizes);
		NeuralNetwork(const std::vector<size_t> &layer_sizes);

		// Methods
		void SGD(std::vector<std::vector<float>> training_data, std::vector<uint8_t> training_labels);
};
