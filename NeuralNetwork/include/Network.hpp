#pragma once
#include "utils.hpp"
#include "Eigen/Dense"
#include <iostream>

struct LayerView {
	// view memory used for network
	Eigen::Map<Eigen::MatrixXf> weights;
	Eigen::Map<Eigen::VectorXf> biases;
    Eigen::Map<Eigen::VectorXf> activations;

	// view of memory used for backProp gradients 
	Eigen::Map<Eigen::MatrixXf> dW;
    Eigen::Map<Eigen::VectorXf> db;
    Eigen::Map<Eigen::VectorXf> delta;

	LayerView(float *start, size_t r, size_t c) :
		weights(start, r, c),
		biases(start + (r * c), r),
		activations(start + (r * c) + r, r) ,
		dW(start + (r * c) + (2 * r), r, c),
		db(start + (2 * r * c) + (2 * r), r),
		delta(start + (2 * r * c) + (3 * r), r)
		{
			weights.setRandom();
			weights *= std::sqrt(6.0f / (r + c));
			biases.setZero();
			activations.setZero();
			dW.setZero();
			db.setZero();
			delta.setZero();
		}
};

class NeuralNetwork {
	private:
		// Attributes
		std::unique_ptr<float[]> buffer;
		std::vector<LayerView> network;
		Eigen::VectorXf inputActivations;
		Eigen::VectorXf expectedOutput;
		
		// Constants
		const size_t miniBatchSize = 32; 
		const float η = 1.0f; // η => learning rate (how large a step we take along our gradient)
		const float scale;

		// Methods
		void _feedForward(const std::vector<float> &image);
		void _backProp();
		void _adjustNetwork();

	public: 
		// Constructors
		NeuralNetwork(std::vector<size_t> &layer_sizes);
		NeuralNetwork(const std::vector<size_t> &layer_sizes);

		// Methods
		void SGD(std::vector<std::vector<float>> &training_data, std::vector<uint8_t> &training_labels);
		float test(std::vector<std::vector<float>> &test_data, std::vector<uint8_t> &test_labels);
};
