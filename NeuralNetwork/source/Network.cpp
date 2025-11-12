#include "../include/Network.hpp"

NeuralNetwork::NeuralNetwork(const std::vector<size_t> &layout) : 
	inputActivations(layout.front()),
	expectedOutput(layout.back()),
	scale(η / miniBatchSize)
{
	// validate network size
	if (layout.size() < 3) {
		throw std::invalid_argument("Network must have at least 1 hidden layer.");
	}

	// allocate network buffer
	network.reserve(layout.size() - 1);
	size_t total_size = 0;

	for (size_t layer = 0; layer < layout.size() - 1; ++layer) {
		size_t rows = layout[layer + 1];
		size_t cols = layout[layer];
		total_size += (2 * rows * cols) + (4 * rows);
	}
	buffer = std::make_unique<float[]>(total_size);

	// assign map views of each layer
	float *start = this->buffer.get();
	for (size_t layer = 0; layer < layout.size() - 1; ++layer) {
		size_t rows = layout[layer + 1];
		size_t cols = layout[layer];

		network.emplace_back(LayerView(start, rows, cols));
		start += (2 * rows * cols) + (4 * rows);
	}
}

NeuralNetwork::NeuralNetwork(std::vector<size_t> &layer_sizes) 
	: NeuralNetwork(static_cast<const std::vector<size_t>&>(layer_sizes)) 
{}

void NeuralNetwork::SGD(std::vector<std::vector<float>> &trainingData, std::vector<uint8_t> &trainingLabels) {
	// Used to track network's improvement across batches
	size_t progressBar = 1;
	size_t progressBarMax = 40;
	// size_t batchNumber = 0;
	// float currentBatchCost = 0.0f;

	// train network on each image in the dataset
	for (size_t i = 0; i < trainingData.size(); ++i) {
		this->_feedForward(trainingData[i]);

		// vectorize exepected output 
		this->expectedOutput.setZero();
		this->expectedOutput[trainingLabels[i]] = 1.0f;

		// currentBatchCost += calculateCost(this->network.back().activations, this->expectedOutput);
		this->_backProp();

		// adjust parameters after each mini-batch
		if ((i > 1 && i % this->miniBatchSize == 0) || i == trainingData.size() - 1) {
			// std::cout << "Mini-batch " << batchNumber++ << " cost: " << currentBatchCost / miniBatchSize << "\n";
			// currentBatchCost = 0;
			this->_adjustNetwork();

			// update progress bar
			if (i >= (progressBar * trainingData.size() / progressBarMax)) {
				std::cout << "|";
				std::cout.flush();
				progressBar++;
			}
		}
	}
	std::cout << std::endl;
}

void NeuralNetwork::_feedForward(const std::vector<float> &image) {
	// copy inputs so original image is not overwritten
	this->inputActivations = Eigen::VectorXf(Eigen::Map<const Eigen::VectorXf>(image.data(), image.size()));
	Eigen::VectorXf activations = this->inputActivations;
    
	for (auto &layer : this->network) {
        layer.activations.noalias() = (layer.weights * activations) + layer.biases;
        layer.activations = layer.activations.unaryExpr(&sigmoid);
        activations = layer.activations;
    }
}

void NeuralNetwork::_backProp() {
	// Calculate output delta (how much each node contributed to final cost)
    this->network.back().delta = (this->network.back().activations - this->expectedOutput).cwiseProduct(this->network.back().activations.unaryExpr(&sigmoidPrime));

	// Backpropagate delta
	for (int layer = static_cast<int>(this->network.size()) - 2; layer >= 0; --layer) {
		LayerView &current = this->network[layer];
		LayerView &next = this->network[layer + 1];
		current.delta = (next.weights.transpose() * next.delta).cwiseProduct(current.activations.unaryExpr(&sigmoidPrime));
	}

	// Compute Gradients
	Eigen::VectorXf prevActivations = this->inputActivations;
	for (auto &layer : this->network) {
		layer.dW += layer.delta * prevActivations.transpose();
		layer.db += layer.delta;
		prevActivations = layer.activations;
	}
}

void NeuralNetwork::_adjustNetwork() {
	for (auto &layer : this->network) {
		// apply changes to weights and biases 
		layer.weights.noalias() -= this->scale * layer.dW;
		layer.biases.noalias() -= this->scale * layer.db;
		
		// reset for next batch 
		layer.dW.setZero();
		layer.db.setZero();
		layer.delta.setZero();
    }
}

float NeuralNetwork::test(std::vector<std::vector<float>> &test_data, std::vector<uint8_t> &test_labels) {
	size_t successCount = 0;
	Eigen::Index prediction;

	for (size_t i = 0; i < test_data.size(); i++) {
		// feed each image into the network
		this->_feedForward(test_data[i]);

		// compare predicted output to expected value
		this->network.back().activations.maxCoeff(&prediction);
		successCount += (static_cast<uint8_t>(prediction) == test_labels[i]);

		// std::cout << "Prediction: " << prediction << "\n";
		// std::cout << "Expected: " << static_cast<int>(test_labels[i]) << "\n";
	}
	return (static_cast<float>(successCount) / test_data.size());
}
