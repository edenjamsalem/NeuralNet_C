#include "../include/Network.hpp"

NeuralNetwork::NeuralNetwork(const std::vector<size_t> &layout) : 
	inputActivations(layout.front()),
	expectedOutput(layout.back()),
	miniBatchSize(32)
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
		total_size += (2 * rows * cols) + (3 * rows);
	}
	buffer = std::make_unique<float[]>(total_size);

	// assign map views of each layer
	float *start = this->buffer.get();
	for (size_t layer = 0; layer < layout.size() - 1; ++layer) {
		size_t rows = layout[layer + 1];
		size_t cols = layout[layer];

		network.emplace_back(LayerView(start, rows, cols));
		start += (2 * rows * cols) + (3 * rows);
	}
}

NeuralNetwork::NeuralNetwork(std::vector<size_t> &layer_sizes) 
	: NeuralNetwork(static_cast<const std::vector<size_t>&>(layer_sizes)) 
{}

void NeuralNetwork::SGD(std::vector<std::vector<float>> trainingData, std::vector<uint8_t> trainingLabels) {
	// Used to track network's improvement across batches
	size_t batchNumber = 0;
	float currentBatchCost = 0.0f;

	// train network on each image in the dataset
	for (size_t i = 0; i < trainingData.size(); ++i) {
		this->feedForward(trainingData[i]);

		// vectorize exepected output 
		this->expectedOutput.setZero();
		this->expectedOutput[trainingLabels[i]] = 1.0f;

		currentBatchCost += calculateCost(this->network.back().activations, this->expectedOutput);
		this->backProp();

		// adjust parameters after each mini-batch
		if ((i > 1 && i % this->miniBatchSize == 0) || i == trainingData.size() - 1) {
			std::cout << "Mini-batch " << batchNumber++ << " cost: " << currentBatchCost / miniBatchSize << "\n";
			this->adjustNetwork();
			currentBatchCost = 0;
		}
	}
}

void NeuralNetwork::feedForward(const std::vector<float> &image) {
	// copy inputs so original image is not overwritten
	this->inputActivations = Eigen::VectorXf(Eigen::Map<const Eigen::VectorXf>(image.data(), image.size()));
	Eigen::VectorXf &activations = this->inputActivations;
    
	for (auto &layer : this->network) {
        layer.activations.noalias() = (layer.weights * activations) + layer.biases;
        layer.activations = layer.activations.unaryExpr(&sigmoid);
        activations = layer.activations;
    }
}

void NeuralNetwork::backProp() {
	// calculate output delta (how much each node contributed to final cost)
    Eigen::VectorXf delta = (this->network.back().activations - this->expectedOutput).cwiseProduct(this->network.back().activations.unaryExpr(&sigmoidPrime));

	for (size_t layer = this->network.size() - 1; layer > 0; --layer) {
		// Previous activations: input image for first hidden layer
		const Eigen::VectorXf &prev_activations = (layer == 0) ? this->inputActivations : this->network[layer - 1].activations;

		// Accumulate gradients
		this->network[layer].dW += delta * prev_activations.transpose();
		this->network[layer].db += delta;

		// δᶩ = (Wᵗ δˡ⁺¹) ⊙ σ'(aᶩ)
		Eigen::VectorXf sp = prev_activations.unaryExpr(&sigmoidPrime);
		delta = (this->network[layer].weights.transpose() * delta).cwiseProduct(sp);
	}
}

void NeuralNetwork::adjustNetwork() {
	// η => learning rate (how large a step we take along our gradient)
	const float η = 0.1f;

	for (auto &layer : this->network) {
		// apply changes to weights and biases 
		layer.weights -= η * (layer.dW / this->miniBatchSize);
		layer.biases -= η * (layer.db / this->miniBatchSize);
		
		// reset dW and db for next batch 
		layer.dW.setZero();
		layer.db.setZero();
    }
}
