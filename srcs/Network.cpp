#include "../includes/header.hpp"

NeuralNetwork::NeuralNetwork(const std::vector<size_t> &layer_sizes) : 
	num_layers(layer_sizes.size() - 1),
	layer_sizes(layer_sizes),
	input_activations(layer_sizes[0])
{
	// allocate network buffer
	network.reserve(num_layers);
	size_t total_size = 0;

	for (size_t layer = 0; layer < num_layers; ++layer) {
		size_t rows = this->layer_sizes[layer + 1];
		size_t cols = this->layer_sizes[layer];
		total_size += (2 * rows * cols) + (3 * rows);
	}
	buffer = std::make_unique<float[]>(total_size);

	// assign map views of each layer
	float *start = this->buffer.get();
	for (size_t layer = 0; layer < num_layers; ++layer) {
		size_t rows = this->layer_sizes[layer + 1];
		size_t cols = this->layer_sizes[layer];

		network.emplace_back(LayerView(start, rows, cols));
		start += (2 * rows * cols) + (3 * rows);
	}
}

void NeuralNetwork::SGD(mnist::MNIST_dataset<std::__1::vector, std::__1::vector<float, std::__1::allocator<float>>, uint8_t> dataset) {
	Eigen::VectorXf expected_output(this->layer_sizes.back());
	const size_t miniBatchSize = 32;

	// Used to track improvement in network's accuracy on each batch
	size_t batchNumber = 1;
	float currentBatchCost = 0.0f;

	// train network on each image in the dataset
	for (size_t i = 0; i < dataset.training_images.size(); ++i) {
		this->feedForward(dataset.training_images[i]);

		// vectorize exepected output && calculate cost 
		expected_output.setZero();
		expected_output[dataset.training_labels[i]] = 1.0f;
		currentBatchCost += calculateCost(this->network.back().activations, expected_output);

		// backProp to calculate gradients for mini-batch
		this->backProp(expected_output);

		// adjust parameters after each mini-batch
		if ((i > 1 && i % miniBatchSize == 0)) {
			std::cout << "Mini-batch " << batchNumber << " cost: " << currentBatchCost / miniBatchSize << "\n";
			this->adjustNetwork(miniBatchSize);
			currentBatchCost = 0;
			batchNumber++;
		}
	}
}

void NeuralNetwork::feedForward(const std::vector<float> &image) {
	this->input_activations = Eigen::VectorXf(Eigen::Map<const Eigen::VectorXf>(image.data(), image.size()));
	Eigen::VectorXf activations = this->input_activations;
    
	for (auto &layer : this->network) {
        layer.activations.noalias() = (layer.weights * activations) + layer.biases;
        layer.activations = layer.activations.unaryExpr(&sigmoid);
        activations = layer.activations;
    }
}

void NeuralNetwork::backProp(Eigen::VectorXf &expected_output) {
	// calculate output delta (how much each node contributed to final cost)
    Eigen::VectorXf delta = (this->network.back().activations - expected_output).cwiseProduct(this->network.back().activations.unaryExpr(&sigmoidPrime));

	for (size_t layer = this->num_layers - 1; layer > 0; --layer) {
		// Previous activations: input image for first hidden layer
		const Eigen::VectorXf &prev_activations = (layer == 0) ? this->input_activations : this->network[layer - 1].activations;

		// Accumulate gradients
		this->network[layer].dW += delta * prev_activations.transpose();
		this->network[layer].db += delta;

		// δᶩ = (Wᵗ δˡ⁺¹) ⊙ σ'(aᶩ)
		Eigen::VectorXf sp = prev_activations.unaryExpr(&sigmoidPrime);
		delta = (this->network[layer].weights.transpose() * delta).cwiseProduct(sp);
	}
}

void NeuralNetwork::adjustNetwork(const size_t mini_batch_size) {
	// η => learning rate (how large a step we take along our gradient)
	const float η = 0.1f;

	for (auto &layer : this->network) {
		// apply changes to weights and biases 
		layer.weights -= η * (layer.dW / mini_batch_size);
		layer.biases -= η * (layer.db / mini_batch_size);
		
		// reset dW and db for next batch 
		layer.dW.setZero();
		layer.db.setZero();
    }
}
