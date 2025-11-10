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
	float currentBatchCost = 0.0f; // only used to track improvements in network's accuracy 
	(void)currentBatchCost;
	const size_t mini_batch_size = 64;

	// train network on each image in the dataset
	for (size_t i = 0; i < dataset.training_images.size(); ++i) {
		this->feedForward(dataset.training_images[i]);

		// vectorize exepected output && calculate cost 
		expected_output.setZero();
		expected_output[dataset.training_labels[i]] = 1.0f;
		currentBatchCost += calculateCost(this->network.back().activations, expected_output);

		// backProp to calculate gradients for mini-batch
		this->backProp();

		// adjust parameters after each mini-batch
		if ((i > 1 && i % mini_batch_size == 0)) {
			this->adjustNetwork();
			currentBatchCost = 0;
		}
	}
}

void NeuralNetwork::feedForward(const std::vector<float> &image) {
	this->input_activations = Eigen::VectorXf::Map(image.data(), image.size());
	Eigen::VectorXf activations = this->input_activations;
    
	for (auto &layer : this->network) {
        layer.activations = (layer.weights * activations) + layer.biases;
        layer.activations = layer.activations.unaryExpr(&sigmoid);
        activations = layer.activations;
    }
}

void NeuralNetwork::backProp() {

}

// η => learning rate (how large a step we take along our gradient)
void NeuralNetwork::adjustNetwork() {
	// const float η = 0.1f;
	// for (size_t layer = this->layer_sizes.back(); layer > 0; --layer) {

	// }
}
