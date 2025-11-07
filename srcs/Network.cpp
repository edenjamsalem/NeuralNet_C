#include "../includes/header.hpp"

NeuralNetwork::NeuralNetwork(const std::vector<size_t> &layer_sizes) : 
	num_layers(layer_sizes.size() - 1), 
	max_layer_len(*std::max_element(layer_sizes.begin() + 1, layer_sizes.end())),
	layer_sizes(layer_sizes),
	network(num_layers)
{
	for (size_t layer = 0; layer < this->num_layers; layer++) {
		size_t rows = layer_sizes[layer + 1];
		size_t cols = layer_sizes[layer];
		
		this->network[layer].biases.resize(rows);
		this->network[layer].activations.resize(rows);
		this->network[layer].weights.resize(rows, cols);

		this->network[layer].biases.setRandom();
		this->network[layer].activations.setZero();
		this->network[layer].weights.setRandom();
	}
}

void NeuralNetwork::SGD(mnist::MNIST_dataset<std::__1::vector, std::__1::vector<float, std::__1::allocator<float>>, uint8_t> dataset) {
	Eigen::VectorXf expected_output(this->layer_sizes.back());
	float currentBatchCost = 0.0f;

	// train network on each image in the dataset
	for (size_t i = 0; i < dataset.training_images.size(); i++) {
		this->feedForward(dataset.training_images[i]);

		// vectorize exepected output && calculate cost 
		expected_output.setZero();
		expected_output[dataset.training_labels[i]] = 1.0f;
		currentBatchCost += this->calculateCost(expected_output);

		// adjust parameters after each mini-batch
		if ((i > 1 && i % MINI_BATCH_SIZE == 0)) {
			this->adjust_parameters(currentBatchCost / MINI_BATCH_SIZE);
			currentBatchCost = 0;
		}
	}
}

void NeuralNetwork::feedForward(std::vector<float> image) {
	// a ==> current layer activations
	Eigen::VectorXf a = Eigen::VectorXf::Map(image.data(), image.size());

    for (size_t layer = 0; layer < this->num_layers; ++layer) {
        Eigen::VectorXf z = (this->network[layer].weights * a) + this->network[layer].biases;
        a = z.unaryExpr(&sigmoid);
        this->network[layer].activations = a;
    }	
}

float NeuralNetwork::calculateCost(Eigen::VectorXf expected_ouput) {
	Eigen::VectorXf output = this->network.back().activations;
	Eigen::VectorXf diff = output - expected_ouput;
	float cost = diff.squaredNorm() / diff.size();
	return (cost);
}

void NeuralNetwork::adjust_parameters(size_t currentBatchCost) {
	(void)currentBatchCost;
}

void NeuralNetwork::backProp() {

}
